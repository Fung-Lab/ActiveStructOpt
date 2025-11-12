from activestructopt.common.constraints import lj_rmins, lj_repulsion_mt, lj_reject
from activestructopt.model.base import BaseModel
from activestructopt.dataset.base import BaseDataset
from activestructopt.objective.base import BaseObjective
from activestructopt.optimizer.base import BaseOptimizer
from activestructopt.sampler.base import BaseSampler
from activestructopt.common.registry import registry
from pymatgen.transformations.standard_transformations import RotationTransformation
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import IStructure
from pymatgen.core import Structure, Lattice
from ase import Atoms
from ase.optimize import FIRE
from ase.constraints import ExpCellFilter
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
import torch
import numpy as np

# Glass, Colin W., Artem R. Oganov, and Nikolaus Hansen. 
# USPEX—Evolutionary crystal structure prediction." 
# Computer physics communications 175, no. 11-12 (2006): 713-720.

def get_lr_rotated_struct(s):
  new_s = s.copy()
  # Align first vector to x-axis
  if new_s.lattice.matrix[0][0] != 0 or new_s.lattice.matrix[0][1] != 0:
    axis_to_rotate = np.cross(new_s.lattice.matrix[0], [1.0, 0.0, 0.0])
    angle_to_rotate = np.arccos(np.dot(new_s.lattice.matrix[0], 
                [1.0, 0.0, 0.0]) / np.linalg.norm(new_s.lattice.matrix[0]))
    rt = RotationTransformation(axis_to_rotate, angle_to_rotate, 
                                angle_in_radians = True)
    new_s = rt.apply_transformation(new_s)

  # Align second vector to xy-plane
  angle_to_rotate = -np.arctan2(new_s.lattice.matrix[1][2], 
                                new_s.lattice.matrix[1][1])
  rt = RotationTransformation([1, 0, 0], angle_to_rotate, 
                              angle_in_radians = True)
  new_s = rt.apply_transformation(new_s)
  
  return new_s

def uspex_select_weights(pop, objectives, survival = 0.6, p = 1):
  Nsurv = int(np.floor(pop * survival))
  # https://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy-without-sorting-array-twice
  ranks = torch.argsort(torch.argsort(objectives)).detach().numpy()
  weights = np.clip(Nsurv - ranks, a_min = 0) ** p
  return weights / np.sum(weights)
  

def uspex_heredity(struct1, struct2, shift1prob = 1.0, shift2prob = 0.05, 
                  add_by_density = True):
  s1 = struct1.copy()
  s2 = struct2.copy()

  achi = np.random.choice(3)
  achx = np.random.rand()
  latw = np.random.rand()

  s1 = s1.translate_sites(range(len(s1)), [
    np.random.rand() if (np.random.rand() < (
        shift1prob if achi == 0 else shift2prob)) else 0.0, 
    np.random.rand() if (np.random.rand() < (
        shift1prob if achi == 1 else shift2prob)) else 0.0, 
    np.random.rand() if (np.random.rand() < (
        shift1prob if achi == 2 else shift2prob)) else 0.0], 
    frac_coords = True)
  
  s1 = get_lr_rotated_struct(s1)
  s2 = get_lr_rotated_struct(s2)

  new_lat_mat = latw * s1.lattice.matrix + (1 - latw) * s2.lattice.matrix 

  new_struct = Structure(new_lat_mat, [], [])

  for i in range(len(s1)):
    if (s1.sites[i].frac_coords[achi] % 1.0) < achx:
      new_struct = new_struct.append(s1.sites[i].species, 
        s1.sites[i].frac_coords, coords_are_cartesian = False)

  for i in range(len(s2)):
    if (s2.sites[i].frac_coords[achi] % 1.0) > achx:
      new_struct = new_struct.append(s2.sites[i].species, 
        s2.sites[i].frac_coords, coords_are_cartesian = False)

  comp_to_match = s1.composition.as_dict()

  for k in comp_to_match.keys():
    if comp_to_match[k] < new_struct.composition.as_dict()[k]:
      n_to_remove = new_struct.composition.as_dict()[k] - comp_to_match[k]
      is_to_remove = np.random.choice(np.where(
        [s.species_string == k for s in new_struct.sites])[0], 
        int(n_to_remove), replace = False)
      new_struct = new_struct.remove_sites(is_to_remove)

    if comp_to_match[k] > new_struct.composition.as_dict()[k]:
      n_to_add = comp_to_match[k] - new_struct.composition.as_dict()[k]

      added = 0

      while added < int(n_to_add):
        if add_by_density:
          raise NotImplementedError
        else:
          interval_check = 1 - achx
        
        choose_from_s1 = np.random.rand() < interval_check
        new_site_i = np.random.choice(np.where(
          [s.species_string == k for s in (
          s1 if choose_from_s1 else s2).sites])[0])
        new_site_coords = (s1 if choose_from_s1 else s2
                            ).sites[new_site_i].frac_coords
        new_site_spec = (s1 if choose_from_s1 else s2
                          ).sites[new_site_i].species

        not_found_new_site = np.min([np.linalg.norm(
          s.frac_coords - new_site_coords) for s in new_struct.sites]
          ) > 1e-6
        if not_found_new_site:
          new_struct = new_struct.append(new_site_spec, 
              new_site_coords, coords_are_cartesian = False)
          added += 1

  return new_struct

def uspex_mutation(struct, σl = 0.7):
  new_struct = struct.copy()
  strain_mat = np.array([[1 + np.random.normal(scale = σl), 
              np.random.normal(scale = σl) / 2, 
              np.random.normal(scale = σl) / 2], 
            [np.random.normal(scale = σl) / 2, 
              1 + np.random.normal(scale = σl), 
              np.random.normal(scale = σl) / 2], 
            [np.random.normal(scale = σl) / 2, 
              np.random.normal(scale = σl) / 2, 
              1 + np.random.normal(scale = σl)]])
  new_struct.lattice = Lattice(np.matmul(strain_mat, 
                                          new_struct.lattice.matrix))
  return new_struct

def uspex_permutation(struct, nperms = 3):
  # Glass, Colin W., Artem R. Oganov, and Nikolaus Hansen. 
  # USPEX—Evolutionary crystal structure prediction." 
  # Computer physics communications 175, no. 11-12 (2006): 713-720.
  new_struct = struct.copy()

  for _ in range(nperms):
    sp_in_sys = list(new_struct.composition.as_dict().keys())
    sp_to_swap = np.random.choice(len(sp_in_sys), 2, replace = False)
    site_i = np.random.choice(np.where(
      [s.species_string == sp_in_sys[sp_to_swap[0]
                                    ] for s in new_struct.sites])[0])
    site_j = np.random.choice(np.where(
      [s.species_string == sp_in_sys[sp_to_swap[1]
                                    ] for s in new_struct.sites])[0])
    sp_i = new_struct.sites[site_i].species
    sp_j = new_struct.sites[site_j].species
    new_struct = new_struct.replace(site_i, sp_j)
    new_struct = new_struct.replace(site_j, sp_i)

  return new_struct


@registry.register_optimizer("USPEX")
class USPEX(BaseOptimizer):
  def __init__(self) -> None:
    pass
  
  def run(self, model: BaseModel, dataset: BaseDataset, 
    objective: BaseObjective, sampler: BaseSampler, 
    pop = 64, gens = 100, optimize_atoms = True, 
    optimize_lattice = True, save_obj_values = False, 
    constraint_scale = 1.0, constraint_buffer = 0.85, 
    random_starts = False, fmax = 0.01, nmax = 100, 
    survival = 0.6, select_p = 1, p_her = 0.85, p_mut = 0.1,
    shift1prob = 1.0, shift2prob = 0.05, σl = 0.7, nperms = 3,
    w_adapt = 0.5, N_adapt = 4, 
    **kwargs) -> IStructure:
    
    adaptor = AseAtomsAdaptor()

    population = [sampler.sample(
      ) for _ in range(pop)] if random_starts else [dataset.structures[
      j].copy() if j < dataset.N else sampler.sample(
      ) for j in range(pop)]

    obj_values = torch.zeros((gens, pop), device = 'cpu'
      ) if save_obj_values else None
    
    device = model.device
    orbff = pretrained.orb_v3_conservative_inf_omat(
        device=device,
        precision="float32-high",   # or "float32-highest" / "float64
    )
    calc = ORBCalculator(orbff, device=device)

    natoms = len(population[0])
    ljrmins = torch.tensor(lj_rmins, device = device) * constraint_buffer
    best_obj = torch.tensor([float('inf')], device = device)
    best_struct = population[0].copy()
    target = torch.tensor(dataset.target, device = device)
    
    split = int(np.ceil(np.log2(pop)))
    orig_split = split

    #print("starting loop")

    Vuc = dataset.structures[0].volume

    for i in range(gens):
      predicted = False
      # Local Energy Optimization (TODO: Make this parallel)

      ## TEST ##
      from pymatgen.io.cif import CifParser
      struct1 = CifParser("starting/2.cif").get_structures(primitive = False)[0]
      atoms = adaptor.get_atoms(struct1)
      atoms.calc = calc
      print("Pristine Energy:", atoms.get_potential_energy())
      
      for si in range(pop):
        atoms = adaptor.get_atoms(population[si])
        atoms.calc = calc
        
        print(atoms.get_potential_energy())

        print("Model:", calc.model)
        print("Model heads:", calc.model.heads)
        print("Implemented properties:", calc.implemented_properties)
        print("Energy result shape:", calc.results.get("energy", None).shape)
        # https://github.com/neutrons/inspired/blob/6ae3654647769be1f1619adcfc8e42266963d3dd/src/inspired/gui/mlff_worker.py#L124
        if optimize_lattice:
          ecf = ExpCellFilter(atoms)
          dyn = FIRE(ecf, logfile = None, loginterval=-1)
        else:
          dyn = FIRE(atoms, logfile = None, loginterval=-1)
        dyn.run(fmax = fmax, steps = nmax)
        population[si] = adaptor.get_structure(atoms)

      data_pos = [torch.Tensor([site.coords.tolist(
        ) for site in struct.sites]).to(model.device) for struct in population]
      data_cell = [torch.Tensor(struct.lattice.matrix.tolist(
        )).to(model.device) for struct in population]
      
      obj_values_gen = torch.zeros((pop,), device = 'cpu'
        ) if save_obj_values else None

      while not predicted:
        try:
          for k in range(2 ** (orig_split - split)):
            starti = k * (2 ** split)
            stopi = min((k + 1) * (2 ** split) - 1, pop - 1)

            batch_data = model.batch_pos_cell(
              data_pos[starti:(stopi+1)], data_cell[starti:(stopi+1)], 
              population[0])

            #print("batched data")
            
            predictions = model.predict(batch_data, prepared = True, 
              mask = dataset.simfunc.mask)

            #print("predicted")

            objs, obj_total = objective.get(predictions, target, 
              device = device, N = stopi - starti + 1)

            #print("objective obtained")

            lj_repuls = lj_repulsion_mt(batch_data, ljrmins)

            #print("repulsions calculated")

            for j in range(stopi - starti + 1):
              objs[j] += constraint_scale * lj_repuls[j]
              obj_total += constraint_scale * lj_repuls[j]
              objs[j] = objs[j].detach()
              if save_obj_values:
                obj_values[i, starti + j] = objs[j].detach().cpu()
              obj_values_gen[starti + j] = objs[j].detach().cpu()

            #print("objectives added")

            objs_to_compare = torch.nan_to_num(objs, nan = torch.inf)
            for j in range(stopi - starti + 1):
              if data_pos[starti + j].isnan().any() or (
                data_cell[starti + j].isnan().any()) or (
                objs_to_compare[j].isnan().any()):
                objs_to_compare[j] = torch.inf

            min_obj_iter = torch.min(objs_to_compare)
            if (min_obj_iter < best_obj).item():
              best_obj = min_obj_iter.detach()
              obj_arg = torch.argmin(objs_to_compare)
              best_struct = population[starti + obj_arg.item()].copy()

            #print("updated best structure")

            del predictions, objs, obj_total
          predicted = True
        except torch.cuda.OutOfMemoryError:
          split -= 1
          assert split >= 0, "Out of memory with only one structure"

        if i != gens - 1:
          population[0] = best_struct.copy() # elitism
          select_weights = uspex_select_weights(pop, obj_values_gen, survival, 
                                                select_p)
          Vuc = w_adapt * Vuc + (1 - w_adapt) * np.mean(
            [p.volume for p in population[np.argsort(
            obj_values_gen.detach().numpy())[:N_adapt]]])
          for j in range(pop - 1):
            rejected = True
            while rejected:
              choose_op = np.random.rand()
              if choose_op < p_her:
                parents_i = np.random.choice(pop, size = 2, p = select_weights, 
                                            replace = False)
                new_struct = uspex_heredity(population[parents_i[0]], 
                  population[parents_i[1]], shift1prob = shift1prob, 
                  shift2prob = shift2prob, add_by_density = False)
              elif choose_op < p_her + p_mut:
                parent_i = np.random.choice(pop, p = select_weights)
                new_struct = uspex_mutation(population[parent_i], σl = σl)
              else:
                parent_i = np.random.choice(pop, p = select_weights)
                new_struct = uspex_permutation(population[parent_i], 
                                              nperms = nperms)
                
              new_struct.scale_lattice(Vuc)
              rejected = lj_reject(new_struct, buffer = constraint_buffer)
            pop[j + 1] = new_struct

    return best_struct, obj_values

