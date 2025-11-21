from activestructopt.common.constraints import lj_rmins, lj_repulsion_mt, lj_reject
from activestructopt.objective.base import BaseObjective
from activestructopt.optimizer.base import BaseOptimizer
from activestructopt.dataset.base import BaseDataset
from activestructopt.sampler.base import BaseSampler
from activestructopt.common.registry import registry
from activestructopt.model.base import BaseModel
from activestructopt.model.groundtruth import GroundTruth
from pymatgen.transformations.standard_transformations import RotationTransformation
from pymatgen.core.structure import IStructure
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from ase.calculators.calculator import Calculator
from ase import filters as asefilters
from ase import Atoms
from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs
from orb_models.forcefield.calculator import ORBCalculator
from orb_models.forcefield.base import batch_graphs
from orb_models.forcefield import pretrained
import numpy as np
import torch

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
  weights = np.clip(Nsurv - ranks, a_min = 0, a_max = None) ** p
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
    if comp_to_match[k] < new_struct.composition.as_dict().get(k, 0):
      n_to_remove = new_struct.composition.as_dict().get(k, 0) - comp_to_match[k]
      is_to_remove = np.random.choice(np.where(
        [s.species_string == k for s in new_struct.sites])[0], 
        int(n_to_remove), replace = False)
      new_struct = new_struct.remove_sites(is_to_remove)

    if comp_to_match[k] > new_struct.composition.as_dict().get(k, 0):
      n_to_add = comp_to_match[k] - new_struct.composition.as_dict().get(k, 0)

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

        found_new_site = np.min([np.linalg.norm(
          s.frac_coords - new_site_coords) for s in new_struct.sites]
          ) > 1e-6 if len(new_struct.sites) > 0 else True
        if found_new_site:
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

class PassThroughCalc(Calculator):
  def __init__(self, directory = "."):
    Calculator.__init__(self, directory = directory)
    self.results = {}
    self.implemented_properties = ["free_energy", "stress", "forces"]
    pass

  def get_property(self, name, atoms = None, allow_calculation = True):
    result = self.results[name]
    if isinstance(result, np.ndarray):
      result = result.copy()
    return result

# A modified version of the ASE FIRE Optimizer to make batched predictions
# https://gitlab.com/ase/ase/-/blob/9fe18854390079711a314b0ce4bf34109b0eb19a/ase/optimize/fire.py#L6
# https://gitlab.com/ase/ase/-/blob/master/ase/optimize/optimize.py
# https://gitlab.com/ase/ase/-/blob/master/ase/filters.py
class BatchFIRE():
  def __init__(self, atoms: list[Atoms], calc, dt: float = 0.1, maxstep: float = 0.2,
    dtmax: float = 1.0, Nmin: int = 5, finc: float = 1.1, fdec: float = 0.5, 
    astart: float = 0.1, fa: float = 0.99, a: float = 0.1, 
    downhill_check: bool = False, opt_lat = False, device = 'cuda', filtername = 'UnitCellFilter'):
    
    self.opt_lat = opt_lat
    self.nstructs = len(atoms)
    self.atoms = atoms
    self.natoms = len(self.atoms[0])
    self.calcs = [PassThroughCalc() for _ in self.atoms]
    for i in range(self.nstructs):
        self.atoms[i].calc = self.calcs[i]
    self.atoms = [getattr(asefilters, filtername)(a) for a in self.atoms] if opt_lat else self.atoms
    
    self.calc = calc
    self.ndofs = 3 * len(self.atoms[0])
    self.fmax = None
    self.downhill_check = downhill_check
    self.maxstep = maxstep
    self.dtmax = dtmax
    self.Nmin = Nmin
    self.finc = finc
    self.fdec = fdec
    self.astart = astart
    self.nsteps = 0
    self.dts = [dt for _ in atoms]
    self.Nsteps = [0 for _ in atoms]
    self.fas = [fa for _ in atoms]
    self.a_s = [a for _ in atoms]
    self.optimizables = [a.__ase_optimizable__() for a in atoms]
    self.initialize()

  def initialize(self):
    self.vels = [None for _ in self.atoms]
    self.e_lasts = [None for _ in self.atoms]
    self.r_lasts = [None for _ in self.atoms]
    self.vel_lasts = [None for _ in self.atoms]

  def run(self, fmax = 0.05, steps = 100_000_000):
    self.fmax = fmax
    while self.nsteps < steps:
      # compute the next step
      self.step()
      self.nsteps += 1

  def step(self):
    atom_graphs = [ase_atoms_to_atom_graphs(
      a.atoms if self.opt_lat else a,
      system_config = self.calc.system_config,
      max_num_neighbors = self.calc.max_num_neighbors,
      edge_method = self.calc.edge_method,
      half_supercell = self.calc.half_supercell,
      device = self.calc.device,
    ) for a in self.atoms]
    batch = batch_graphs(atom_graphs)
    out = self.calc.model.predict(batch)
    for i in range(self.nstructs):
      self.calcs[i].results = {
        'free_energy':  out['energy'][i].detach().cpu().item(),
        'forces': out['grad_forces'][(i * self.natoms):((i + 1) * 
                                      self.natoms)].detach().cpu().numpy(),
        'stress': out['grad_stress'][i].detach().cpu().numpy(),
      }
      self.atoms[i].results = self.calcs[i].results

    xs = [a.get_positions().ravel()  for a in self.atoms]
    values = [a.get_potential_energy(force_consistent=True) for a in self.atoms]
    gradients = [a.get_forces().ravel() for a in self.atoms]
    
    for i in range(self.nstructs):
      if self.vels[i] is None:
        self.vels[i] = np.zeros(self.ndofs)
        if self.downhill_check:
          self.e_lasts[i] = values[i]
          self.r_lasts[i] = xs[i]
          self.vel_lasts[i] = self.vels[i].copy()
      else:
        is_uphill = False
        if self.downhill_check:
          e = values[i]
          # Check if the energy actually decreased
          if e > self.e_lasts[i]:
            self.atoms[i].set_positions(self.r_lasts[i].reshape(-1, 3))
            is_uphill = True
          self.e_lasts[i] = values[i]
          self.r_lasts[i] = xs[i]
          self.vel_last = self.vels[i].copy()

        vf = np.vdot(gradients[i], self.vels[i])
        grad2 = np.vdot(gradients[i], gradients[i])
        if vf > 0.0 and not is_uphill:
          self.vels[i] = ((1.0 - self.a_s[i]) * self.vels[i] + self.a_s[i] * 
            gradients[i] / np.sqrt(grad2) * np.sqrt(np.vdot(self.vels[i], 
                                                            self.vels[i])))
          if self.Nsteps[i] > self.Nmin:
            self.dts[i] = min(self.dts[i] * self.finc, self.dtmax)
            self.a_s[i] *= self.fas[i]
          self.Nsteps[i] += 1
        else:
          self.vels[i][:] *= 0.0
          self.a_s[i] = self.astart
          self.dts[i] *= self.fdec
          self.Nsteps[i] = 0

        self.vels[i] += self.dts[i] * gradients[i]
        dr = self.dts[i] * self.vels[i]
        normdr = np.sqrt(np.vdot(dr, dr))
        if normdr > self.maxstep:
          dr = self.maxstep * dr / normdr
        r = self.atoms[i].get_positions().ravel()
        self.atoms[i].set_positions((r + dr).reshape(-1, 3))

@registry.register_optimizer("USPEX")
class USPEX(BaseOptimizer):
  def __init__(self) -> None:
    orbff = pretrained.orb_v3_conservative_inf_omat(
        device='cuda',
        precision="float32-high",   # or "float32-highest" / "float64
    )
    self.calc = ORBCalculator(orbff, device='cuda')
  
  def run(self, model: BaseModel, dataset: BaseDataset, 
    objective: BaseObjective, sampler: BaseSampler, 
    pop = 64, gens = 100, optimize_atoms = True, 
    optimize_lattice = True, save_obj_values = False, 
    constraint_scale = 1.0, constraint_buffer = 0.85, 
    random_starts = False, fmax = 0.01, nmax = 100, 
    survival = 0.6, select_p = 1, p_her = 0.85, p_mut = 0.1,
    shift1prob = 1.0, shift2prob = 0.05, σl = 0.7, nperms = 3,
    w_adapt = 0.5, N_adapt = 4, filtername = 'UnitCellFilter',
    **kwargs) -> IStructure:

    if len(dataset.structures[0].composition.as_dict().keys()) < 2:
      # If monatomic, ignore permutations
      p_her = p_her / (p_her + p_mut)
      p_mut = 1.0 - p_her
    
    adaptor = AseAtomsAdaptor()

    population = [sampler.sample(
      ) for _ in range(pop)] if random_starts else [dataset.structures[
      j].copy() if j < dataset.N else sampler.sample(
      ) for j in range(pop)]

    obj_values = torch.zeros((gens, pop), device = 'cpu'
      ) if save_obj_values else None
    
    device = model.device

    ljrmins = torch.tensor(lj_rmins, device = device) * constraint_buffer
    best_obj = torch.tensor([float('inf')], device = device)
    best_struct = population[0].copy()
    target = torch.tensor(dataset.target, device = device)
    
    split = int(np.ceil(np.log2(pop)))
    orig_split = split

    #print("starting loop")

    Vuc = dataset.structures[0].volume

    for i in range(gens):
      # Local Energy Optimization (TODO: Make this parallel)
      dyn = BatchFIRE([adaptor.get_atoms(population[si]) for si in range(pop)], self.calc,
                      opt_lat = optimize_lattice, device = device, filtername = filtername)
      dyn.run(fmax = fmax, steps = nmax)
      for si in range(pop):
        population[si] = adaptor.get_structure(dyn.atoms[si].atoms)

      data_pos = [torch.Tensor([site.coords.tolist(
        ) for site in struct.sites]).to(model.device) for struct in population]
      data_cell = [torch.Tensor(struct.lattice.matrix.tolist(
        )).to(model.device) for struct in population]
      
      obj_values_gen = torch.zeros((pop,), device = 'cpu')
      predicted = False
      while not predicted:
        try:
          for k in range(2 ** (orig_split - split)):
            starti = k * (2 ** split)
            stopi = min((k + 1) * (2 ** split) - 1, pop - 1)

            if type(model) is GroundTruth:
              predictions = model.predict(population[starti:(stopi+1)], 
                prepared = False, mask = mask = dataset.simfunc.mask)
            else:
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
            [population[pi].volume for pi in np.argsort(
            obj_values_gen.detach().numpy())[:N_adapt]])
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
            population[j + 1] = new_struct

    return best_struct, obj_values

