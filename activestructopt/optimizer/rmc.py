from activestructopt.model.base import BaseModel
from activestructopt.dataset.base import BaseDataset
from activestructopt.objective.base import BaseObjective
from activestructopt.optimizer.base import BaseOptimizer
from activestructopt.sampler.base import BaseSampler
from activestructopt.common.registry import registry
from pymatgen.core.structure import IStructure
import numpy as np
from activestructopt.common.dataloader import prepare_data_pmg
from activestructopt.common.constraints import lj_rmins, lj_repulsion
import torch

def step(structure, latticeprob, σr, σl, σθ, step_type = 'one'):
  new_struct = structure.copy()
  if np.random.rand() < latticeprob:
    lattice_step(new_struct, σl, σθ)
  else:
    positions_step(new_struct, σr, step_type = step_type)
  return new_struct

def lattice_step(structure, σl, σθ):
  structure.lattice = structure.lattice.from_parameters(
    np.maximum(0.0, structure.lattice.a + σl * np.random.randn()),
    np.maximum(0.0, structure.lattice.b + σl * np.random.randn()), 
    np.maximum(0.0, structure.lattice.c + σl * np.random.randn()), 
    structure.lattice.alpha + σθ * np.random.randn(), 
    structure.lattice.beta + σθ * np.random.randn(), 
    structure.lattice.gamma + σθ * np.random.randn()
  )

def positions_step(structure, σr, step_type = 'one'):
  if step_type == 'one':
    atom_i = np.random.choice(range(len(structure)))
    structure.sites[atom_i].a = (structure.sites[atom_i].a + 
      σr * np.random.rand() / structure.lattice.a) % 1
    structure.sites[atom_i].b = (structure.sites[atom_i].b + 
      σr * np.random.rand() / structure.lattice.b) % 1
    structure.sites[atom_i].c = (structure.sites[atom_i].c + 
      σr * np.random.rand() / structure.lattice.c) % 1
  else:
    structure.perturb(σr)

@registry.register_optimizer("RMC")
class RMC(BaseOptimizer):
  def __init__(self) -> None:
    pass

  def run(self, model: BaseModel, dataset: BaseDataset, 
    objective: BaseObjective, sampler: BaseSampler, 
    starts = 10, iters_per_start = 1000, σ = 0.0025, latticeprob = 0.5, 
    σr = 0.01, σl = 0.01, σθ = 0.1, 
    save_obj_values = False, **kwargs) -> IStructure:

    device = model.device
    ljrmins = torch.tensor(lj_rmins, device = device)
    
    structures = [dataset.structures[np.random.randint(len(
      dataset.structures))].copy() for _ in range(starts)]
    prev_structures = [s.copy for s in structures]
    target = torch.tensor(dataset.target, device = device)

    best_obj = torch.tensor([float('inf')], device = device)
    best_structure = None

    obj_vals = None
    if save_obj_values:
      obj_vals = torch.zeros((iters_per_start, starts), device = 'cpu')

    prev_objs = torch.inf * torch.ones(starts, device = device)

    split = int(np.ceil(np.log2(starts)))
    orig_split = split

    for i in range(iters_per_start):
      predicted = False
      while not predicted:
        try:
          for k in range(2 ** (orig_split - split)):
            starti = k * (2 ** split)
            stopi = min((k + 1) * (2 ** split) - 1, starts - 1)

            data = [prepare_data_pmg(s, dataset.config, pos_grad = True, 
              device = device, preprocess = True
              ) for s in structures[starti:(stopi + 1)]]
              
            predictions = model.predict(data, prepared = True, 
              mask = dataset.simfunc.mask)

            objs, _ = objective.get(predictions, target, 
              device = device, N = stopi - starti + 1)

            for j in range(stopi - starti + 1):
              objs[j] += lj_repulsion(data[j], ljrmins)
            if save_obj_values:
              obj_vals[i, starti:(stopi + 1)] = objs.cpu()

            Δobjs = objs - prev_objs[starti:(stopi + 1)]
            better = Δobjs <= 0
            hastings = torch.log(torch.rand(stopi - starti + 1, 
              device = device)) < Δobjs / (-2 * σ ** 2)
            accept = torch.logical_or(better, hastings)
            for j in range(stopi - starti + 1):
              if (objs[j] < best_obj).item():
                best_obj = objs[j]
                best_structure = structures[starti + j]
              if (accept[j]).item():
                prev_structures[starti + j] = structures[starti + j].copy()
                prev_objs[starti + j] = objs[j]
              structures[starti + j] = prev_structures[starti + j].copy()
              structures[starti + j] = step(structures[starti + j], latticeprob, 
                σr, σl, σθ, step_type = 'all')
          predicted = True
          del data, predictions, objs, hastings, accept, Δobjs, better
        except torch.cuda.OutOfMemoryError:
          split -= 1
          assert split >= 0, "Out of memory with only one structure"

    return best_structure, obj_vals
