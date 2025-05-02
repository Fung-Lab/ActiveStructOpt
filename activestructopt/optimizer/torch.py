from activestructopt.common.dataloader import prepare_data_pmg, reprocess_data
from activestructopt.common.constraints import lj_rmins, lj_repulsion
from activestructopt.model.base import BaseModel
from activestructopt.dataset.base import BaseDataset
from activestructopt.objective.base import BaseObjective
from activestructopt.optimizer.base import BaseOptimizer
from activestructopt.sampler.base import BaseSampler
from activestructopt.common.registry import registry
from pymatgen.core.structure import IStructure
from pymatgen.core import Lattice
import torch
import numpy as np

@registry.register_optimizer("Torch")
class Torch(BaseOptimizer):
  def __init__(self) -> None:
    pass

  def run(self, model: BaseModel, dataset: BaseDataset, 
    objective: BaseObjective, sampler: BaseSampler, 
    starts = 128, iters_per_start = 100, optimizer = "Adam",
    optimizer_args = {}, optimize_atoms = True, 
    optimize_lattice = False, save_obj_values = False, 
    constraint_scale = 1.0, pos_lr = 0.001, cell_lr = 0.001,
    constraint_buffer = 0.85, random_starts = False, 
    save_only_constrained_structures = False,
    **kwargs) -> IStructure:
    
    starting_structures = [sampler.sample(
      ) for j in range(starts)] if random_starts else [dataset.structures[j].copy(
      ) if j < dataset.N else sampler.sample(
      ) for j in range(starts)]

    obj_values = torch.zeros((iters_per_start, starts), device = 'cpu'
      ) if save_obj_values else None
    
    device = model.device
    nstarts = len(starting_structures)
    natoms = len(starting_structures[0])
    ljrmins = torch.tensor(lj_rmins, device = device) * constraint_buffer
    best_obj = torch.tensor([float('inf')], device = device)
    if optimize_atoms:
      best_x = torch.zeros(3 * natoms, device = device)
    if optimize_lattice:
      best_cell = torch.zeros((3, 3), device = device)
    target = torch.tensor(dataset.target, device = device)
    
    data = [prepare_data_pmg(s, dataset.config, pos_grad = True, device = device, 
      preprocess = False) for s in starting_structures]
    for i in range(nstarts): # process node features
      reprocess_data(data[i], dataset.config, device, edges = False)
    
    to_optimize = []
    if optimize_atoms:
      to_optimize += [{'params': d.pos, 'lr': pos_lr} for d in data]
    if optimize_lattice:
      to_optimize += [{'params': d.cell, 'lr': cell_lr} for d in data]
    optimizer = getattr(torch.optim, optimizer)(to_optimize, 
      **(optimizer_args))
    
    split = int(np.ceil(np.log2(nstarts)))
    orig_split = split

    for i in range(iters_per_start):
      predicted = False
      while not predicted:
        try:
          for k in range(2 ** (orig_split - split)):
            starti = k * (2 ** split)
            stopi = min((k + 1) * (2 ** split) - 1, nstarts - 1)

            optimizer.zero_grad()
            for j in range(nstarts):
              data[j].cell.requires_grad_(False)
              data[j].pos.requires_grad_(False)
            for j in range(stopi - starti + 1):
              if optimize_atoms:
                data[starti + j].pos.requires_grad_()
              if optimize_lattice:
                data[starti + j].cell.requires_grad_()
              reprocess_data(data[starti + j], dataset.config, device, 
                nodes = False)

            predictions = model.predict(data[starti:(stopi+1)], 
              prepared = True, mask = dataset.simfunc.mask)

            objs, obj_total = objective.get(predictions, target, 
              device = device, N = stopi - starti + 1)

            lj_repulsions = torch.zeros(stopi - starti + 1, device = device)
            for j in range(stopi - starti + 1):
              lj_repuls = lj_repulsion(data[starti + j], ljrmins)
              objs[j] += constraint_scale * lj_repuls
              obj_total += constraint_scale * lj_repuls
              objs[j] = objs[j].detach()
              lj_repulsions[j] = lj_repuls
              if save_obj_values:
                obj_values[i, starti + j] = objs[j].detach().cpu()

            objs_to_compare = torch.nan_to_num(objs, nan = torch.inf)
            for j in range(stopi - starti + 1):
              if data[starti + j].pos.isnan().any() or (
                data[starti + j].cell[0].isnan().any()) or (
                objs_to_compare[j].isnan().any()):
                objs_to_compare[j] = torch.inf

            min_obj_iter = torch.min(objs_to_compare)
            if (min_obj_iter < best_obj).item():
              best_obj = min_obj_iter.detach()
              obj_arg = torch.argmin(objs_to_compare)
              if (not save_only_constrained_structures) or (
                lj_repulsions[obj_arg.item()] <= torch.tensor(
                [0.0], device = device)).item():
                if optimize_atoms:
                  best_x = data[starti + obj_arg.item()].pos.clone().detach().flatten()
                if optimize_lattice:
                  best_cell = data[starti + obj_arg.item()].cell[0].clone().detach()

            if i != iters_per_start - 1:
              obj_total.backward()
              optimizer.step()
            del predictions, objs, obj_total
          predicted = True
        except torch.cuda.OutOfMemoryError:
          split -= 1
          assert split >= 0, "Out of memory with only one structure"

    if optimize_atoms:
      new_x = best_x.detach().cpu().numpy()
      del best_x
    if optimize_lattice:
      new_cell = best_cell.detach().cpu().numpy()
      del best_cell

    del target, data
    new_structure = starting_structures[0].copy()

    if optimize_lattice:
      new_structure.lattice = Lattice(new_cell)
    if optimize_atoms:
      for i in range(len(new_structure)):
        try:
          new_structure[i].coords = new_x[(3 * i):(3 * (i + 1))]
        except np.linalg.LinAlgError as e:
          print(best_obj)
          print(new_cell)
          print(new_structure.lattice)
          print(new_x)
          raise e

    
    return new_structure, obj_values
