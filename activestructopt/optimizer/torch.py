from activestructopt.common.dataloader import prepare_data, reprocess_data
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
    starts = 128, iters_per_start = 100, lr = 0.01, optimizer = "Adam",
    optimizer_args = {}, optimize_atoms = True, 
    optimize_lattice = False, save_obj_values = False, **kwargs) -> IStructure:
    
    starting_structures = [dataset.structures[j].copy(
      ) if j < dataset.N else sampler.sample(
      ) for j in range(starts)]

    obj_values = torch.zeros((2 * iters_per_start, starts), device = 'cpu'
      ) if save_obj_values else None
    
    device = model.device
    nstarts = len(starting_structures)
    natoms = len(starting_structures[0])
    ljrmins = torch.tensor(lj_rmins, device = device)
    best_obj = torch.tensor([float('inf')], device = device)
    if optimize_atoms:
      best_x = torch.zeros(3 * natoms, device = device)
    if optimize_lattice:
      best_cell = torch.zeros((3, 3), device = device)
    target = torch.tensor(dataset.target, device = device)
    
    data = [prepare_data(s, dataset.config, pos_grad = True, device = device, 
      preprocess = False) for s in starting_structures]
    for i in range(nstarts): # process node features
      reprocess_data(data[i], dataset.config, device, edges = False)
    
    if optimize_atoms:
      pos_optimizer = getattr(torch.optim, optimizer)([d.pos for d in data], 
        lr = lr, **(optimizer_args))
    if optimize_lattice:
      cell_optimizer = getattr(torch.optim, optimizer)([d.cell for d in data], 
        lr = lr, **(optimizer_args))
    
    split = int(np.ceil(np.log2(nstarts)))
    orig_split = split

    for i in range(iters_per_start):
      predicted = False
      while not predicted:
        try:
          if optimize_lattice:
            for k in range(2 ** (orig_split - split)):
              starti = k * (2 ** split)
              stopi = min((k + 1) * (2 ** split) - 1, nstarts - 1)

              cell_optimizer.zero_grad()
              for j in range(nstarts):
                data[j].cell.requires_grad_(False)
                data[j].pos.requires_grad_(False)
              for j in range(stopi - starti + 1):
                data[starti + j].cell.requires_grad_()
                reprocess_data(data[starti + j], dataset.config, device, 
                  nodes = False)

              predictions = model.predict(data[starti:(stopi+1)], 
                prepared = True, mask = dataset.simfunc.mask)

              objs, obj_total = objective.get(predictions, target, 
                device = device, N = stopi - starti + 1)
              for j in range(stopi - starti + 1):
                objs[j] += lj_repulsion(data[starti + j], ljrmins)
                obj_total += lj_repulsion(data[starti + j], ljrmins)
                objs[j] = objs[j].detach()
                if save_obj_values:
                  obj_values[2 * i, starti + j] = objs[j].detach().cpu()

              if (torch.min(objs) < best_obj).item():
                best_obj = torch.min(objs).detach()
                if optimize_atoms:
                  best_x = data[starti + torch.argmin(objs).item()].pos.detach(
                    ).flatten()
                best_cell = data[starti + torch.argmin(objs).item(
                  )].cell[0].detach()

              if i != iters_per_start - 1 and not optimize_atoms:
                obj_total.backward()
                cell_optimizer.step()
              del predictions, objs, obj_total              

          if optimize_atoms:
            for k in range(2 ** (orig_split - split)):
              starti = k * (2 ** split)
              stopi = min((k + 1) * (2 ** split) - 1, nstarts - 1)

              pos_optimizer.zero_grad()
              for j in range(nstarts):
                data[j].cell.requires_grad_(False)
                data[j].pos.requires_grad_(False)
              for j in range(stopi - starti + 1):
                data[starti + j].pos.requires_grad_()
                reprocess_data(data[starti + j], dataset.config, device, 
                  nodes = False)

              predictions = model.predict(data[starti:(stopi+1)], 
                prepared = True, mask = dataset.simfunc.mask)

              objs, obj_total = objective.get(predictions, target, 
                device = device, N = stopi - starti + 1)
              for j in range(stopi - starti + 1):
                objs[j] += lj_repulsion(data[starti + j], ljrmins)
                obj_total += lj_repulsion(data[starti + j], ljrmins)
                objs[j] = objs[j].detach()
                if save_obj_values:
                  obj_values[2 * i + 1, starti + j] = objs[j].detach().cpu()

              if (torch.min(objs) < best_obj).item():
                best_obj = torch.min(objs).detach()
                best_x = data[starti + torch.argmin(objs).item()].pos.detach(
                  ).flatten()
                if optimize_lattice:
                  best_cell = data[starti + torch.argmin(objs).item(
                    )].cell[0].detach()
              
              if i != iters_per_start - 1:
                obj_total.backward()
                pos_optimizer.step()
              del predictions, objs, obj_total
          predicted = True
        except torch.cuda.OutOfMemoryError:
          split -= 1
          assert split >= 0, "Out of memory with only one structure"

    if optimize_atoms:
      new_x = best_x.detach().cpu().numpy()
    if optimize_lattice:
      new_cell = best_cell.detach().cpu().numpy()
    
    del best_obj, best_x, target, data
    new_structure = starting_structures[0].copy()

    if optimize_lattice:
      new_structure.lattice = Lattice(new_cell)
    if optimize_atoms:
      for i in range(len(new_structure)):
        new_structure[i].coords = new_x[(3 * i):(3 * (i + 1))]
    
    return new_structure, obj_values
