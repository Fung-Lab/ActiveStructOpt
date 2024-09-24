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
    nstarts = 128, iters_per_start = 100, lr = 0.01, optimizer = "Adam",
    optimizer_args = {}, optimize_atoms = True, 
    optimize_lattice = False, save_obj_values = False, opt_start = None, **kwargs) -> IStructure:
    
    if opt_start is None:
      starting_structures = [dataset.structures[j].copy(
        ) if j < dataset.N else sampler.sample(
        ) for j in range(nstarts)]
    else: 
      starting_structures = [opt_start for _ in range(nstarts)]

    obj_values = torch.zeros((iters_per_start, nstarts), device = 'cpu'
      ) if save_obj_values else None
    
    device = model.device
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
    
    to_optimize = []
    if optimize_atoms:
      to_optimize += [d.pos for d in data]
    if optimize_lattice:
      to_optimize += [d.cell for d in data]
    optimizer = getattr(torch.optim, optimizer)(to_optimize, lr = lr, 
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
              if hasattr(data[j], 'displacement'):
                data[j].displacement = data[j].displacement.detach()
              data[j].cell.detach()
              data[j].pos.detach()
              
            for j in range(stopi - starti + 1):
              if optimize_atoms:
                data[starti + j].pos = data[starti + j].pos.detach()
                data[starti + j].pos.requires_grad_()
              if optimize_lattice:
                data[starti + j].cell = data[starti + j].cell.detach()
                data[starti + j].cell.requires_grad_()
              if optimize_lattice:
                #https://github.com/Fung-Lab/MatDeepLearn_dev/blob/main/matdeeplearn/models/base_model.py#L110
                #https://github.com/mir-group/nequip/blob/main/nequip/nn/_grad_output.py
                #https://github.com/atomistic-machine-learning/schnetpack/issues/165
                data[starti + j].displacement = torch.zeros((1, 
                  3, 3), dtype = data[starti + j].pos.dtype, 
                  device=data[starti + j].pos.device)
                data[starti + j].displacement.requires_grad_(True)
                symmetric_displacement = 0.5 * (data[starti + j].displacement + 
                  data[starti + j].displacement.transpose(-1, -2))
                data[starti + j].pos = (data[starti + j].pos + torch.bmm(
                  data[starti + j].pos.unsqueeze(0),
                  symmetric_displacement).squeeze(-2)).squeeze(0)            
                data[starti + j].cell = data[starti + j].cell + torch.bmm(
                  data[starti + j].cell, symmetric_displacement) 
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
                obj_values[i, starti + j] = objs[j].detach().cpu()

            min_obj_iter = torch.min(torch.nan_to_num(objs, nan = torch.inf))
            if (min_obj_iter < best_obj).item():
              best_obj = min_obj_iter.detach()
              obj_arg = torch.argmin(torch.nan_to_num(objs, nan = torch.inf))
              if optimize_atoms:
                best_x = data[starti + obj_arg.item()].pos.detach().flatten()
              if optimize_lattice:
                best_cell = data[starti + obj_arg.item()].cell[0].detach()

            if i != iters_per_start - 1:
              obj_total.backward()
              if optimize_lattice:
                # https://github.com/Fung-Lab/MatDeepLearn_dev/blob/main/matdeeplearn/models/torchmd_etEarly.py#L229
                for j in range(stopi - starti + 1):
                  volume = torch.einsum("zi,zi->z", 
                    data[starti + j].cell[:, 0, :], torch.cross(
                    data[starti + j].cell[:, 1, :], 
                    data[starti + j].cell[:, 2, :], dim = 1)).unsqueeze(-1)
                  data[starti + j].cell.grad = -data[
                    starti + j].displacement.grad / volume.view(-1, 1, 1)
                  
              optimizer.step()
            del predictions, objs, obj_total
          predicted = True
        except torch.cuda.OutOfMemoryError:
          split -= 1
          assert split >= 0, "Out of memory with only one structure"

    if optimize_atoms:
      new_x = best_x.detach().cpu().numpy()
    if optimize_lattice:
      new_cell = best_cell.detach().cpu().numpy()
    
    del best_x, target, data
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