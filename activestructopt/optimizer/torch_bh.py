from activestructopt.common.dataloader import prepare_data_pmg, reprocess_data
from activestructopt.common.constraints import lj_rmins, lj_repulsion, lj_reject
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

@registry.register_optimizer("TorchBH")
class TorchBH(BaseOptimizer):
  def __init__(self) -> None:
    pass

  def run(self, model: BaseModel, dataset: BaseDataset, 
    objective: BaseObjective, sampler: BaseSampler, 
    hops = 128, iters_per_hops = 100, optimizer = "Adam",
    optimizer_args = {}, optimize_atoms = True, 
    optimize_lattice = False, save_obj_values = False, 
    constraint_scale = 1.0, pos_lr = 0.001, cell_lr = 0.001,
    perturbrmin = 0.0, perturbrmax = 0.2, perturblσ = 0.1,
    **kwargs) -> IStructure:

    device = model.device
    
    structure = dataset.structures[0].copy()
    prev_structure = structure.copy()
    target = torch.tensor(dataset.target, device = device)

    best_obj = torch.tensor([float('inf')], device = device)
    if optimize_atoms:
      best_x = torch.zeros(3 * natoms, device = device)
    if optimize_lattice:
      best_cell = torch.zeros((3, 3), device = device)

    rmc_sampler = SingleAtomPerturbation(structure, perturbrmin = σr, 
      perturbrmax = σr, perturblmax = 0, perturbθmax = 0, 
      lattice_prob = 0)

    #obj_vals = None
    #if save_obj_values:
    #  obj_vals = torch.zeros((iters_per_start, starts), device = 'cpu')

    prev_obj = torch.tensor([float('inf')], device = device)

    for _ in range(hops):
      data = [prepare_data_pmg(structure, dataset.config, pos_grad = False, 
        device = device, preprocess = True, cell_grad = False
        )]

      to_optimize = []
      if optimize_atoms:
        to_optimize += [{'params': data.pos, 'lr': pos_lr}]
      if optimize_lattice:
        to_optimize += [{'params': data.cell, 'lr': cell_lr}]
      optimizer = getattr(torch.optim, optimizer)(to_optimize, 
        **(optimizer_args))

      best_local_obj = torch.tensor([float('inf')], device = device)
      best_local_x = None
      best_local_cell = None
      
      for j in range(iters_per_hops):
        optimizer.zero_grad()
        if optimize_atoms:
          data.pos.requires_grad_()
        if optimize_lattice:
          data.cell.requires_grad_()
        reprocess_data(data, dataset.config, device, nodes = False)
        predictions = model.predict([data], prepared = True, 
          mask = dataset.simfunc.mask)
        _, obj_total = objective.get(predictions, target, device = device, 
          N = 1)
        obj_total += constraint_scale * lj_repulsion(data, ljrmins)
        obj_total = torch.nan_to_num(obj_total, nan = torch.inf)
        if (obj_total < best_local_obj).item():
          best_local_obj = obj_total.detach()
          if optimize_atoms:
            best_local_x = data.pos.detach().flatten()
          if optimize_lattice:
            best_local_cell = data.cell[0].detach()
          if (obj_total < best_obj).item():
            best_obj = best_local_obj
            if optimize_atoms:
              best_x = best_local_x
            if optimize_lattice:
              best_cell = best_local_cell
        if j != iters_per_hops - 1:
          obj_total.backward()
          optimizer.step()

      new_obj = best_local_obj
      
      Δobjs = new_obj - prev_obj
      better = Δobjs <= 0
      hastings = torch.log(torch.rand(1, device = device)) < Δobjs / (
        -2 * σ ** 2)
      accept = torch.logical_or(better, hastings)
      if (accept).item():
        if optimize_atoms:
          new_x = best_local_x.cpu().numpy()
        if optimize_lattice:
          new_cell = best_local_cell.cpu().numpy()
        
        prev_structure = starting_structure.copy()
        
        if optimize_lattice:
          prev_structure.lattice = Lattice(new_cell)
        if optimize_atoms:
          for i in range(len(prev_structure)):
            try:
              prev_structure[i].coords = new_x[(3 * i):(3 * (i + 1))]
            except np.linalg.LinAlgError as e:
              print(best_obj)
              print(new_cell)
              print(prev_structure.lattice)
              print(new_x)
              raise e
        prev_structure = structure.copy()
        prev_obj = obj_total
    
      rmc_sampler.initial_structure = prev_structure.copy()
      structure = rmc_sampler.sample()

    if optimize_atoms:
      new_x = best_x.detach().cpu().numpy()
    if optimize_lattice:
      new_cell = best_cell.detach().cpu().numpy()

    best_structure = starting_structure.copy()

    if optimize_lattice:
      best_structure.lattice = Lattice(new_cell)
    if optimize_atoms:
      for i in range(len(best_structure)):
        try:
          best_structure[i].coords = new_x[(3 * i):(3 * (i + 1))]
        except np.linalg.LinAlgError as e:
          print(best_obj)
          print(new_cell)
          print(best_structure.lattice)
          print(new_x)
          raise e
      
    return best_structure, None
