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
    
    starting_structure = dataset.structures[0]
    device = model.device
    natoms = len(starting_structure)
    ljrmins = torch.tensor(lj_rmins, device = device)
    best_obj = torch.tensor([float('inf')], device = device)
    if optimize_atoms:
      best_x = torch.zeros(3 * natoms, device = device)
    if optimize_lattice:
      best_cell = torch.zeros((3, 3), device = device)
    target = torch.tensor(dataset.target, device = device)
    
    data = prepare_data_pmg(starting_structure, dataset.config, 
      pos_grad = optimize_atoms, cell_grad = optimize_lattice,
      device = device, preprocess = False)
    reprocess_data(data, dataset.config, device, edges = False)
    
    to_optimize = []
    if optimize_atoms:
      to_optimize += [{'params': data.pos, 'lr': pos_lr}]
    if optimize_lattice:
      to_optimize += [{'params': data.cell, 'lr': cell_lr}]
    optimizer = getattr(torch.optim, optimizer)(to_optimize, 
      **(optimizer_args))

    for i in range(hops):
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
        if (obj_total < best_obj).item():
          best_obj = obj_total.detach()
          if optimize_atoms:
            best_x = data.pos.detach().flatten()
          if optimize_lattice:
            best_cell = data.cell[0].detach()
        if j != iters_per_hops - 1:
          obj_total.backward()
          optimizer.step()
      
      rejected = True
      new_structure = starting_structure.copy()
      if optimize_atoms:
        new_x = data.pos.detach().flatten().cpu().numpy()
      if optimize_lattice:
        new_cell = data.cell[0].detach().cpu().numpy()

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

      while rejected:
        try:
          hop_structure = new_structure.copy()
          hop_structure.perturb(np.random.uniform(perturbrmin, perturbrmax))
          hop_structure.lattice = Lattice(hop_structure.lattice.matrix + 
            perturblσ * np.random.normal(0, 1, (3, 3)))
          rejected = lj_reject(hop_structure)
        except:
          rejected = True

      data = prepare_data_pmg(hop_structure, dataset.config, 
        pos_grad = optimize_atoms, cell_grad = optimize_lattice,
        device = device, preprocess = False)
      reprocess_data(data, dataset.config, device, edges = False)

    if optimize_atoms:
      new_x = best_x.detach().cpu().numpy()
    if optimize_lattice:
      new_cell = best_cell.detach().cpu().numpy()

    new_structure = starting_structure.copy()

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

    return new_structure, None
