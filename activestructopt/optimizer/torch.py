from activestructopt.common.dataloader import prepare_data_pmg, reprocess_data
from activestructopt.model.base import BaseModel
from activestructopt.dataset.base import BaseDataset
from activestructopt.objective.base import BaseObjective
from activestructopt.optimizer.base import BaseOptimizer, select_distinct
from activestructopt.optimizer.torch_mt import to_structure
from activestructopt.sampler.base import BaseSampler
from activestructopt.constraint.base import BaseConstraint
from activestructopt.common.registry import registry
from pymatgen.core.structure import IStructure
from pymatgen.core import Lattice
import torch
import numpy as np

@registry.register_optimizer("Torch")
class Torch(BaseOptimizer):
  def __init__(self) -> None:
    pass

  def run(self, model: list[BaseModel], dataset: BaseDataset, 
    objective: BaseObjective, sampler: BaseSampler,
    constraints: list[BaseConstraint], N = 1,
    starts = 128, iters_per_start = 100, optimizer = "Adam",
    optimizer_args = {}, optimize_atoms = True,
    optimize_lattice = False, save_obj_values = False,
    pos_lr = 0.001, cell_lr = 0.001, random_starts = False,
    save_only_constrained_structures = False, min_distance = 0.25,
    **kwargs) -> list[IStructure]:
    """Returns the best structure found by each start, sorted by objective,
    keeping only the first N that are at least min_distance (Å, see
    optimizer.base.rms_distance) from each other and the dataset."""

    starting_structures = [sampler.sample(
      ) for j in range(starts)] if random_starts else [dataset.structures[j].copy(
      ) if j < dataset.N else sampler.sample(
      ) for j in range(starts)]

    obj_values = torch.zeros((iters_per_start, starts), device = 'cpu'
      ) if save_obj_values else None
    
    device = model.device
    nstarts = len(starting_structures)
    # Best objective, positions and cell found so far by each start
    best_objs = torch.full((nstarts,), float('inf'), device = device)
    best_pos = [None for _ in range(nstarts)]
    best_cells = [None for _ in range(nstarts)]
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

            constraint_vals = torch.zeros(stopi - starti + 1, device = device)
            for j in range(stopi - starti + 1):
              for constraint in constraints:
                constraint_vals[j] += constraint.soft_constraint(data[starti + j], data_style = 'mt')
              objs[j] += constraint_vals[j]
              obj_total += constraint_vals[j]
              objs[j] = objs[j].detach()
              if save_obj_values:
                obj_values[i, starti + j] = objs[j].detach().cpu()

            objs_to_compare = torch.nan_to_num(objs, nan = torch.inf)
            for j in range(stopi - starti + 1):
              if data[starti + j].pos.isnan().any() or (
                data[starti + j].cell[0].isnan().any()) or (
                objs_to_compare[j].isnan().any()):
                objs_to_compare[j] = torch.inf

            # Save Best Value of each start
            improved = objs_to_compare < best_objs[starti:(stopi+1)]
            if save_only_constrained_structures:
              improved = improved & (constraint_vals <= 0.0)
            for j in torch.nonzero(improved).flatten().tolist():
              best_objs[starti + j] = objs_to_compare[j].detach()
              best_pos[starti + j] = data[starti + j].pos.clone().detach()
              best_cells[starti + j] = data[starti + j].cell[0].clone().detach()

            if i != iters_per_start - 1:
              obj_total.backward()
              optimizer.step()
            del predictions, objs, obj_total
          predicted = True
        except torch.cuda.OutOfMemoryError:
          split -= 1
          assert split >= 0, "Out of memory with only one structure"

    del target, data

    best_objs = best_objs.cpu()
    order = [j for j in torch.argsort(best_objs).tolist() if
      best_pos[j] is not None and torch.isfinite(best_objs[j])]
    candidates = [to_structure(starting_structures[0], best_pos[j],
      best_cells[j]) for j in order]

    return select_distinct(candidates, dataset.structures, N,
      min_distance = min_distance), obj_values
