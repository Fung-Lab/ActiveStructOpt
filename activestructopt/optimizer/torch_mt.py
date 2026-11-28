from activestructopt.model.base import BaseModel
from activestructopt.dataset.base import BaseDataset
from activestructopt.objective.base import BaseObjective
from activestructopt.optimizer.base import BaseOptimizer, BestPerStart
from activestructopt.sampler.base import BaseSampler
from activestructopt.constraint.base import BaseConstraint
from activestructopt.energy.base import BaseEnergy
from activestructopt.common.registry import registry
from pymatgen.core.structure import IStructure
from pymatgen.core import Lattice
import torch
import numpy as np

@registry.register_optimizer("TorchMT")
class TorchMT(BaseOptimizer):
  def __init__(self) -> None:
    pass

  def run(self, models: list[BaseModel], dataset: BaseDataset,
    objective: BaseObjective, sampler: BaseSampler,
    constraints: list[BaseConstraint], N = 1, starts = 128,
    iters_per_start = 100, optimizer = "Adam",
    optimizer_args = {}, optimize_atoms = True,
    optimize_lattice = False, save_obj_values = False,
    pos_lr = 0.001, cell_lr = 0.001, random_starts = False,
    save_only_constrained_structures = False, min_distance = 0.25,
    energy_model: BaseEnergy = None, energy_scale = 0.0,
    nuisance_lr = None, **kwargs) -> list[IStructure]:
    """Returns the best structure found by each start, sorted by objective,
    keeping only the first N that are at least min_distance (Å, see
    optimizer.base.rms_distance) from each other and the dataset.

    If energy_model is given and energy_scale is nonzero, energy_scale times
    the energy (eV) of each structure is added to its objective.

    If a simulation has structure-wide nuisance parameters (see
    BaseSimulation.get_nuisance, e.g. ΔE0 of EXAFS), each start has its own,
    initialized from the predictions for its starting structure (a grid
    search for ΔE0) and optimized alongside the structure with learning rate
    nuisance_lr (or the simulation's default). The model's predictions are
    of the raw spectra, which the nuisance fit maps onto the target before
    the objective."""
    use_energy = energy_model is not None and energy_scale != 0.0

    starting_structures = [sampler.sample(
      ) for _ in range(starts)] if random_starts else [dataset.structures[
      j].copy() if j < dataset.N else sampler.sample(
      ) for j in range(starts)]

    obj_values = torch.zeros((len(models), iters_per_start, starts),
      device = 'cpu') if save_obj_values else None

    device = models[0].device
    nstarts = len(starting_structures)
    # Best structures found by each start. The unmodified starts taken from
    # the dataset are not recorded, as they have already been simulated.
    best = BestPerStart(nstarts, device,
      save_only_constrained = save_only_constrained_structures,
      skip_first = [(not random_starts) and j < dataset.N
        for j in range(nstarts)])
    targets = [torch.tensor(target, device = device) for target in dataset.targets]

    data_pos = [torch.Tensor([site.coords.tolist() for site in struct.sites]
      ).to(device) for struct in starting_structures]
    data_cell = [torch.Tensor(struct.lattice.matrix.tolist()).to(device
      ) for struct in starting_structures]

    # Nuisance parameters of each model's simulation, one tensor per start
    nuisances = [getattr(dataset.simfuncs[m], 'get_nuisance',
      lambda device: None)(device) for m in range(len(models))]
    nuisance_params = [None if nu is None else [torch.zeros(nu.n_params,
      device = device, requires_grad = True) for _ in range(nstarts)]
      for nu in nuisances]

    to_optimize = []
    if optimize_atoms:
      to_optimize += [{'params': p, 'lr': pos_lr} for p in data_pos]
    if optimize_lattice:
      to_optimize += [{'params': c, 'lr': cell_lr} for c in data_cell]
    for nu, params in zip(nuisances, nuisance_params):
      if nu is not None:
        to_optimize += [{'params': p, 'lr': nu.default_lr if (
          nuisance_lr is None) else nuisance_lr} for p in params]
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
              data_cell[j].requires_grad_(False)
              data_pos[j].requires_grad_(False)

            for j in range(stopi - starti + 1):
              if optimize_atoms:
                data_pos[starti + j].requires_grad_()
              if optimize_lattice:
                data_cell[starti + j].requires_grad_()

            batch_data = models[0].batch_pos_cell(
              data_pos[starti:(stopi+1)], data_cell[starti:(stopi+1)],
              starting_structures[0])

            # Get Predictions
            predictions = [models[m].predict(batch_data, prepared = True,
              mask = dataset.simfuncs[m].mask) for m in range(len(models))]

            # Fit of the nuisance parameters, initialized at the first
            # iteration from the predictions for the starting structures
            for m, nu in enumerate(nuisances):
              if nu is None:
                continue
              batch_params = nuisance_params[m][starti:(stopi + 1)]
              if i == 0:
                init = nu.initial_params(predictions[m], targets[m])
                with torch.no_grad():
                  for j, p in enumerate(batch_params):
                    p.copy_(init[j])
              predictions[m] = nu.apply(predictions[m], targets[m],
                torch.stack(batch_params))

            # Calculate Objectives
            objs, obj_total = objective.get(predictions, targets,
              device = device, N = stopi - starti + 1, M = len(models))

            # Calculate Constraints
            constraint_vals = torch.zeros(stopi - starti + 1, device = device)
            for constraint in constraints:
              constraint_vals += constraint.soft_constraint(batch_data, data_style = 'mt')

            # Calculate Energies
            energy_vals = energy_scale * energy_model.get(batch_data
              ) if use_energy else torch.zeros(stopi - starti + 1,
              device = device)

            for j in range(stopi - starti + 1):
              for m in range(len(models)):
                objs[m][j] += (constraint_vals[j] + energy_vals[j]
                  ) / len(models)
                objs[m][j] = objs[m][j].detach()
                if save_obj_values:
                  obj_values[m, i, starti + j] = objs[m][j].detach().cpu()
              obj_total += constraint_vals[j] + energy_vals[j]

            objs_to_compare = torch.sum(torch.nan_to_num(objs, nan = torch.inf),
              dim = 0)

            for j in range(stopi - starti + 1):
              if data_pos[starti + j].isnan().any() or (
                data_cell[starti + j].isnan().any()) or (
                objs_to_compare[j].isnan().any()):
                objs_to_compare[j] = torch.inf

            # Save Best Value of each start
            best.update(i, starti, objs_to_compare, constraint_vals,
              data_pos[starti:(stopi+1)], data_cell[starti:(stopi+1)])

            if i != iters_per_start - 1:
              obj_total.backward()
              optimizer.step()
              for nu, params in zip(nuisances, nuisance_params):
                if nu is not None:
                  for p in params[starti:(stopi + 1)]:
                    nu.project_(p)

            del predictions, objs, obj_total, energy_vals
          predicted = True
        except torch.cuda.OutOfMemoryError:
          split -= 1
          assert split >= 0, "Out of memory with only one structure"
        except RuntimeError as e:
          if "out of memory" in str(e) or "unknown error" in str(e):
            split -= 1
            assert split >= 0, "Out of memory with only one structure"
          else:
            raise e

    del targets, data_pos, data_cell, nuisance_params

    return best.select(starting_structures[0], dataset.structures, N,
      min_distance = min_distance), obj_values
