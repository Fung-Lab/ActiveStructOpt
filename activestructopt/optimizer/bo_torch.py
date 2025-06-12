from activestructopt.model.base import BaseModel
from activestructopt.dataset.rmc_list import RMCList
from activestructopt.objective.base import BaseObjective
from activestructopt.optimizer.base import BaseOptimizer
from activestructopt.sampler.base import BaseSampler
from activestructopt.common.registry import registry
from pymatgen.core.structure import IStructure
import torch

@registry.register_optimizer("BOTorch")
class BOTorch(BaseOptimizer):
  def __init__(self) -> None:
    from botorch.optim import optimize_acqf

  def run(self, model: BaseModel, dataset: RMCList, 
    objective: BaseObjective, sampler: BaseSampler, **kwargs) -> IStructure:
    bounds = torch.stack([torch.zeros(dataset.num_atoms * 3), 
      torch.ones(dataset.num_atoms * 3)]).to(torch.double)
    candidate, _ = optimize_acqf(model.acqf, bounds = bounds, q = 1, 
      num_restarts = dataset.N, raw_samples = dataset.N
    )
    new_structure = dataset.structures[0].copy()
    for i in range(dataset.num_atoms):
        new_structure.sites[i].frac_coords = candidate[0][(i * 3):((i + 1) * 3)]
    return new_structure, None
