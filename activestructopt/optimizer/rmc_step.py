from activestructopt.model.base import BaseModel
from activestructopt.dataset.rmc_list import RMCList
from activestructopt.objective.base import BaseObjective
from activestructopt.optimizer.base import BaseOptimizer
from activestructopt.sampler.base import BaseSampler
from activestructopt.common.registry import registry
from pymatgen.core.structure import IStructure


@registry.register_optimizer("RMCStep")
class RMCStep(BaseOptimizer):
  def __init__(self) -> None:
    pass

  def run(self, models: list[BaseModel], dataset: RMCList, 
    objective: BaseObjective, sampler: BaseSampler, N = 1,
    **kwargs) -> list[IStructure]:
    if N != 1:
      raise NotImplementedError('RMCStep optimizer only suggests N = 1 structure')
    sampler.initial_structure = dataset.curr_structure
    return [sampler.sample()], None
