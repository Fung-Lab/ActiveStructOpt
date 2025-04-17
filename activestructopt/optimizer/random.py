from activestructopt.model.base import BaseModel
from activestructopt.dataset.base import BaseDataset
from activestructopt.objective.base import BaseObjective
from activestructopt.optimizer.base import BaseOptimizer
from activestructopt.sampler.base import BaseSampler
from activestructopt.common.registry import registry
from pymatgen.core.structure import IStructure

@registry.register_optimizer("Random")
class Random(BaseOptimizer):
  def __init__(self) -> None:
    pass

  def run(self, model: BaseModel, dataset: BaseDataset, 
    objective: BaseObjective, sampler: BaseSampler, require_unique = False,
    **kwargs) -> IStructure:
    new_structure = sampler.sample()
    not_unique = True
    while not_unique:
      not_unique = False
      for s in dataset.structures:
        if new_structure.matches(s):
          not_unique = True
          new_structure = sampler.sample()
    return new_structure, None
