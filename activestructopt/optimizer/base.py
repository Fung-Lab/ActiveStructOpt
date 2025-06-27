from activestructopt.model.base import BaseModel
from activestructopt.dataset.base import BaseDataset
from activestructopt.objective.base import BaseObjective
from activestructopt.sampler.base import BaseSampler
from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
  @abstractmethod
  def __init__(self) -> None:
    pass

  @abstractmethod
  def run(self, model: list[BaseModel], dataset: BaseDataset, 
    objective: BaseObjective, sampler: BaseSampler, **kwargs):
    pass
