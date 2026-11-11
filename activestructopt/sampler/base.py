from activestructopt.constraint.base import BaseConstraint
from abc import ABC, abstractmethod
from pymatgen.core.structure import IStructure

class BaseSampler(ABC):
  @abstractmethod
  def __init__(self, initial_structure: IStructure, 
    constraints: list[BaseConstraint], **kwargs) -> None:
    pass

  @abstractmethod
  def sample(self) -> IStructure:
    pass
