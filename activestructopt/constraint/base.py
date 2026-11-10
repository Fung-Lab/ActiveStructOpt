from abc import ABC, abstractmethod
from pymatgen.core.structure import IStructure

class BaseConstraint(ABC):
  @abstractmethod
  def __init__(self, **kwargs):
    pass

  @abstractmethod
  def soft_constraint(self, data, **kwargs):
    pass

  @abstractmethod
  def hard_constraint(self, structure: IStructure, **kwargs):
    pass
