from abc import ABC, abstractmethod
from pymatgen.core.structure import IStructure

class ASOSimulationException(Exception):
    pass

class BaseSimulation(ABC):
  @abstractmethod
  def __init__(self, initial_structure: IStructure, **kwargs) -> None:
    pass

  @abstractmethod
  def setup_config(self, config):
    pass

  @abstractmethod
  def get(self, struct: IStructure):
    pass

  @abstractmethod
  def resolve(self):
    pass

  @abstractmethod
  def garbage_collect(self, is_better):
    pass

  @abstractmethod
  def get_mismatch(self, to_compare, target):
    pass
