from abc import ABC, abstractmethod

class BaseEnergy(ABC):
  @abstractmethod
  def __init__(self):
    pass

  @abstractmethod
  def get(self, data):
    pass
