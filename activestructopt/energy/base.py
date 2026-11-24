from abc import ABC, abstractmethod

class BaseEnergy(ABC):
  @abstractmethod
  def __init__(self, **kwargs):
    pass

  @abstractmethod
  def get(self, data):
    """Returns a tensor of the energy (eV) of each structure in the batch
    data (the MatterTune ORB AtomGraphs used by the 'mt' data style),
    differentiable with respect to the positions and cell."""
    pass
