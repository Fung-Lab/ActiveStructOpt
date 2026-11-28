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

  def get_nuisance(self, device = 'cpu'):
    """None, or an object (see EXAFSNuisance in simulation/exafs.py) holding the
    structure-wide free parameters (e.g. an energy shift) that are fit
    between the simulated spectra and the target, rather than learned by the
    model. The model is then trained on the raw spectra, and the optimizer
    fits these parameters for each start alongside the structure, through
    n_params, default_lr, initial_params(prediction, target),
    apply(prediction, target, params) and project_(params)."""
    return None
