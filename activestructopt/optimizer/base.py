from activestructopt.model.base import BaseModel
from activestructopt.dataset.base import BaseDataset
from activestructopt.objective.base import BaseObjective
from activestructopt.sampler.base import BaseSampler
from activestructopt.constraint.base import BaseConstraint
from pymatgen.core.structure import IStructure
from abc import ABC, abstractmethod
import numpy as np

class BaseOptimizer(ABC):
  @abstractmethod
  def __init__(self) -> None:
    pass

  @abstractmethod
  def run(self, model: list[BaseModel], dataset: BaseDataset,
    objective: BaseObjective, sampler: BaseSampler,
    constraints: list[BaseConstraint], N = 1, **kwargs):
    """Returns a list of up to N suggested structures, best first, and the
    saved objective values (or None)."""
    pass

def rms_distance(s1: IStructure, s2: IStructure):
  """RMS distance (Å) between corresponding sites of two structures with the
  same species in the same order, using each structure's own lattice, after
  removing the mean displacement (a rigid translation). Each site of s2 is
  taken at the periodic image nearest (in fractional coordinates) to the same
  site of s1. Returns inf if the sites do not correspond."""
  if len(s1) != len(s2) or s1.species != s2.species:
    return np.inf
  f1 = s1.frac_coords
  f2 = s2.frac_coords
  f2 = f2 + np.round(f1 - f2)
  disp = f1 @ s1.lattice.matrix - f2 @ s2.lattice.matrix
  disp -= np.mean(disp, axis = 0)
  return np.sqrt(np.mean(np.sum(disp ** 2, axis = 1)))

def select_distinct(candidates: list[IStructure],
  references: list[IStructure], N, min_distance = 0.25):
  """Goes through candidates (sorted best first) and keeps each one that is
  at least min_distance (Å, see rms_distance) from every reference structure
  (e.g. the dataset) and every candidate already kept, until N are kept. If
  none are distinct, the first candidate is returned alone."""
  selected = []
  for candidate in candidates:
    if all(rms_distance(candidate, s) >= min_distance
      for s in references + selected):
      selected.append(candidate)
      if len(selected) == N:
        break
  if len(selected) < N:
    print(f'WARNING: only {len(selected)} of {N} requested structures are at '
      f'least {min_distance} Å from each other and the dataset')
  if len(selected) == 0 and len(candidates) > 0:
    selected = [candidates[0]]
  return selected
