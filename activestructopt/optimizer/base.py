from activestructopt.model.base import BaseModel
from activestructopt.dataset.base import BaseDataset
from activestructopt.objective.base import BaseObjective
from activestructopt.sampler.base import BaseSampler
from activestructopt.constraint.base import BaseConstraint
from pymatgen.core.structure import IStructure
from pymatgen.core import Lattice
from abc import ABC, abstractmethod
import numpy as np
import torch

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

def select_distinct(candidates, references: list[IStructure], N,
  min_distance = 0.25):
  """Goes through candidates (an iterable, best first) and keeps each one
  that is at least min_distance (Å, see rms_distance) from every reference
  structure (e.g. the dataset) and every candidate already kept, until N are
  kept. May return fewer than N, or none."""
  selected = []
  if N <= 0:
    return selected
  for candidate in candidates:
    if all(rms_distance(candidate, s) >= min_distance
      for s in references + selected):
      selected.append(candidate)
      if len(selected) == N:
        break
  return selected

def to_structure(template: IStructure, pos, cell) -> IStructure:
  """A copy of template with the given Cartesian positions and cell
  tensors."""
  new_structure = template.copy()
  new_cell = cell.cpu().numpy()
  new_x = pos.cpu().numpy().reshape(-1, 3)
  new_structure.lattice = Lattice(new_cell)
  for i in range(len(new_structure)):
    try:
      new_structure[i].coords = new_x[i]
    except np.linalg.LinAlgError as e:
      print(new_cell)
      print(new_x)
      raise e
  return new_structure

class BestPerStart:
  """Tracks the best structure found by each start of a multi-start
  optimizer, and selects the suggested structures from them.

  For each start, the main record is the structure with the lowest objective
  (among those with no constraint violation, if save_only_constrained). If
  save_only_constrained, a fallback record is also kept: the structure with
  the smallest constraint violation, ties broken by the objective.

  The first evaluation (the unmodified starting structure) of the starts in
  skip_first is never recorded: for starts taken from the dataset, it is a
  structure that has already been simulated."""
  def __init__(self, nstarts, device, save_only_constrained = False,
    skip_first = None):
    self.save_only_constrained = save_only_constrained
    self.skip_first = torch.tensor([False] * nstarts if skip_first is None
      else skip_first, dtype = torch.bool, device = device)
    self.objs = torch.full((nstarts,), float('inf'), device = device)
    self.pos = [None] * nstarts
    self.cells = [None] * nstarts
    if save_only_constrained:
      self.fb_violations = torch.full((nstarts,), float('inf'),
        device = device)
      self.fb_objs = torch.full((nstarts,), float('inf'), device = device)
      self.fb_pos = [None] * nstarts
      self.fb_cells = [None] * nstarts

  def update(self, i, starti, objs, violations, pos, cells):
    """Records the structures of starts starti, starti + 1, ... evaluated at
    iteration i, with objectives objs, constraint violations violations
    (>= 0), positions pos and cells cells."""
    with torch.no_grad():
      n = len(objs)
      objs = torch.nan_to_num(objs.detach(), nan = float('inf'))
      violations = torch.nan_to_num(violations.detach(), nan = float('inf'))
      allowed = torch.isfinite(objs)
      if i == 0:
        allowed = allowed & ~self.skip_first[starti:(starti + n)]

      improved = allowed & (objs < self.objs[starti:(starti + n)])
      if self.save_only_constrained:
        improved = improved & (violations <= 0.0)
      for j in torch.nonzero(improved).flatten().tolist():
        self.objs[starti + j] = objs[j]
        self.pos[starti + j] = pos[j].clone().detach()
        self.cells[starti + j] = cells[j].clone().detach()

      if self.save_only_constrained:
        fb_viol = self.fb_violations[starti:(starti + n)]
        fb_obj = self.fb_objs[starti:(starti + n)]
        fb_improved = allowed & ((violations < fb_viol) | (
          (violations == fb_viol) & (objs < fb_obj)))
        for j in torch.nonzero(fb_improved).flatten().tolist():
          self.fb_violations[starti + j] = violations[j]
          self.fb_objs[starti + j] = objs[j]
          self.fb_pos[starti + j] = pos[j].clone().detach()
          self.fb_cells[starti + j] = cells[j].clone().detach()

  def select(self, template: IStructure, references: list[IStructure], N,
    min_distance = 0.25):
    """Up to N structures, each at least min_distance (Å) from the
    references and each other: first the main records, by objective, then
    (if save_only_constrained and too few are found) the fallback records,
    by constraint violation, then objective. Structures are built from the
    records lazily, as they are checked."""
    objs = self.objs.cpu().numpy()
    order = [j for j in np.argsort(objs, kind = 'stable')
      if self.pos[j] is not None]
    selected = select_distinct((to_structure(template, self.pos[j],
      self.cells[j]) for j in order), references, N, min_distance)
    n_main = len(selected)

    if self.save_only_constrained and n_main < N:
      fb_viol = self.fb_violations.cpu().numpy()
      fb_objs = self.fb_objs.cpu().numpy()
      fb_order = [j for j in np.lexsort((fb_objs, fb_viol))
        if self.fb_pos[j] is not None]
      selected += select_distinct((to_structure(template, self.fb_pos[j],
        self.fb_cells[j]) for j in fb_order), references + selected,
        N - n_main, min_distance)
      print(f'WARNING: only {n_main} of {N} suggested structures satisfy '
        f'the constraints; {len(selected) - n_main} more are the least '
        'violating')

    if len(selected) < N:
      print(f'WARNING: only {len(selected)} of {N} requested structures are '
        f'at least {min_distance} Å from each other and the dataset')
    return selected
