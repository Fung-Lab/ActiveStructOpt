from activestructopt.common.registry import registry
from activestructopt.common.mlip import get_mlip
from activestructopt.constraint.base import BaseConstraint
from activestructopt.constraint.mlip_burn_in import make_md_supercell
from pymatgen.core.structure import IStructure
import numpy as np

@registry.register_constraint("MLIPSupercellRelax")
class MLIPSupercellRelax(BaseConstraint):
  """Rejects structures that are not at a local minimum in the supercell used
  for the EXAFS MD. The structure is built into that supercell, rattled to
  break its symmetry (so that saddle points, e.g. ones that are only minima
  in the smaller cell, can relax away), and its positions are relaxed with
  the cell fixed, as in the NVT MD.

  A structure is rejected if any atom ends more than max_disp (Å) from its
  starting position, if the energy drops by more than max_energy_drop
  (meV/atom), or (if require_converged) if the relaxation does not converge.
  Tight convergence is not needed: at a minimum the rattle relaxes back within
  a few tens of steps, and from a saddle FIRE moves well away within a few
  hundred, so the relaxation is capped at steps rather than run to
  convergence.
  """
  def __init__(self, mlip = 'uma-s-1p2p1', task_name = 'omat',
    inference_settings = 'nitro', device = 'cuda', minimum_length = 8.4,
    rattle = 0.05, fmax = 0.08, steps = 300, require_converged = False,
    max_disp = 0.3, max_energy_drop = 2.0, seed = 0, **kwargs):
    # Shared and only loaded when first used (see common/mlip.py)
    self.mlip = get_mlip(mlip, task_name, inference_settings, device)

    self.minimum_length = minimum_length
    self.rattle = rattle
    self.fmax = fmax
    self.steps = steps
    self.require_converged = require_converged
    self.max_disp = max_disp
    self.max_energy_drop = max_energy_drop
    self.seed = seed
    # Diagnostics from the last call, for calibrating the thresholds
    self.last_converged = None
    self.last_max_disp = None
    self.last_energy_drop = None
    self.last_reason = None

  @property
  def calculator(self):
    return self.mlip.calculator

  def soft_constraint(self, data, **kwargs):
    raise NotImplementedError(
      'MLIPSupercellRelax only implements hard constraints')

  def hard_constraint(self, structure: IStructure, **kwargs):
    """Returns True (rejected) if the structure moves or drops in energy too
    much when re-relaxed in the MD supercell."""
    try:
      from ase.optimize import FIRE
    except ImportError as e:
      raise ImportError('Error loading ase') from e

    atoms = make_md_supercell(structure, self.minimum_length)
    atoms.calc = self.calculator
    natoms = len(atoms)
    e0 = atoms.get_potential_energy()
    p0 = atoms.positions.copy()

    atoms.rattle(stdev = self.rattle, seed = self.seed)
    FIRE(atoms, logfile = None).run(fmax = self.fmax, steps = self.steps)

    forces = atoms.get_forces()
    self.last_converged = bool(np.all(np.isfinite(forces)) and
      np.max(np.linalg.norm(forces, axis = 1)) <= self.fmax)
    # Positions are not wrapped during the relaxation, so no minimum image
    disps = np.linalg.norm(atoms.positions - p0, axis = 1)
    self.last_max_disp = np.max(disps)
    self.last_energy_drop = 1000 * (e0 - atoms.get_potential_energy()
      ) / natoms
    atoms.calc = None

    self.last_reason = None
    if not np.all(np.isfinite(forces)):
      self.last_reason = 'Non-finite forces'
    elif self.require_converged and not self.last_converged:
      self.last_reason = 'Relaxation did not converge'
    elif not self.last_max_disp <= self.max_disp:
      self.last_reason = f'Max displacement {self.last_max_disp:.3f} Å'
    elif not self.last_energy_drop <= self.max_energy_drop:
      self.last_reason = f'Energy drop {self.last_energy_drop:.2f} meV/atom'
    return self.last_reason is not None
