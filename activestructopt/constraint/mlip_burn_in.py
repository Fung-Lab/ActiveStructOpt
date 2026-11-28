from activestructopt.common.registry import registry
from activestructopt.common.mlip import get_mlip
from activestructopt.common.d2min import D2minMonitor, block_frames
from activestructopt.constraint.base import BaseConstraint
from pymatgen.core.structure import IStructure
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np

class _BurnInDriftExceeded(Exception):
  pass

def make_md_supercell(structure: IStructure, minimum_length = 8.4):
  """The ASE supercell used for the EXAFS MD (run_md in simulation/exafs.py):
  the structure repeated along each lattice vector until it is at least
  minimum_length long."""
  from ase.build.supercells import make_supercell
  atoms = AseAtomsAdaptor().get_atoms(structure)
  cell_lengths = np.linalg.norm(atoms.cell.array, axis = 1)
  repetitions = np.maximum(1, np.ceil(
    minimum_length / cell_lengths).astype(int))
  return make_supercell(atoms, np.diag(repetitions))

@registry.register_constraint("MLIPBurnIn")
class MLIPBurnIn(BaseConstraint):
  """Rejects structures that rearrange during the burn-in of the MLIP MD run
  used by the EXAFS simulation (run_md in simulation/exafs.py), with the same
  D^2_min check that run_md applies to the whole run (see common/d2min.py).
  Only the burn-in is run, and no trajectory or log files are written.

  A structure is rejected if, in any block of d2min_block_ps of the burn-in,
  the D^2_min of the block-averaged positions relative to the starting
  structure is above d2min_level for more than d2min_max_frac of the atoms,
  or above d2min_max_atom for any atom.
  """
  def __init__(self, mlip = 'uma-s-1p2p1', task_name = 'omat',
    inference_settings = 'nitro', device = 'cuda', timestep = 1.25,
    temp = 300, friction = 0.002, check_interval = 4, burnin_steps = 1600,
    d2min_block_ps = 0.5, d2min_cutoff = 3.5, d2min_level = 0.05,
    d2min_max_frac = 0.5, d2min_max_atom = 0.5,
    minimum_length = 8.4, seed = 0, **kwargs):
    # Shared and only loaded when first used (see common/mlip.py)
    self.mlip = get_mlip(mlip, task_name, inference_settings, device)

    self.timestep = timestep
    self.temp = temp
    self.friction = friction
    self.check_interval = check_interval
    self.burnin_steps = burnin_steps
    self.d2min_block_ps = d2min_block_ps
    self.d2min_cutoff = d2min_cutoff
    self.d2min_level = d2min_level
    self.d2min_max_frac = d2min_max_frac
    self.d2min_max_atom = d2min_max_atom
    self.minimum_length = minimum_length
    self.seed = seed
    # Diagnostics from the last call, for calibrating the thresholds: the
    # largest fraction of atoms above d2min_level and the largest atom
    # D^2_min (Å^2) over the blocks checked
    self.last_d2min_frac = None
    self.last_d2min_max = None
    self.last_step = None
    self.last_reason = None

  @property
  def calculator(self):
    return self.mlip.calculator

  def soft_constraint(self, data, **kwargs):
    raise NotImplementedError('MLIPBurnIn only implements hard constraints')

  def hard_constraint(self, structure: IStructure, **kwargs):
    """Returns True (rejected) if any of the burn-in criteria (see the class
    docstring) are violated."""
    try:
      from ase import units
      from ase.constraints import FixCom
      from ase.md.langevin import Langevin
      from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    except ImportError as e:
      raise ImportError('Error loading ase') from e

    atoms = make_md_supercell(structure, self.minimum_length)
    atoms.set_constraint(FixCom())
    atoms.calc = self.calculator

    MaxwellBoltzmannDistribution(atoms, temperature_K = self.temp,
      rng = np.random.RandomState(self.seed))
    monitor = D2minMonitor(atoms,
      block_frames(self.d2min_block_ps, self.timestep, self.check_interval),
      cutoff = self.d2min_cutoff, level = self.d2min_level,
      max_frac = self.d2min_max_frac, max_atom = self.d2min_max_atom)

    dyn = Langevin(atoms,
      timestep = self.timestep * units.fs,
      temperature_K = self.temp,
      friction = self.friction / units.fs,
      fixcm = False)

    self.last_d2min_frac = 0.0
    self.last_d2min_max = 0.0
    self.last_step = 0
    self.last_reason = None

    def check_d2min():
      self.last_step = dyn.nsteps
      reason = monitor.update(atoms)
      self.last_d2min_frac = monitor.max_frac_seen
      self.last_d2min_max = monitor.max_atom_seen
      if reason is not None:
        self.last_reason = reason
        raise _BurnInDriftExceeded

    dyn.attach(check_d2min, interval = self.check_interval)
    try:
      dyn.run(self.burnin_steps)
    except _BurnInDriftExceeded:
      return True
    return False
