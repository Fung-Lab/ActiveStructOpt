from activestructopt.common.registry import registry
from activestructopt.common.mlip import get_mlip
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
  """Rejects structures that drift too far during the burn-in of the MLIP MD
  run used by the EXAFS simulation (run_md in simulation/exafs.py). Only the
  burn-in is run, and no trajectory or log files are written.

  A structure is rejected if, at any check during the burn-in:
    - the RMS displacement from the starting positions exceeds max_rms_burnin
    - any single atom's displacement exceeds max_atom_disp_burnin (if not None),
      which catches local rearrangements that the RMS averages away
    - after epot_skip_steps, the running mean (over the last epot_window
      checks) of (Epot - Epot_0 - Ekin) / N, in meV/atom, is below
      -max_epot_drop_burnin (if not None). Near a minimum, the average
      potential energy above the minimum equals the average kinetic energy
      (equipartition), so this goes clearly negative when the structure falls
      into a lower energy basin, or was not at a minimum to begin with.
  """
  def __init__(self, mlip = 'uma-s-1p2p1', task_name = 'omat',
    inference_settings = 'nitro', device = 'cuda', timestep = 1.25,
    temp = 300, friction = 0.002, check_interval = 4, burnin_steps = 1600,
    max_rms_burnin = 0.5, max_atom_disp_burnin = None,
    max_epot_drop_burnin = None, epot_skip_steps = 400, epot_window = 50,
    minimum_length = 8.4, seed = 0, **kwargs):
    # Shared and only loaded when first used (see common/mlip.py)
    self.mlip = get_mlip(mlip, task_name, inference_settings, device)

    self.timestep = timestep
    self.temp = temp
    self.friction = friction
    self.check_interval = check_interval
    self.burnin_steps = burnin_steps
    self.max_rms_burnin = max_rms_burnin
    self.max_atom_disp_burnin = max_atom_disp_burnin
    self.max_epot_drop_burnin = max_epot_drop_burnin
    self.epot_skip_steps = epot_skip_steps
    self.epot_window = epot_window
    self.minimum_length = minimum_length
    self.seed = seed
    # Diagnostics from the last call, for calibrating the thresholds
    self.last_rms_displacement = None
    self.last_max_atom_displacement = None
    self.last_epot_excess = None
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
    natoms = len(atoms)

    MaxwellBoltzmannDistribution(atoms, temperature_K = self.temp,
      rng = np.random.RandomState(self.seed))
    initial_positions = atoms.get_scaled_positions().copy()
    epot0 = atoms.get_potential_energy()

    dyn = Langevin(atoms,
      timestep = self.timestep * units.fs,
      temperature_K = self.temp,
      friction = self.friction / units.fs,
      fixcm = False)

    self.last_rms_displacement = 0.0
    self.last_max_atom_displacement = 0.0
    self.last_epot_excess = None
    self.last_step = 0
    self.last_reason = None
    epot_excesses = []

    def reject(reason):
      self.last_reason = reason
      raise _BurnInDriftExceeded

    def check_drift():
      if dyn.nsteps >= self.burnin_steps:
        return
      self.last_step = dyn.nsteps

      disp_frac = atoms.get_scaled_positions() - initial_positions
      disp_frac -= np.round(disp_frac)
      disp_cart = disp_frac @ atoms.cell.array
      atom_disps = np.linalg.norm(disp_cart, axis = 1)
      rms_displacement = np.sqrt(np.mean(atom_disps ** 2))
      self.last_rms_displacement = rms_displacement
      self.last_max_atom_displacement = np.max(atom_disps)
      if rms_displacement > self.max_rms_burnin:
        reject(f'RMS displacement {rms_displacement:.3f} Å')
      if (self.max_atom_disp_burnin is not None and
        self.last_max_atom_displacement > self.max_atom_disp_burnin):
        reject(f'Max atom displacement {self.last_max_atom_displacement:.3f} Å')

      if self.max_epot_drop_burnin is not None:
        epot_excesses.append(1000 * (atoms.get_potential_energy() - epot0 -
          atoms.get_kinetic_energy()) / natoms)
        if dyn.nsteps >= self.epot_skip_steps:
          self.last_epot_excess = np.mean(epot_excesses[-self.epot_window:])
          if self.last_epot_excess < -self.max_epot_drop_burnin:
            reject(f'Epot excess {self.last_epot_excess:.2f} meV/atom')

    dyn.attach(check_drift, interval = self.check_interval)
    try:
      dyn.run(self.burnin_steps)
    except _BurnInDriftExceeded:
      return True
    return False
