from activestructopt.common.registry import registry
from activestructopt.constraint.base import BaseConstraint
from pymatgen.core.structure import IStructure
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np

class _BurnInDriftExceeded(Exception):
  pass

@registry.register_constraint("MLIPBurnIn")
class MLIPBurnIn(BaseConstraint):
  """Rejects structures that drift too far during the burn-in of the MLIP MD
  run used by the EXAFS simulation (run_md in simulation/exafs.py). Only the
  burn-in is run, and no trajectory or log files are written."""
  def __init__(self, mlip = 'uma-s-1p2p1', task_name = 'omat',
    inference_settings = 'nitro', device = 'cuda', timestep = 1.25,
    temp = 300, friction = 0.002, check_interval = 4, burnin_steps = 1600,
    max_rms_burnin = 0.5, minimum_length = 8.4, seed = 0, **kwargs):
    if mlip == 'uma-s-1p2p1':
      try:
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
      except ImportError as e:
        raise ImportError("Error loading fairchem.core") from e
      predictor = pretrained_mlip.get_predict_unit('uma-s-1p2p1',
        device = device, inference_settings = inference_settings)
      self.calculator = FAIRChemCalculator(predictor, task_name = task_name)
    else:
      raise NotImplementedError(f'MLIP {mlip} not implemented')

    self.adaptor = AseAtomsAdaptor()
    self.timestep = timestep
    self.temp = temp
    self.friction = friction
    self.check_interval = check_interval
    self.burnin_steps = burnin_steps
    self.max_rms_burnin = max_rms_burnin
    self.minimum_length = minimum_length
    self.seed = seed
    self.last_rms_displacement = None
    self.last_step = None

  def soft_constraint(self, data, **kwargs):
    raise NotImplementedError('MLIPBurnIn only implements hard constraints')

  def hard_constraint(self, structure: IStructure, **kwargs):
    """Returns True (rejected) if the RMS displacement from the starting
    positions exceeds max_rms_burnin during the burn-in."""
    try:
      from ase import units
      from ase.build.supercells import make_supercell
      from ase.constraints import FixCom
      from ase.md.langevin import Langevin
      from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    except ImportError as e:
      raise ImportError('Error loading ase') from e

    atoms = self.adaptor.get_atoms(structure)
    cell_lengths = np.linalg.norm(atoms.cell.array, axis = 1)
    repetitions = np.maximum(1, np.ceil(
      self.minimum_length / cell_lengths).astype(int))
    atoms = make_supercell(atoms, np.diag(repetitions))
    atoms.set_constraint(FixCom())
    atoms.calc = self.calculator

    MaxwellBoltzmannDistribution(atoms, temperature_K = self.temp,
      rng = np.random.RandomState(self.seed))
    initial_positions = atoms.get_scaled_positions().copy()

    dyn = Langevin(atoms,
      timestep = self.timestep * units.fs,
      temperature_K = self.temp,
      friction = self.friction / units.fs,
      fixcm = False)

    self.last_rms_displacement = 0.0
    self.last_step = 0

    def check_drift():
      if dyn.nsteps >= self.burnin_steps:
        return
      disp_frac = atoms.get_scaled_positions() - initial_positions
      disp_frac -= np.round(disp_frac)
      disp_cart = disp_frac @ atoms.cell.array
      rms_displacement = np.sqrt(np.mean(np.sum(disp_cart ** 2, axis = 1)))
      self.last_rms_displacement = rms_displacement
      self.last_step = dyn.nsteps
      if rms_displacement > self.max_rms_burnin:
        raise _BurnInDriftExceeded

    dyn.attach(check_drift, interval = self.check_interval)
    try:
      dyn.run(self.burnin_steps)
    except _BurnInDriftExceeded:
      return True
    return False
