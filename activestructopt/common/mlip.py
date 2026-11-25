"""A pretrained MLIP (e.g. UMA) shared by everything that uses it in a process
(e.g. the UMAEnergy energy model and the MLIPForces constraint), and helpers
for using it on the MatterTune ORB AtomGraphs batches of the 'mt' data style,
with gradients with respect to the positions and cell attached by hand, since
the ASE calculator is not differentiable through torch."""
from collections import OrderedDict
import numpy as np

# Shared MLIPs, keyed by (mlip, task_name, inference_settings, device)
_MLIPS = {}

def get_mlip(mlip = 'uma-s-1p2p1', task_name = 'omat',
  inference_settings = 'nitro', device = 'cuda'):
  """The MLIP with these settings, shared so the model is loaded and kept in
  GPU memory once per process. The model itself is only loaded when first
  used (see MLIP.calculator), so a process that builds the constraints and
  energy models but never evaluates them (e.g. the main process when the
  optimization runs in sbatch GPU jobs) never touches the GPU."""
  key = (mlip, task_name, inference_settings, device)
  if key not in _MLIPS:
    _MLIPS[key] = MLIP(*key)
  return _MLIPS[key]

class MLIP:
  """A pretrained MLIP with its ASE calculator, and a cache of its results so
  that users evaluating the same structure (e.g. an energy model and a
  constraint in the same optimizer iteration) only compute it once."""
  def __init__(self, mlip, task_name, inference_settings, device,
    cache_size = 4096):
    if not mlip.startswith('uma'):
      raise NotImplementedError(f'MLIP {mlip} not implemented')
    self.mlip = mlip
    self.task_name = task_name
    self.inference_settings = inference_settings
    self.device = device
    self._calculator = None
    self.cache = OrderedDict()
    self.cache_size = cache_size

  @property
  def calculator(self):
    """The ASE calculator, loading the model on first access."""
    if self._calculator is None:
      try:
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
      except ImportError as e:
        raise ImportError("Error loading fairchem.core") from e
      predictor = pretrained_mlip.get_predict_unit(self.mlip,
        device = self.device, inference_settings = self.inference_settings)
      self._calculator = FAIRChemCalculator(predictor,
        task_name = self.task_name)
    return self._calculator

  def calculate(self, atoms):
    """A dict of the energy (eV), forces (eV/Å, (natoms, 3)) and stress
    (eV/Å^3, (3, 3)) of atoms, which is not modified."""
    key = (atoms.numbers.tobytes(), atoms.positions.tobytes(),
      atoms.cell.array.tobytes())
    if key in self.cache:
      self.cache.move_to_end(key)
      return self.cache[key]

    atoms = atoms.copy()
    atoms.calc = self.calculator
    results = {
      'energy': atoms.get_potential_energy(),
      'forces': atoms.get_forces(apply_constraint = False),
      'stress': atoms.get_stress(voigt = False, apply_constraint = False),
    }
    self.cache[key] = results
    if len(self.cache) > self.cache_size:
      self.cache.popitem(last = False)
    return results

def mt_systems(data):
  """Splits a MatterTune ORB AtomGraphs batch into (positions, atomic numbers,
  cell) tensors for each structure. The positions are already wrapped into
  the cell, keeping their gradient with respect to the positions and cell."""
  import torch
  n_node = data.n_node.detach().cpu().tolist()
  positions = torch.split(data.node_features['positions'], n_node)
  numbers = torch.split(data.atomic_numbers, n_node)
  cells = data.system_features['cell']
  return list(zip(positions, numbers, cells))

def to_atoms(pos, numbers, cell):
  """ASE Atoms (float64) for one structure."""
  from ase import Atoms
  return Atoms(numbers = numbers.detach().cpu().numpy(),
    positions = pos.detach().cpu().numpy().astype(np.float64),
    cell = cell.detach().cpu().numpy().astype(np.float64), pbc = True)

def cell_gradient(atoms, results):
  """dE/dh (eV/Å) at fixed Cartesian positions, with the lattice vectors as
  the rows of h and x = f h, from the MLIP results for atoms. The stress
  gives the derivative at fixed fractional coordinates f, V h^-T σ, so
  dE/dh = V h^-T σ + f^T F."""
  inv_cell = np.linalg.inv(atoms.cell.array)
  frac = atoms.positions @ inv_cell
  return atoms.get_volume() * inv_cell.T @ results['stress'] + (
    frac.T @ results['forces'])

def attach_gradient(value, pos, cell, grad_pos, grad_cell = None):
  """A tensor equal to value whose gradient with respect to pos (and cell,
  if grad_cell is given) is grad_pos (and grad_cell)."""
  import torch
  out = torch.tensor(value, dtype = pos.dtype, device = pos.device)
  out = out + torch.sum(torch.tensor(grad_pos, dtype = pos.dtype,
    device = pos.device) * (pos - pos.detach()))
  if grad_cell is not None:
    out = out + torch.sum(torch.tensor(grad_cell, dtype = cell.dtype,
      device = cell.device) * (cell - cell.detach()))
  return out
