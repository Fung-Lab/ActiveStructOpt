"""Helpers for using an ASE calculator (e.g. a pretrained UMA MLIP) on the
MatterTune ORB AtomGraphs batches used by the 'mt' data style, with gradients
with respect to the positions and cell attached by hand, since the calculator
is not differentiable through torch."""
import numpy as np

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

def to_atoms(pos, numbers, cell, calculator):
  """ASE Atoms (float64) for one structure, with the calculator attached."""
  from ase import Atoms
  atoms = Atoms(numbers = numbers.detach().cpu().numpy(),
    positions = pos.detach().cpu().numpy().astype(np.float64),
    cell = cell.detach().cpu().numpy().astype(np.float64), pbc = True)
  atoms.calc = calculator
  return atoms

def cell_gradient(atoms):
  """dE/dh (eV/Å) at fixed Cartesian positions, with the lattice vectors as
  the rows of h and x = f h. The stress gives the derivative at fixed
  fractional coordinates f, V h^-T σ, so dE/dh = V h^-T σ + f^T F."""
  cell = atoms.cell.array
  inv_cell = np.linalg.inv(cell)
  forces = atoms.get_forces(apply_constraint = False)
  stress = atoms.get_stress(voigt = False, apply_constraint = False)
  frac = atoms.positions @ inv_cell
  return atoms.get_volume() * inv_cell.T @ stress + frac.T @ forces

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
