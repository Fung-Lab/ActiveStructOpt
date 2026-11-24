from activestructopt.common.registry import registry
from activestructopt.common.mlip import (get_mlip, mt_systems, to_atoms,
  cell_gradient, attach_gradient)
from activestructopt.constraint.base import BaseConstraint
from pymatgen.core.structure import IStructure
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np

@registry.register_constraint("MLIPForces")
class MLIPForces(BaseConstraint):
  """Penalizes the largest MLIP force magnitude above force_cutoff (eV/Å):
  the soft constraint is max_i max(0, |F_i| - force_cutoff)^2, and the hard
  constraint rejects structures with any |F_i| > force_cutoff. The MLIP is
  shared with the other users of the same MLIP settings (see
  common/mlip.py)."""
  def __init__(self, mlip = 'uma-s-1p2p1', task_name = 'omat',
    inference_settings = 'nitro', device = 'cuda', force_cutoff = 3.0,
    fd_step = 0.01, **kwargs):
    self.mlip = get_mlip(mlip, task_name, inference_settings, device)
    self.adaptor = AseAtomsAdaptor()
    self.force_cutoff = force_cutoff
    # Step (Å) of the finite difference of the forces used for the gradient
    # of the 'mt' soft constraint
    self.fd_step = fd_step

  def soft_constraint(self, data, data_style = 'pmg', **kwargs):
    if data_style == 'pmg':
      forces = self.mlip.calculate(self.adaptor.get_atoms(data))['forces']
      force_mags = np.linalg.norm(forces, axis = 1)
      return np.max(np.maximum(0, (force_mags - self.force_cutoff)) ** 2)
    elif data_style == 'mt':
      try:
        import torch
      except ImportError as e:
        raise ImportError('Error loading torch') from e
      return torch.stack([self._mt_penalty(pos, numbers, cell)
        for pos, numbers, cell in mt_systems(data)])
    else:
      raise NotImplementedError(f'Data style {data_style} not implemented')

  def _mt_penalty(self, pos, numbers, cell):
    """The penalty of one structure, with its gradient with respect to the
    positions (and the cell, if it requires a gradient) attached.

    With k the atom with the largest force, P = max(0, |F_k| - c)^2, so
    dP/dx = sum_i v_i . dF_i/dx with v_k = 2 max(0, |F_k| - c) F_k / |F_k|
    (zero for the other atoms). Since F = -dE/dx, this is -H v for the
    Hessian H, found from a central finite difference of the forces along v:
    dP/dx = (F(x + εv̂) - F(x - εv̂)) |v| / (2ε). Likewise, from the
    derivative dE/dh of the energy with respect to the cell at fixed
    positions, dP/dh = -(dE/dh(x + εv̂) - dE/dh(x - εv̂)) |v| / (2ε).
    Below the cutoff, the penalty and its gradient are zero and no extra
    MLIP calls are made."""
    atoms = to_atoms(pos, numbers, cell)
    forces = self.mlip.calculate(atoms)['forces']
    force_mags = np.linalg.norm(forces, axis = 1)
    k = np.argmax(force_mags)
    excess = force_mags[k] - self.force_cutoff
    need_cell = cell.requires_grad

    if excess <= 0:
      return attach_gradient(0.0, pos, cell, np.zeros_like(forces),
        np.zeros((3, 3)) if need_cell else None)

    v_norm = 2 * excess
    direction = np.zeros_like(forces)
    direction[k] = forces[k] / force_mags[k]

    displaced = []
    for sign in [1, -1]:
      d_atoms = atoms.copy()
      d_atoms.positions = atoms.positions + sign * self.fd_step * direction
      d_results = self.mlip.calculate(d_atoms)
      displaced.append((d_results['forces'],
        cell_gradient(d_atoms, d_results) if need_cell else None))

    scale = v_norm / (2 * self.fd_step)
    grad_pos = (displaced[0][0] - displaced[1][0]) * scale
    grad_cell = -(displaced[0][1] - displaced[1][1]) * scale if (
      need_cell) else None
    return attach_gradient(excess ** 2, pos, cell, grad_pos, grad_cell)

  def hard_constraint(self, structure: IStructure, **kwargs):
    forces = self.mlip.calculate(self.adaptor.get_atoms(structure))['forces']
    return np.max(np.linalg.norm(forces, axis = 1)) > self.force_cutoff
