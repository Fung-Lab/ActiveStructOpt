from activestructopt.common.registry import registry
from activestructopt.common.mlip import (get_mlip, mt_systems, to_atoms,
  cell_gradient, attach_gradient)
from activestructopt.energy.base import BaseEnergy

@registry.register_energy("UMAEnergy")
class UMAEnergy(BaseEnergy):
  """Energy from a pretrained UMA MLIP (shared with the other users of the
  same MLIP settings, see common/mlip.py) on each structure of the batch.

  The calculator is not differentiable through torch, so the returned energy
  is attached to the graph with the MLIP's own derivatives: the forces give
  dE/dpositions, and the stress gives dE/dcell (only if the cell requires a
  gradient). Its value is the MLIP energy, and its gradient with respect to
  the positions and cell is the MLIP gradient."""
  def __init__(self, mlip = 'uma-s-1p2p1', task_name = 'omat',
    inference_settings = 'nitro', device = 'cuda', **kwargs):
    self.mlip = get_mlip(mlip, task_name, inference_settings, device)

  def get(self, data):
    try:
      import torch
    except ImportError as e:
      raise ImportError('Error loading torch') from e

    energies = []
    for pos, numbers, cell in mt_systems(data):
      atoms = to_atoms(pos, numbers, cell)
      results = self.mlip.calculate(atoms)
      energies.append(attach_gradient(results['energy'], pos, cell,
        -results['forces'],
        cell_gradient(atoms, results) if cell.requires_grad else None))
    return torch.stack(energies)
