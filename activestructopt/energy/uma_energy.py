from activestructopt.common.registry import registry
from activestructopt.common.mlip import (mt_systems, to_atoms, cell_gradient,
  attach_gradient)
from activestructopt.energy.base import BaseEnergy

@registry.register_energy("UMAEnergy")
class UMAEnergy(BaseEnergy):
  """Energy from a pretrained UMA MLIP, evaluated with its ASE calculator (as
  in constraint/mlip_forces.py) on each structure of the batch.

  The calculator is not differentiable through torch, so the returned energy
  is attached to the graph with the MLIP's own derivatives: the forces give
  dE/dpositions, and the stress gives dE/dcell (only computed if the cell
  requires a gradient). Its value is the MLIP energy, and its gradient with
  respect to the positions and cell is the MLIP gradient."""
  def __init__(self, mlip = 'uma-s-1p2p1', task_name = 'omat',
    inference_settings = 'nitro', device = 'cuda', **kwargs):
    try:
      from fairchem.core import FAIRChemCalculator, pretrained_mlip
    except ImportError as e:
      raise ImportError("Error loading fairchem.core") from e
    predictor = pretrained_mlip.get_predict_unit(mlip, device = device,
      inference_settings = inference_settings)
    self.calculator = FAIRChemCalculator(predictor, task_name = task_name)

  def get(self, data):
    try:
      import torch
    except ImportError as e:
      raise ImportError('Error loading torch') from e

    energies = []
    for pos, numbers, cell in mt_systems(data):
      atoms = to_atoms(pos, numbers, cell, self.calculator)
      energy = atoms.get_potential_energy()
      forces = atoms.get_forces(apply_constraint = False)
      energies.append(attach_gradient(energy, pos, cell, -forces,
        cell_gradient(atoms) if cell.requires_grad else None))
    return torch.stack(energies)
