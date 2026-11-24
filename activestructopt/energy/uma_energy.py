from activestructopt.common.registry import registry
from activestructopt.energy.base import BaseEnergy
import numpy as np

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
      from ase import Atoms
    except ImportError as e:
      raise ImportError('Error loading torch or ase') from e

    # MatterTune ORB AtomGraphs; the positions are already wrapped into the
    # cell, keeping their gradient with respect to the positions and cell
    n_node = data.n_node.detach().cpu().tolist()
    positions = torch.split(data.node_features['positions'], n_node)
    numbers = torch.split(data.atomic_numbers, n_node)
    cells = data.system_features['cell']

    energies = []
    for pos, z, cell in zip(positions, numbers, cells):
      pos_np = pos.detach().cpu().numpy().astype(np.float64)
      cell_np = cell.detach().cpu().numpy().astype(np.float64)
      atoms = Atoms(numbers = z.detach().cpu().numpy(), positions = pos_np,
        cell = cell_np, pbc = True)
      atoms.calc = self.calculator
      energy = atoms.get_potential_energy()
      forces = atoms.get_forces(apply_constraint = False)

      # Value: the MLIP energy. Gradient: -forces for the positions.
      e = torch.tensor(energy, dtype = pos.dtype, device = pos.device)
      e = e - torch.sum(torch.tensor(forces, dtype = pos.dtype,
        device = pos.device) * (pos - pos.detach()))

      if cell.requires_grad:
        # With the lattice vectors as the rows of h and Cartesian positions
        # x = f h, the stress gives the derivative at fixed fractional
        # coordinates f, V h^-T σ. At fixed Cartesian positions, which the
        # positions above already account for, dE/dh = V h^-T σ + f^T F.
        stress = atoms.get_stress(voigt = False, apply_constraint = False)
        inv_cell = np.linalg.inv(cell_np)
        frac = pos_np @ inv_cell
        de_dcell = atoms.get_volume() * inv_cell.T @ stress + frac.T @ forces
        e = e + torch.sum(torch.tensor(de_dcell, dtype = cell.dtype,
          device = cell.device) * (cell - cell.detach()))

      energies.append(e)
    return torch.stack(energies)
