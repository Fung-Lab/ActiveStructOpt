from activestructopt.common.registry import registry
from activestructopt.energy.base import BaseEnergy
import torch

def denormalize(self, x, batch):
  """Denormalize the energy prediction."""
  x = self.normalizer.inverse(x).squeeze(-1)
  n_nodes = torch.ones(torch.sum(batch.n_node), device = x.device,
    dtype = batch.n_node.dtype)
  if self.atom_avg:
    x = x * n_nodes
  return torch.sum((x + self.reference(batch.atomic_numbers, n_nodes)).reshape((
    batch.n_node.shape[0], batch.n_node[0])), dim = 1)

@registry.register_energy("ORBEnergy")
class ORBEnergy(BaseEnergy):
  """Energy from the pretrained ORB v3 model, evaluated directly on the
  MatterTune ORB AtomGraphs batch."""
  def __init__(self, device = 'cuda:0', **kwargs):
    try:
      import orb_models
      from orb_models.forcefield.pretrained import orb_v3_direct_inf_omat
    except ImportError as e:
      raise ImportError('Error loading orb_models') from e
    orb_models.forcefield.forcefield_heads.EnergyHead.denormalize = denormalize
    self.model = orb_v3_direct_inf_omat(device = device)

  def get(self, data):
    return self.model.forward(data)['energy']
