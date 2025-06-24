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

@registry.register_energy("MTEnergy")
class MTEnergy(BaseEnergy):
  def __init__(self):
    import orb_models
    from orb_models.forcefield.pretrained import orb_v3_direct_inf_omat
    orb_models.forcefield.forcefield_heads.EnergyHead.denormalize = denormalize
    self.model = orb_v3_direct_inf_omat(device = 'cuda:0')

  def get(self, batch):
    return self.model.forward(batch)['energy']
    