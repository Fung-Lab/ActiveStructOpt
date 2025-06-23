from activestructopt.common.registry import registry
from activestructopt.energy.base import BaseEnergy
import torch

def denormalize(self, x, batch):
    """Denormalize the energy prediction."""
    x = self.normalizer.inverse(x).squeeze(-1)
    n_nodes = torch.ones(torch.sum(batch.n_node), device = x.device)
    if self.atom_avg:
        x = x * n_nodes
    return x + self.reference(batch.atomic_numbers, n_nodes)

def normalize(self, x, batch, reference, online):
    """Normalize the energy prediction."""
    n_nodes = torch.ones(torch.sum(batch.n_node), device = x.device)
    if reference is None:
        reference = self.reference(batch.atomic_numbers, n_nodes)
    x = x - reference
    if self.atom_avg:
        x = x / n_nodes
    return self.normalizer(x, online=online)

@registry.register_dataset("MTEnergy")
class MTEnergy(BaseEnergy):
  def __init__(self):
    import orb_models
    from orb_models.forcefield.pretrained import orb_v3_direct_inf_omat
    orb_models.forcefield.forcefield_heads.EnergyHead.denormalize = denormalize
    orb_models.forcefield.forcefield_heads.EnergyHead.normalize = normalize
    self.model = orb_models.forcefield.pretrained.orb_v3_direct_inf_omat(device = 'cuda:0')

  def get(self, batch):
    return self.model.forward(batch)['energy']
    