from activestructopt.common.registry import registry
from activestructopt.energy.base import BaseEnergy
import torch

def denormalize(self, x, batch):
    """Denormalize the energy prediction."""
    x = self.normalizer.inverse(x).squeeze(-1)
    if self.atom_avg:
        x = x * torch.ones(torch.sum(batch.n_node), device = x.device)
    return x + self.reference(batch.atomic_numbers, batch.n_node)

def normalize(self, x, batch, reference, online):
    """Normalize the energy prediction."""
    if reference is None:
        reference = self.reference(batch.atomic_numbers, batch.n_node)
    x = x - reference
    if self.atom_avg:
        x = x / torch.ones(torch.sum(batch.n_node), device = x.device)
    return self.normalizer(x, online=online)

@registry.register_dataset("MTEnergy")
class MTEnergy(BaseEnergy):
  def __init__(self):
    from orb_models.forcefield.pretrained import orb_v3_direct_inf_omat
    orb_v3_direct_inf_omat.heads.energy.denormalize = denormalize
    orb_v3_direct_inf_omat.heads.energy.normalize = normalize
    self.model = orb_v3_direct_inf_omat(device = 'cuda:0')

  def get(self, batch):
    return self.model.forward(batch)['energy']
    