from activestructopt.common.registry import registry
from activestructopt.energy.base import BaseEnergy

@registry.register_dataset("MTEnergy")
class MTEnergy(BaseEnergy):
  def __init__(self):
    from orb_models.forcefield.pretrained import orb_v3_direct_inf_omat
    self.model = orb_v3_direct_inf_omat(device = 'cuda:0')

  def get(self, batch):
    return self.model.forward(batch)['energy']
    