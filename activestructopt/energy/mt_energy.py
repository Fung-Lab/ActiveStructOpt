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
  return x + self.reference(batch.atomic_numbers, n_nodes)

def normalize(self, x, batch, reference, online):
  """Normalize the energy prediction."""
  n_nodes = torch.ones(torch.sum(batch.n_node), device = x.device, 
    dtype = batch.n_node.dtype)
  if reference is None:
    reference = self.reference(batch.atomic_numbers, n_nodes)
  x = x - reference
  if self.atom_avg:
    x = x / n_nodes
  return self.normalizer(x, online=online)

def forward(self, batch):
  """Forward pass of DirectForcefieldRegressor."""
  out = self.model(batch)
  node_features = out["node_features"]
  for name, head in self.heads.items():
    res = head(node_features, batch)
    out[name] = res

  if self.pair_repulsion:
    out_pair_repulsion = self.pair_repulsion_fn(batch)
    for name, head in self.heads.items():
      raw_repulsion = self._get_raw_repulsion(name, out_pair_repulsion)
      if raw_repulsion is not None:
        raw = head.denormalize(out[name], batch)
        print(raw)
        print(raw_repulsion)
        out[name] = head.normalize(raw + raw_repulsion, batch, online=False)
  return out


@registry.register_dataset("MTEnergy")
class MTEnergy(BaseEnergy):
  def __init__(self):
    import orb_models
    from orb_models.forcefield.pretrained import orb_v3_direct_inf_omat
    orb_models.forcefield.forcefield_heads.EnergyHead.denormalize = denormalize
    orb_models.forcefield.forcefield_heads.EnergyHead.normalize = normalize
    orb_models.forcefield.direct_regressor.DirectForcefieldRegressor.forward = forward
    self.model = orb_models.forcefield.pretrained.orb_v3_direct_inf_omat(
      device = 'cuda:0')

  def get(self, batch):
    return self.model.forward(batch)['energy']
    