from activestructopt.model.base import BaseModel
from activestructopt.common.registry import registry
from activestructopt.dataset.base import BaseDataset
from pymatgen.core import Lattice
import torch
import copy

@registry.register_model("GroundTruth")
class GroundTruth(BaseModel):
  def __init__(self, config, simfunc, **kwargs):
    self.simfunc = simfunc
    self.device = 'cuda'

  def train(self, dataset: BaseDataset, **kwargs):
    return None, None, torch.empty(0)

  def predict(self, data, prepared = False, **kwargs):
    if prepared:
      gt = self.simfunc.get_and_resolve_prepared(data)
    else:
      gt = torch.zeros((len(data), self.simfunc.outdim), device = self.device)
      for i in range(len(data)):
        sim_promise = copy.deepcopy(self.simfunc)
        sim_promise.get(data[i])
        sim_spec = sim_promise.resolve()
        gt[i, :] = np.mean(sim_spec[np.array(sim_promise.mask)], axis = 0)

    unc = torch.zeros(gt.size(), device = gt.device)

    return torch.stack((gt, unc))
