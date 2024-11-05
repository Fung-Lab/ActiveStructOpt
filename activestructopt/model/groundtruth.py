from activestructopt.model.base import BaseModel
from activestructopt.common.registry import registry
from activestructopt.dataset.base import BaseDataset
import torch

@registry.register_model("GroundTruth")
class GroundTruth(BaseModel):
  def __init__(self, config, simfunc, **kwargs):
    self.simfunc = simfunc
    self.device = 'cuda'

  def train(self, dataset: BaseDataset, **kwargs):
    return None, None, torch.empty(0)

  def predict(self, structure, **kwargs):
    self.simfunc.get(structure[0])
    gt = self.simfunc.resolve()
    unc = torch.zeros(gt.size(), device = gt.device)

    return torch.stack((gt, unc)).unsqueeze(1)
