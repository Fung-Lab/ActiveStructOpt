from activestructopt.model.base import BaseModel
from activestructopt.common.registry import registry
from activestructopt.dataset.base import BaseDataset

@registry.register_model("GroundTruth")
class GroundTruth(BaseModel):
  def __init__(self, config, simfunc, **kwargs):
    self.simfunc = simfunc
    self.device = 'cuda'

  def train(self, dataset: BaseDataset, **kwargs):
    try:
      import torch
    except ImportError as e:
      raise ImportError('Error loading torch') from e
    return None, None, torch.empty(0)

  def predict(self, data, **kwargs):
    try:
      import torch
    except ImportError as e:
      raise ImportError('Error loading torch') from e
    gt = self.simfunc.get_and_resolve_prepared(data)
    unc = torch.zeros(gt.size(), device = gt.device)

    return torch.stack((gt, unc))
