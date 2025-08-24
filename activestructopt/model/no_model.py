from activestructopt.model.base import BaseModel
from activestructopt.common.registry import registry
from activestructopt.dataset.base import BaseDataset
import torch

@registry.register_model("NoModel")
class NoModel(BaseModel):
  def __init__(self, config, **kwargs):
    pass

  def train(self, dataset: BaseDataset, sim_index, **kwargs):
    return None, None, torch.empty(0)

  def predict(self, structure, **kwargs):
    return torch.empty(0)
