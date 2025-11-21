from activestructopt.model.base import BaseModel
from activestructopt.common.registry import registry
from activestructopt.dataset.base import BaseDataset
from pymatgen.core import Lattice
import torch

@registry.register_model("GroundTruth")
class GroundTruth(BaseModel):
  def __init__(self, config, simfunc, **kwargs):
    self.simfunc = simfunc
    self.device = 'cuda'

  def train(self, dataset: BaseDataset, **kwargs):
    return None, None, torch.empty(0)

  def batch_pos_cell(pos, cells, init_struct):
    structs = []
    for i in range(len(pos)):
      new_struct = init_struct.copy()
      new_struct.lattice = Lattice(cell.detach().cpu().numpy())
      for j in range(len(new_struct)):
        new_struct.sites[i].coords = pos[i][j].detach().cpu().numpy()
      structs.append(new_struct)
    return structs

  def predict(self, data, **kwargs):
    gt = self.simfunc.get_and_resolve_prepared(data)
    unc = torch.zeros(gt.size(), device = gt.device)

    return torch.stack((gt, unc))
