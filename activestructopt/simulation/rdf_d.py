from activestructopt.simulation.base import BaseSimulation
from activestructopt.common.registry import registry
from activestructopt.common.dataloader import prepare_data_pmg
import numpy as np
from pymatgen.core.structure import IStructure
import torch

@registry.register_simulation("RDFD")
class RDFD(BaseSimulation):
  def __init__(self, initial_structure, σ = 0.05, dr = 0.01, max_r = 12.0, 
    device = 'cpu', **kwargs) -> None:
    self.σ = σ
    self.dr = dr
    self.max_r = max_r
    self.device = device
    self.rs = torch.tensor(np.arange(0.5, self.max_r, self.dr), 
      device = self.device)
    self.rlen = self.rs.size(0)
    self.natoms = len(initial_structure)
    self.normalization = 4 * self.natoms ** 2 * np.pi * np.sqrt(2 * np.pi * 
      self.σ ** 2) * torch.square(self.rs)
    
  def setup_config(self, config):
    self.config = config
    #config['dataset']['preprocess_params']['prediction_level'] = 'node'
    #config['dataset']['preprocess_params']['output_dim'] = self.outdim
    return config

  def get(self, data):
    if isinstance(data, IStructure):
      self.data = prepare_data_pmg(data, self.config['dataset'], 
        device = self.device, preprocess = True)
    else:
      self.data = data
    # https://github.com/Fung-Lab/MatDeepLearn_dev/blob/main/matdeeplearn/models/torchmd_etEarly.py#L230
    self.volume = torch.einsum("zi,zi->z", self.data.cell[:, 0, :], 
      torch.cross(self.data.cell[:, 1, :], self.data.cell[:, 2, :], dim = 1)
      ).unsqueeze(-1) 

  def resolve(self):
    ews = self.data.edge_weight
    elen = ews.size(0)
    rs = self.rs.repeat(elen, 1)
    ews = ews.unsqueeze(1)
    ews = ews.repeat(1, self.rlen)
    mdl_rdf = torch.sum(torch.exp(-torch.square((rs - ews) / (np.sqrt(2) * 
      self.σ))), axis = 0) * self.volume / self.normalization

    return mdl_rdf

  def garbage_collect(self, is_better):
    return

  def get_mismatch(self, to_compare, target):
    return np.mean((to_compare - target) ** 2)
