from activestructopt.common.registry import registry
from activestructopt.constraints
from pymatgen.core.structure import IStructure
import torch
import periodictable
from pathlib import Path
import numpy as np

@registry.register_constraint("LennardJonesRepulsion")
class LennardJonesRepulsion(BaseConstraint)
  def __init__(self, scale = 40, buffer = 0.85,)
    self.lj_rmins = np.genfromtxt(str(Path(__file__
      ).parent / 'lj_rmins.csv'), delimiter = ',') * self.buffer
    self.el_symbols = np.array([periodictable.elements[
      i].symbol for i in range(95)])
    self.scale = scale
    self.buffer = buffer

  def get_z(site):
    return np.argmax(self.el_symbols == site.species.elements[0].symbol)

  def soft_constraint(self, data, data_style = ""):
    if data_style == 'mdl':
      rmins = self.lj_rmins[(data.z[data.edge_index[0]] - 1), 
        (data.z[data.edge_index[1]] - 1)] 
      repulsions = torch.where(rmins > data.edge_weight, 
        torch.pow(rmins / data.edge_weight, 12), 1.0)
      return (torch.mean(repulsions) - 1) / self.scale
    elif data_style == 'mt':
      edge_splitter = data.n_edge.to('cpu').detach().tolist()
      senders = torch.split(data.senders, edge_splitter)
      receivers = torch.split(data.receivers, edge_splitter)
      edge_weights = torch.split(torch.norm(data.edge_features['vectors'], dim = 1), 
        edge_splitter)

      to_return = [None for _ in range(len(senders))]

      for i in range(len(senders)):
        senders_z = data.atomic_numbers[senders[i]]
        receivers_z = data.atomic_numbers[receivers[i]]

        rmins = self.lj_rmins[(senders_z - 1), (receivers_z - 1)]
        repulsions = torch.where(rmins > edge_weights[i], 
          torch.pow(rmins / edge_weights[i], 12), 1.0)
        to_return[i] = ((torch.mean(repulsions) - 1) / self.scale).clone()
      return to_return
    elif data_style == 'pmg':
      repulsions = []
      for i in range(len(structure)):
        for j in range(i, len(structure)):
          rmin = self.lj_rmins[self.get_z(structure.sites[i]) - 1, self.get_z(
            structure.sites[j]) - 1]
          r = np.min([structure.lattice.a, structure.lattice.b, 
            structure.lattice.c]) if i == j else structure.sites[i].distance(
            structure.sites[j])
          repulsions.append(max(0, (rmin / r) ** 12 - 1))
      return np.mean(repulsions) / self.scale
    else:
      raise NotImplementedError(f'Data style {data_style} not implemented')

  def hard_constraint(self, structure: IStructure)
    for i in range(len(structure)):
      for j in range(i + 1, len(structure)):
        if structure.sites[i].distance(structure.sites[j]) < (self.lj_rmins[self.get_z(
          structure.sites[i]) - 1][self.get_z(structure.sites[j]) - 1]):
          return True
    return False
