from activestructopt.dataset.base import BaseDataset
from activestructopt.simulation.base import BaseSimulation
from activestructopt.sampler.base import BaseSampler
from pymatgen.core.structure import IStructure
from activestructopt.common.registry import registry
import numpy as np
import torch
import copy

@registry.register_dataset("BOSet")
class BOSet(BaseDataset):
  def __init__(self, simulation: list[BaseSimulation], sampler: BaseSampler, 
    initial_structure: IStructure, target, config, seed = 0, N = 30, 
    **kwargs) -> None:
    np.random.seed(seed)
    self.simfunc = simulation
    self.N = N
    self.start_N = N
    self.target = target
    self.num_atoms = len(initial_structure)
    self.structures = [initial_structure.copy()]
    for i in range(N - 1):
      self.structures.append(sampler.sample())
    self.X = torch.zeros(N, self.num_atoms * 3, dtype=torch.double)
    for i in range(N):
      for j in range(self.num_atoms):
        self.X[i, (j * 3):((j + 1) * 3)] = torch.tensor(
          self.structures[i].frac_coords[j], dtype = torch.double)
    y_promises = [copy.deepcopy(self.simfunc) for _ in self.structures]
    for i, s in enumerate(self.structures):
      y_promises[i].get(s)
    self.ys = [yp.resolve() for yp in y_promises]
    self.mismatches = [simulation.get_mismatch(y, target) for y in self.ys]
    Y = -torch.tensor(self.mismatches, dtype=torch.double)
    self.Y = -torch.unsqueeze(Y, 1)

  def update(self, new_structure: IStructure):
    self.structures.append(new_structure)
    self.N += 1
    y_promise = copy.deepcopy(self.simfunc)
    y_promise.get(new_structure)
    new_y = y_promise.resolve()
    new_mismatch = self.simfunc.get_mismatch(new_y, self.target)
    self.mismatches.append(new_mismatch)
    newY = torch.zeros(self.N, 1, dtype=torch.double)
    newY[:-1] = self.Y
    newY[-1] = -new_mismatch
    newX = torch.zeros(self.N, self.num_atoms * 3, dtype=torch.double)
    newX[:-1] = self.X
    for j in range(self.num_atoms):
      newX[-1, (j * 3):((j + 1) * 3)] = torch.tensor(
        new_structure.frac_coords[j], dtype = torch.double)
    self.X = newX
    self.Y = newY
