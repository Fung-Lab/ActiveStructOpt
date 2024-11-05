from activestructopt.common.dataloader import prepare_data_pmg
from activestructopt.common.constraints import lj_reject
from activestructopt.common.registry import registry
from activestructopt.dataset.base import BaseDataset
from activestructopt.simulation.base import BaseSimulation
from activestructopt.sampler.base import BaseSampler
from pymatgen.core.structure import IStructure
import numpy as np
import copy

@registry.register_dataset("KFoldsDataset")
class KFoldsDataset(BaseDataset):
  def __init__(self, simulation: BaseSimulation, sampler: BaseSampler, 
    initial_structure: IStructure, target, config, N = 100, split = 0.85, 
    k = 5, device = 'cuda', seed = 0, **kwargs) -> None:
    np.random.seed(seed)
    self.device = device
    self.config = config
    self.target = target
    self.initial_structure = initial_structure
    self.start_N = N
    self.N = N
    self.k = k
    self.simfunc = simulation
    self.structures = [initial_structure.copy(
      ) if i == 0 else sampler.sample() for i in range(N)]
    
    y_promises = [copy.deepcopy(simulation) for _ in self.structures]
    for i, s in enumerate(self.structures):
      y_promises[i].get(s)
    self.ys = [yp.resolve() for yp in y_promises]
    data = [prepare_data_pmg(self.structures[i], config, y = self.ys[i]).to(
      self.device) for i in range(N)]
        
    structure_indices = np.random.permutation(np.arange(1, N))
    trainval_indices = structure_indices[:int(np.round(split * N) - 1)]
    trainval_indices = np.append(trainval_indices, [0])
    self.kfolds = np.array_split(trainval_indices, k)
    self.test_indices = structure_indices[int(np.round(split * N) - 1):]
    self.test_data = [data[i] for i in self.test_indices]
    self.test_targets = [self.ys[i] for i in self.test_indices]
    train_indices = [np.concatenate(
      [self.kfolds[j] for j in range(k) if j != i]) for i in range(k)]
    
    self.datasets = [([data[j] for j in train_indices[i]], 
      [data[j] for j in self.kfolds[i]]) for i in range(k)]

    self.mismatches = [simulation.get_mismatch(y, target) for y in self.ys]

  def update(self, new_structure: IStructure):
    self.structures.append(new_structure)
    y_promise = self.simfunc
    y_promise.get(new_structure)
    y = y_promise.resolve()
    new_mismatch = self.simfunc.get_mismatch(y, self.target)
    y_promise.garbage_collect(new_mismatch <= min(self.mismatches))
    new_data = prepare_data_pmg(new_structure, self.config, y = y).to(self.device)
    fold = len(self.datasets) - 1
    for i in range(len(self.datasets) - 1):
      if len(self.datasets[i][1]) < len(self.datasets[i + 1][1]):
        fold = i
        break
    self.datasets[fold][1].append(new_data)
    for i in range(len(self.datasets)):
      if fold != i:
        self.datasets[i][0].append(new_data)
    self.ys.append(y)
    self.mismatches.append(new_mismatch)
    self.N += 1

  def toJSONDict(self):
    return {
      'start_N': self.start_N,
      'N': self.N,
      'structures': [s.as_dict() for s in self.structures],
      'ys': [y.tolist() for y in self.ys],
      'kfolds': self.kfolds.tolist(),
      'test_indices': self.test_indices.tolist(),
      'test_indices': self.test_indices.tolist(),
      'train_indices': [t.tolist() for t in self.train_indices],
      'mismatches': self.mismatches
    }
