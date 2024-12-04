from activestructopt.common.dataloader import prepare_data_pmg
from activestructopt.common.constraints import lj_reject
from activestructopt.common.registry import registry
from activestructopt.dataset.base import BaseDataset
from activestructopt.simulation.base import BaseSimulation
from activestructopt.sampler.base import BaseSampler
from pymatgen.core.structure import IStructure, Structure
import numpy as np
import copy

@registry.register_dataset("KFoldsDataset")
class KFoldsDataset(BaseDataset):
  def __init__(self, simulation: BaseSimulation, sampler: BaseSampler, 
    initial_structure: IStructure, target, config, N = 100, split = 0.85, 
    k = 5, seed = 0, progress_dict = None, **kwargs) -> None:
    np.random.seed(seed)
    self.config = config
    self.target = target
    self.initial_structure = initial_structure
    self.start_N = N
    self.N = N
    self.k = k
    self.simfunc = simulation

    if progress_dict is None:
      self.structures = [initial_structure.copy(
        ) if i == 0 else sampler.sample() for i in range(N)]
      
      y_promises = [copy.deepcopy(simulation) for _ in self.structures]
      for i, s in enumerate(self.structures):
        y_promises[i].get(s, group = True, separator = ' ')
      self.ys = [yp.resolve() for yp in y_promises]
          
      structure_indices = np.random.permutation(np.arange(1, N))
      trainval_indices = structure_indices[:int(np.round(split * N) - 1)]
      trainval_indices = np.append(trainval_indices, [0])
      self.kfolds = np.array_split(trainval_indices, k)
      for i in range(self.k):
        self.kfolds[i] = self.kfolds[i].tolist()
      self.test_indices = structure_indices[int(np.round(split * N) - 1):]
      self.mismatches = [simulation.get_mismatch(y, target) for y in self.ys]
    else:
      self.start_N = progress_dict['start_N']
      self.N = progress_dict['N']
      self.structures = [Structure.from_dict(
        s) for s in progress_dict['structures']]
      self.ys = [np.array(y) for y in progress_dict['ys']]
      self.kfolds = progress_dict['kfolds']
      self.test_indices = np.array(progress_dict['test_indices'])
      self.mismatches = progress_dict['mismatches']

  def update(self, new_structure: IStructure):
    self.structures.append(new_structure)
    y_promise = self.simfunc
    y_promise.get(new_structure)
    y = y_promise.resolve()
    new_mismatch = self.simfunc.get_mismatch(y, self.target)
    y_promise.garbage_collect(new_mismatch <= min(self.mismatches))
    fold = len(self.datasets) - 1
    for i in range(len(self.datasets) - 1):
      if len(self.datasets[i][1]) < len(self.datasets[i + 1][1]):
        fold = i
        break
    self.kfolds[fold].append(len(self.structures) - 1)
    self.ys.append(y)
    self.mismatches.append(new_mismatch)
    self.N += 1

  def toJSONDict(self):
    return {
      'start_N': self.start_N,
      'N': self.N,
      'structures': [s.as_dict() for s in self.structures],
      'ys': [y.tolist() for y in self.ys],
      'kfolds': self.kfolds,
      'test_indices': [t.tolist() for t in self.test_indices],
      'mismatches': self.mismatches
    }
