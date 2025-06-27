from activestructopt.common.registry import registry
from activestructopt.dataset.base import BaseDataset
from activestructopt.simulation.base import BaseSimulation, ASOSimulationException
from activestructopt.sampler.base import BaseSampler
from pymatgen.core.structure import IStructure, Structure
import numpy as np
import copy

@registry.register_dataset("KFoldsDataset")
class KFoldsDataset(BaseDataset):
  def __init__(self, simulations: list[BaseSimulation], sampler: BaseSampler, 
    initial_structure: IStructure, targets, config, N = 100, split = 0.85, 
    k = 5, seed = 0, progress_dict = None, max_sim_calls = 5, 
    call_sequential = False,
    **kwargs) -> None:
    np.random.seed(seed)
    self.config = config
    self.targets = targets
    self.initial_structure = initial_structure
    self.start_N = N
    self.N = N
    self.k = k
    self.simfuncs = simulations

    if progress_dict is None:
      self.structures = [initial_structure.copy(
        ) if i == 0 else sampler.sample() for i in range(N)]
      
      

      y_promises = [copy.deepcopy(simulations[j]) for _ in self.structures]
      if not call_sequential:
        for i, s in enumerate(self.structures):
          y_promises[i].get(s, group = True, separator = ' ')
      self.ys = [None for _ in y_promises]
      self.mismatches = [np.NaN for _ in y_promises]
      
      sim_calls = 0
      while any(y is None for y in self.ys):
        sim_calls += 1
        for i in range(len(self.structures)):
          if self.ys[i] is None:
            try:
              if call_sequential:
                y_promises[i].get(self.structures[i])
              self.ys[i] = y_promises[i].resolve()
              self.mismatches[i] = y_promises[i].get_mismatch(self.ys[i], targets[j])
              if self.mismatches[i] <= np.nanmin(self.mismatches):
                for j in range(len(self.structures)):
                  if (self.ys[j] is not None) and i != j:
                    y_promises[j].garbage_collect(False)
              else:
                y_promises[i].garbage_collect(False)
            except ASOSimulationException:
              y_promises[i].garbage_collect(False)
              if sim_calls <= max_sim_calls:
                # resample and try again
                self.structures[i] = sampler.sample()
                y_promises[i] = copy.deepcopy(simulations[j])
                if not call_sequential:
                  y_promises[i].get(self.structures[i], group = True, 
                    separator = ' ')

      structure_indices = np.random.permutation(np.arange(1, N))
      trainval_indices = structure_indices[:int(np.round(split * N) - 1)]
      trainval_indices = np.append(trainval_indices, [0])
      self.kfolds = np.array_split(trainval_indices, k)
      for i in range(self.k):
        self.kfolds[i] = self.kfolds[i].tolist()
      self.test_indices = structure_indices[int(np.round(split * N) - 1):]
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
    y_promise = self.simfunc
    y_promise.get(new_structure)
    try:
      y = y_promise.resolve()
    except ASOSimulationException:
      y_promise.garbage_collect(False)
      raise ASOSimulationException
    self.structures.append(new_structure)
    new_mismatch = self.simfunc.get_mismatch(y, self.target)
    y_promise.garbage_collect(new_mismatch <= min(self.mismatches))
    fold = self.k - 1
    for i in range(self.k - 1):
      if len(self.kfolds[i]) < len(self.kfolds[i + 1]):
        fold = i
        break
    self.kfolds[fold].append(len(self.structures) - 1)
    self.ys.append(y)
    self.mismatches.append(new_mismatch)
    self.N += 1

  def toJSONDict(self, save_structures = True):
    return {
      'start_N': self.start_N,
      'N': self.N,
      'structures': [s.as_dict() for s in self.structures] if (
        save_structures) else self.structures[np.argmin(self.mismatches)].as_dict(),
      'ys': [y.tolist() for y in self.ys],
      'kfolds': self.kfolds,
      'test_indices': [t.tolist() for t in self.test_indices],
      'mismatches': self.mismatches
    }
