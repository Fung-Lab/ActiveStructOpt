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
      
      self.ys = [[None for _ in range(N)] for _ in range(len(self.simfuncs))]
      
      sim_calls = 0

      y_promises = [[copy.deepcopy(simulations[j]
        ) for _ in self.structures] for j in range(len(self.simfuncs))]
      if not call_sequential:
        for i, s in enumerate(self.structures):
          for j in range(len(self.simfuncs)):
            y_promises[j][i].get(s, group = True, separator = ' ')
      self.mismatches = [[np.NaN for _ in range(len(self.structures)
        )] for _ in range(len(self.simfuncs))]

      while self.sims_incomplete():
        sim_calls += 1
        for i in range(len(self.structures)):
          if self.sims_incomplete(s = i):
            try:
              for j in range(len(self.simfuncs)):
                if call_sequential:
                  y_promises[j][i].get(self.structures[i])
                self.ys[j][i] = y_promises[j][i].resolve()
                self.mismatches[j][i] = y_promises[j][i].get_mismatch(self.ys[j][i], targets[j])
                if self.mismatches[j][i] <= np.nanmin(self.mismatches[j]):
                  for k in range(len(self.structures)):
                    if type(self.ys[j][k]) != type(None) and i != k:
                      y_promises[j][k].garbage_collect(False)
                else:
                  y_promises[j][i].garbage_collect(False)
            except ASOSimulationException:
              if sim_calls <= max_sim_calls:
                # resample and try again
                self.structures[i] = sampler.sample()
                for j in range(len(self.simfuncs)):
                  y_promises[j][i].garbage_collect(False)
                  y_promises[j][i] = copy.deepcopy(simulations[j])
                  if not call_sequential:
                    y_promises[j][i].get(self.structures[i], group = True, 
                      separator = ' ')

      structure_indices = np.random.permutation(np.arange(1, N))
      trainval_indices = structure_indices[:int(np.round(split * N) - 1)]
      trainval_indices = np.append(trainval_indices, [0])
      self.kfolds = np.array_split(trainval_indices, self.k)
      for i in range(self.k):
        self.kfolds[i] = self.kfolds[i].tolist()

      self.test_indices = structure_indices[int(np.round(split * N) - 1):]
    else:
      self.start_N = progress_dict['start_N']
      self.N = progress_dict['N']
      self.structures = [Structure.from_dict(
        s) for s in progress_dict['structures']]
      pd_ys = progress_dict['ys']
      for i in range(len(pd_ys)):
        for j in range(len(pd_ys[i])):
          pd_ys[i][j] = np.array(pd_ys[i][j])
      self.ys = self.ys
      self.kfolds = progress_dict['kfolds']
      self.test_indices = np.array(progress_dict['test_indices'])
      self.mismatches = progress_dict['mismatches']

  def update(self, new_structure: IStructure):
    for j in range(len(self.simfuncs)):
      y_promise = copy.deepcopy(self.simfuncs[j])
      y_promise.get(new_structure)
      try:
        y = y_promise.resolve()
      except ASOSimulationException:
        y_promise.garbage_collect(False)
        raise ASOSimulationException
      
      new_mismatch = self.simfunc.get_mismatch(y, self.targets[j])
      y_promise.garbage_collect(new_mismatch <= min(self.mismatches[j]))
      self.ys[j].append(y)
      self.mismatches[j].append(new_mismatch)

    fold = self.k - 1
    for i in range(self.k - 1):
      if len(self.kfolds[i]) < len(self.kfolds[i + 1]):
        fold = i
        break

    self.structures.append(new_structure)
    print(fold)
    self.kfolds[fold].append(len(self.structures) - 1)
    print(self.kfolds)
    self.N += 1

  def toJSONDict(self, save_structures = True):
    ys_to_save = self.ys
    for i in range(len(ys_to_save)):
        for j in range(len(ys_to_save[i])):
          ys_to_save[i][j] = ys_to_save[i][j].tolist()
    return {
      'start_N': self.start_N,
      'N': self.N,
      'structures': [s.as_dict() for s in self.structures] if (
        save_structures) else self.structures[np.argmin(self.mismatches
        )].as_dict(),
      'ys': ys_to_save,
      'kfolds': self.kfolds,
      'test_indices': [t.tolist() for t in self.test_indices],
      'mismatches': self.mismatches
    }

  def sims_incomplete(self, s = None):
    for i in range(len(self.ys)):
      if s is None:
        for j in range(len(self.ys[i])):
          if type(self.ys[i][j]) == type(None):
            return True
      else:
        if type(self.ys[i][s]) == type(None):
          return True
    return False
