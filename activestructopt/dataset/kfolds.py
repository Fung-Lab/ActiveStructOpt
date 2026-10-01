from activestructopt.common.registry import registry
from activestructopt.dataset.base import BaseDataset
from activestructopt.simulation.base import BaseSimulation, ASOSimulationException
from activestructopt.sampler.base import BaseSampler
from pymatgen.core.structure import IStructure, Structure
import numpy as np
import copy
import time

@registry.register_dataset("KFoldsDataset")
class KFoldsDataset(BaseDataset):
  def __init__(self, simulations: list[BaseSimulation], sampler: BaseSampler, 
    initial_structure: IStructure, targets, config, N = 100, split = 0.85, 
    k = 5, seed = 0, progress_dict = None, max_sim_calls = 5, sim_time_limit = 60 * 180,
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
        ) if i == 0 else sampler.sample() for i in range(self.N)]
      
      self.ys = [[None for _ in range(self.N)] for _ in range(len(
        self.simfuncs))]
      self.mismatches = [[np.nan for _ in range(len(self.structures)
        )] for _ in range(len(self.simfuncs))]

      y_promises = [[copy.deepcopy(self.simfuncs[j]
        ) for _ in self.structures] for j in range(len(self.simfuncs))]
      for i, s in enumerate(self.structures):
        time.sleep(5)
        for j in range(len(self.simfuncs)):
          y_promises[j][i].get(s, group = True, separator = ' ')

      sim_calls = [1 for _ in range(self.N)]
      sim_timers = [0 for _ in range(self.N)]
      while self.sims_incomplete():
        for i in range(self.N):
          if self.sims_incomplete(s = i):
            sim_timers[i] += 30
            try:
              for j in range(len(self.simfuncs)):
                if self.ys[j][i] is None:
                  if y_promises[j][i].check_done():
                    self.ys[j][i] = y_promises[j][i].resolve()
                    self.mismatches[j][i] = y_promises[j][i].get_mismatch(
                      self.ys[j][i], targets[j])
                    if not np.isfinite(self.mismatches[j][i]):
                      raise ASOSimulationException('NaN Mismatch')
                    if self.mismatches[j][i] <= np.nanmin(self.mismatches[j]):
                      for k in range(self.N):
                        if self.ys[j][k] is not None and i != k:
                          y_promises[j][k].garbage_collect(False)
                    else:
                      y_promises[j][i].garbage_collect(False)
                  elif sim_timers[i] > sim_time_limit:
                    raise ASOSimulationException('Simulation exceeded time limit')
            except ASOSimulationException as e:
              if sim_calls[i] < max_sim_calls:
                # resample and try again
                print(f'retrying structure {i}')
                print(f'exception: {e}')
                self.structures[i] = sampler.sample()
                for j in range(len(self.simfuncs)):
                  self.ys[j][i] = None
                  self.mismatches[j][i] = np.nan
                  y_promises[j][i].garbage_collect(False)
                  y_promises[j][i] = copy.deepcopy(self.simfuncs[j])
                  time.sleep(5)
                  y_promises[j][i].get(self.structures[i], group = True, 
                    separator = ' ')
                sim_calls[i] += 1
                sim_timers[i] = 0
              else:
                raise RuntimeError(f"Max simulation calls reached for structure {i}")
        time.sleep(30)

      structure_indices = np.random.permutation(np.arange(1, self.N))
      trainval_indices = structure_indices[:int(np.round(split * self.N) - 1)]
      trainval_indices = np.append(trainval_indices, [0])
      self.kfolds = np.array_split(trainval_indices, self.k)
      for i in range(self.k):
        self.kfolds[i] = self.kfolds[i].tolist()

      self.test_indices = structure_indices[int(np.round(split * self.N) - 1):]
    else:
      self.start_N = progress_dict['start_N']
      self.N = progress_dict['N']
      self.structures = [Structure.from_dict(
        s) for s in progress_dict['structures']]
      pd_ys = progress_dict['ys']
      for i in range(len(pd_ys)):
        for j in range(len(pd_ys[i])):
          pd_ys[i][j] = np.array(pd_ys[i][j])
      self.ys = pd_ys
      self.kfolds = progress_dict['kfolds']
      self.test_indices = np.array(progress_dict['test_indices'])
      self.mismatches = progress_dict['mismatches']

  def update(self, new_structure: IStructure):
    new_ys = [None for _ in range(len(self.simfuncs))]
    new_mismatches = [None for _ in range(len(self.simfuncs))]
    for j in range(len(self.simfuncs)):
      y_promise = copy.deepcopy(self.simfuncs[j])
      y_promise.get(new_structure)
      try:
        y = y_promise.resolve()
      except ASOSimulationException:
        y_promise.garbage_collect(False)
        raise ASOSimulationException
      
      new_mismatch = self.simfuncs[j].get_mismatch(y, self.targets[j])
      y_promise.garbage_collect(new_mismatch <= min(self.mismatches[j]))

      new_ys[j] = y
      new_mismatches[j] = new_mismatch
      
    for j in range(len(self.simfuncs)):  
      self.ys[j].append(new_ys[j])
      self.mismatches[j].append(new_mismatches[j])

    fold = self.k - 1
    for i in range(self.k - 1):
      if len(self.kfolds[i]) < len(self.kfolds[i + 1]):
        fold = i
        break

    self.structures.append(new_structure)
    self.kfolds[fold].append(len(self.structures) - 1)
    self.N += 1

  def toJSONDict(self, save_structures = True):
    ys_to_save = self.ys
    for i in range(len(ys_to_save)):
      for j in range(len(ys_to_save[i])):
        if not isinstance(ys_to_save[i][j], list):
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
