from activestructopt.simulation.base import BaseSimulation, ASOSimulationException
from activestructopt.dataset.base import BaseDataset
from activestructopt.sampler.base import BaseSampler
from activestructopt.common.registry import registry
from pymatgen.core.structure import IStructure, Structure
import numpy as np
import copy
import time

@registry.register_dataset("RMCList")
class RMCList(BaseDataset):
  def __init__(self, simulations: list[BaseSimulation], sampler: BaseSampler, 
    initial_structure: IStructure, targets, config, seed = 0, σ = 0.0025, 
    max_sim_calls = 5, weights = None, progress_dict = None,
    **kwargs) -> None:
    np.random.seed(seed)
    self.config = config
    self.targets = targets
    self.initial_structure = initial_structure
    self.start_N = 1
    self.N = 1
    self.simfuncs = simulations
    self.weights = weights

    if progress_dict is None:
      self.structures = [initial_structure.copy()]
      
      self.ys = [[None for _ in range(self.N)] for _ in range(len(
        self.simfuncs))]
      
      sim_calls = 0

      y_promises = [[copy.deepcopy(self.simfuncs[j]
        ) for _ in self.structures] for j in range(len(self.simfuncs))]
      self.mismatches = [[np.NaN for _ in range(len(self.structures)
        )] for _ in range(len(self.simfuncs))]

      while self.sims_incomplete():
        sim_calls += 1
        sim_updated = False
        for i in range(self.N):
          if self.sims_incomplete(s = i):
            sim_updated = True
            try:
              for j in range(len(self.simfuncs)):
                y_promises[j][i].get(self.structures[i])
                self.ys[j][i] = y_promises[j][i].resolve()
                self.mismatches[j][i] = y_promises[j][i].get_mismatch(
                  self.ys[j][i], targets[j])
                if self.mismatches[j][i] <= np.nanmin(self.mismatches[j]):
                  for k in range(self.N):
                    if type(self.ys[j][k]) != type(None) and i != k:
                      y_promises[j][k].garbage_collect(False)
                else:
                  y_promises[j][i].garbage_collect(False)
            except ASOSimulationException:
              if sim_calls <= max_sim_calls:
                # resample and try again
                print(f'retrying structure {i}')
                self.structures[i] = sampler.sample()
                for j in range(len(self.simfuncs)):
                  y_promises[j][i].garbage_collect(False)
                  y_promises[j][i] = copy.deepcopy(self.simfuncs[j])
              else:
                raise Exception(f'Max sim calls reached for structure {i}')
        assert sim_updated

      self.curr_structure = self.structures[0]
      new_mismatch = 0
      for j in range(len(self.simfuncs)):  
        new_mismatch += self.mismatches[0][j] if self.weights is None else (
          self.mismatches[0][j] * self.weights[j])
      self.curr_mismatch = new_mismatch
      self.accepted = [True]
      self.σ = σ
    else:
      pd_ys = progress_dict['ys']
      for i in range(len(pd_ys)):
        for j in range(len(pd_ys[i])):
          pd_ys[i][j] = np.array(pd_ys[i][j])
      self.ys = pd_ys
      self.mismatches = progress_dict['mismatches']
      self.start_N = progress_dict['start_N']
      self.N = progress_dict['N']
      self.structures = []
      self.curr_structure = Structure.from_dict(progress_dict['curr_structure'])
      self.curr_mismatch = progress_dict['curr_mismatch']
      self.accepted = progress_dict['accepted']

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
      
    new_mismatch = 0
    for j in range(len(self.simfuncs)):  
      self.ys[j].append(new_ys[j])
      self.mismatches[j].append(new_mismatches[j])
      new_mismatch += new_mismatches[j] if self.weights is None else (
        new_mismatches[j] * self.weights[j])

    Δmse = new_mismatch - self.curr_mismatch
    accept = (Δmse <= 0 or np.log(np.random.rand()) < -Δmse/(2 * self.σ ** 2))

    self.structures.append(new_structure)
    self.accepted.append(accept)
    self.N += 1

    if accept:
      self.curr_structure = new_structure
      self.curr_mismatch = new_mismatch
  
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
        save_structures) else self.structures[np.argmin(self.mismatches)].as_dict(),
      'ys': ys_to_save,
      'mismatches': self.mismatches,
      'curr_structure': self.curr_structure.as_dict(),
      'curr_mismatch': self.curr_mismatch,
      'accepted': self.accepted.tolist(),
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
