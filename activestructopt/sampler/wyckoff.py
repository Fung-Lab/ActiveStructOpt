from activestructopt.sampler.base import BaseSampler
from activestructopt.common.registry import registry
from activestructopt.common.constraints import lj_reject, lj_rmins
from pymatgen.core.structure import IStructure, Structure
from multiprocessing import Process, Manager
from collections import Counter
import numpy as np
import pyxtal

@registry.register_sampler("Wyckoff")
class Wyckoff(BaseSampler):
  def __init__(self, initial_structure: IStructure, seed = 0, 
    perturb_lattice = True, constraint_buffer = 0.85,
    max_retries = 3, max_time = 20, use_random_state = True, sgs = None) -> None:

    # Distribution from Materials Project
    sg_dist = [146, 2720, 14, 407, 178, 7, 145, 97, 351, 39, 768, 1688, 286, 
      5736, 2345, 1, 5, 66, 596, 89, 13, 2, 11, 0, 10, 64, 3, 9, 196, 11, 
      185, 19, 471, 26, 8, 309, 6, 91, 15, 70, 47, 18, 164, 64, 14, 60, 30, 
      0, 1, 3, 111, 69, 27, 33, 459, 76, 248, 265, 245, 327, 634, 4129, 1480, 
      342, 237, 34, 21, 23, 59, 220, 449, 205, 36, 206, 5, 29, 3, 10, 12, 5, 
      12, 157, 12, 31, 64, 95, 221, 204, 0, 7, 8, 107, 0, 1, 5, 41, 5, 5, 17, 
      28, 0, 13, 1, 1, 5, 2, 105, 14, 45, 20, 14, 5, 94, 46, 15, 10, 9, 18, 
      26, 9, 116, 126, 405, 31, 51, 15, 434, 113, 880, 75, 44, 7, 8, 11, 35, 
      289, 79, 26, 1575, 530, 281, 133, 19, 23, 14, 73, 98, 633, 9, 71, 10, 
      83, 1, 28, 57, 215, 22, 6, 48, 158, 112, 48, 77, 571, 73, 1047, 405, 0, 
      0, 0, 0, 1, 306, 125, 3, 297, 0, 0, 0, 32, 17, 40, 3, 4, 69, 412, 108, 
      20, 644, 51, 528, 8, 418, 1567, 0, 2, 16, 245, 27, 31, 47, 22, 11, 170, 
      234, 68, 0, 2, 0, 0, 0, 23, 44, 17, 58, 509, 118, 71, 18, 176, 1116, 2, 
      210, 34, 1563, 45, 788, 9, 210, 105]

    self.rng = np.random.default_rng(seed)
    self.initial_structure = initial_structure
    element_counter = Counter([site.species.elements[
      0].symbol for site in initial_structure.sites])
    self.zs = list(element_counter.keys())
    self.zcounts = list(element_counter.values())
    self.initial_lattice = pyxtal.lattice.Lattice.from_matrix(
      initial_structure.lattice.matrix)
    self.perturb_lattice = perturb_lattice
    self.max_retries = max_retries
    self.max_time = max_time
    self.constraint_buffer = constraint_buffer
    self.tm = pyxtal.tolerance.Tol_matrix.from_matrix(
      self.constraint_buffer * lj_rmins)
    self.use_random_state = use_random_state

    def get_random_crystal(sg, d):
      xtal = pyxtal.pyxtal()
      xtal.from_random(3, sg, self.zs, self.zcounts, 
        random_state = self.rng if self.use_random_state else None, 
        tm = self.tm,
        lattice = None if self.perturb_lattice else self.initial_lattice)
      d['struct'] = xtal.to_pymatgen().as_dict()

    self.get_random_crystal = get_random_crystal

    if sgs is None:
      self.possible_sgs = []
      for i in range(230):
        tries = 0
        while tries < self.max_retries:
          tries += 1
          # https://stackoverflow.com/questions/14920384/stop-code-after-time-period/14920854
          p = Process(target = self.get_random_crystal, args = (i + 1, {}))
          p.start()
          p.join(self.max_time)
          if p.is_alive(): # if didn't complete in max time
            p.terminate()
            p.join()
          else:
            if p.exitcode == 0:
              self.possible_sgs.append(i + 1)
            break

      self.possible_sgs = np.array(self.possible_sgs)
      self.sg_probs = np.array([sg_dist[i - 1] + 1.0 for i in self.possible_sgs])
      # Adding one allows non-zero probability for sgs not in Materials Project
      self.sg_probs /= np.sum(self.sg_probs)
      print(self.possible_sgs)
    else:
      self.possible_sgs = sgs
      self.possible_sgs = np.array(self.possible_sgs)
      self.sg_probs = np.array([sg_dist[i - 1] + 1.0 for i in self.possible_sgs])
      # Adding one allows non-zero probability for sgs not in Materials Project
      self.sg_probs /= np.sum(self.sg_probs)

  def sample(self) -> IStructure:
    rejected = True
    while rejected:      
      manager = Manager()
      d = manager.dict()

      # https://stackoverflow.com/questions/14920384/stop-code-after-time-period/14920854
      p = Process(target = self.get_random_crystal, args = (
        np.random.choice(self.possible_sgs, p = self.sg_probs), d))
      p.start()
      p.join(self.max_time)
      if p.is_alive(): # if didn't complete in max time
        p.terminate()
        p.join()
      else:
        if p.exitcode == 0:
          new_structure = Structure.from_dict(d['struct'])
          rejected = lj_reject(new_structure)

    new_z_inds = {}
    for s in self.zs:
      new_z_inds[s] = []
    
    new_zs = [site.species.elements[0].symbol for site in new_structure.sites]
    for i in range(len(new_zs)):
      new_z_inds[new_zs[i]].append(i)

    z_inds = []
    z_counts = {}
    for s in self.zs:
      z_counts[s] = 0

    old_zs = [site.species.elements[0
      ].symbol for site in self.initial_structure.sites]
    for i in range(len(old_zs)):
      z_inds.append(new_z_inds[old_zs[i]][z_counts[old_zs[i]]])
      z_counts[old_zs[i]] += 1
    
    new_sites = [new_structure.sites[i] for i in z_inds]
    new_structure.sites = new_sites

    return new_structure
