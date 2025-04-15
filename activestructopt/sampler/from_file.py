from activestructopt.sampler.base import BaseSampler
from activestructopt.common.registry import registry
from activestructopt.common.constraints import lj_reject, lj_rmins
from pymatgen.core.structure import IStructure, Structure
import numpy as np
import json

@registry.register_sampler("FromFile")
class FromFile(BaseSampler):
  def __init__(self, initial_structure: IStructure, seed = 0, 
    perturb_lattice = True, constraint_buffer = 0.85, filename = None) -> None:
    assert filename is not None
    data = json.load(open(filename))['structures']
    self.structures = [Structure.from_dict(d) for d in data]
    self.curr_index = 0
    self.max_index = len(self.structures)

  def sample(self) -> IStructure:
    new_structure = self.structures[self.curr_index].copy()
    self.curr_index = self.curr_index + 1
    if self.curr_index == self.max_index:
      print("WARNING: Recycling random structures from file")
      self.curr_index = 0
    return new_structure
