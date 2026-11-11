from activestructopt.sampler.base import BaseSampler
from activestructopt.constraint.base import BaseConstraint
from activestructopt.common.registry import registry
from pymatgen.core.structure import IStructure, Structure
import numpy as np
import json

@registry.register_sampler("FromFile")
class FromFile(BaseSampler):
  def __init__(self, initial_structure: IStructure, 
    constraints: list[BaseConstraint], filename = None, 
    **kwargs) -> None:
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
