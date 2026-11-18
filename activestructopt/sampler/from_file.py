from activestructopt.sampler.base import BaseSampler
from activestructopt.constraint.base import BaseConstraint
from activestructopt.common.registry import registry
from pymatgen.core.structure import IStructure, Structure
import numpy as np
import json

def save_samples_to_file(config, initial_structure, N, filename):
  sampler_constraints = []
  for constraint_config in config['aso_params']['sampler_constraints']:
    constraint_cls = registry.get_constraint_class(constraint_config['name'])
    sampler_constraints.append(constraint_cls(**(constraint_config['args']))
  
  sampler_cls = registry.get_sampler_class(
    config['aso_params']['sampler']['name'])
  sampler = sampler_cls(initial_structure, sampler_constraints,
    **(config['aso_params']['sampler']['args']))

  structures = []
  for _ in range(N):
    structures.append(sampler.sample())

  out_dict = {'structures': [s.as_dict() for s in structures]}
  with open(filename, "w") as file: 
    json.dump(out_dict, file)
  

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
