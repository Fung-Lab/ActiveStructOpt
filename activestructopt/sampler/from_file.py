from activestructopt.sampler.base import BaseSampler
from activestructopt.constraint.base import BaseConstraint
from activestructopt.common.registry import registry, setup_imports
from pymatgen.core.structure import IStructure, Structure
import numpy as np
import json

def save_samples_to_file(config, initial_structure, N, filename):
  setup_imports()
  
  sampler_constraints = []
  for constraint_config in config['aso_params']['sampler_constraints']:
    constraint_cls = registry.get_constraint_class(constraint_config['name'])
    sampler_constraints.append(constraint_cls(**(constraint_config['args'])))
  
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
  

def reorder_like(structure: IStructure, reference: IStructure) -> Structure:
  """Returns structure with its sites reordered so that each site has the same
  element as the corresponding site in reference. Sites of the same element
  keep their relative order."""
  symbols = [site.specie.symbol for site in structure.sites]
  reference_symbols = [site.specie.symbol for site in reference.sites]
  if sorted(symbols) != sorted(reference_symbols):
    raise ValueError(f'Structure composition {structure.composition} does ' +
      f'not match the initial structure composition {reference.composition}')

  remaining = {}
  for i, symbol in enumerate(symbols):
    remaining.setdefault(symbol, []).append(i)
  order = [remaining[symbol].pop(0) for symbol in reference_symbols]
  return Structure.from_sites([structure.sites[i] for i in order])

@registry.register_sampler("FromFile")
class FromFile(BaseSampler):
  def __init__(self, initial_structure: IStructure,
    constraints: list[BaseConstraint], filename = None,
    **kwargs) -> None:
    assert filename is not None
    with open(filename) as file:
      data = json.load(file)['structures']
    self.structures = [reorder_like(Structure.from_dict(d), initial_structure
      ) for d in data]
    self.curr_index = 0
    self.max_index = len(self.structures)

  def sample(self) -> IStructure:
    new_structure = self.structures[self.curr_index].copy()
    self.curr_index = self.curr_index + 1
    if self.curr_index == self.max_index:
      print("WARNING: Recycling random structures from file")
      self.curr_index = 0
    return new_structure
