from activestructopt.sampler.base import BaseSampler
from activestructopt.constraint.base import BaseConstraint
from activestructopt.common.registry import registry, setup_imports
from pymatgen.core.structure import IStructure, Structure
import numpy as np
import json
import time

def _count_hard_constraint_calls(constraint, counts, on_check = None):
  """Wraps constraint.hard_constraint on the instance, so the constraint keeps
  its class, to count how many structures it checks and rejects. on_check, if
  given, is called after each check."""
  hard_constraint = constraint.hard_constraint

  def counted_hard_constraint(structure, **kwargs):
    counts['checked'] += 1
    rejected = hard_constraint(structure, **kwargs)
    if rejected:
      counts['rejected'] += 1
    if on_check is not None:
      on_check()
    return rejected

  constraint.hard_constraint = counted_hard_constraint

def _print_progress(n_accepted, N, constraint_names, constraint_counts,
  start_time):
  message = f'{n_accepted}/{N} accepted ({time.time() - start_time:.0f} s)'
  if len(constraint_counts) > 0:
    # Constraints short circuit, so the first one sees every attempt
    attempts = constraint_counts[0]['checked']
    rejected = attempts - n_accepted
    message += (f', {attempts} attempted, {rejected} rejected ' +
      f'({rejected / max(attempts, 1):.1%})')
    message += ''.join(f'\n  {name}: rejected {counts["rejected"]} of ' +
      f'{counts["checked"]} checked' for name, counts in zip(
      constraint_names, constraint_counts))
  print(message, flush = True)

def save_samples_to_file(config, initial_structure, N, filename):
  setup_imports()

  sampler_constraints = []
  constraint_names = []
  constraint_counts = []
  structures = []
  start_time = time.time()
  # Progress is also printed while structures are being rejected, at most once
  # per progress_interval seconds, so a run that accepts nothing shows why
  progress_interval = 60
  last_print = [start_time]

  def maybe_print_progress():
    if time.time() - last_print[0] >= progress_interval:
      last_print[0] = time.time()
      _print_progress(len(structures), N, constraint_names, constraint_counts,
        start_time)

  for constraint_config in config['aso_params']['sampler_constraints']:
    constraint_cls = registry.get_constraint_class(constraint_config['name'])
    constraint = constraint_cls(**(constraint_config['args']))
    counts = {'checked': 0, 'rejected': 0}
    _count_hard_constraint_calls(constraint, counts, maybe_print_progress)
    sampler_constraints.append(constraint)
    constraint_names.append(constraint_config['name'])
    constraint_counts.append(counts)

  sampler_cls = registry.get_sampler_class(
    config['aso_params']['sampler']['name'])
  sampler = sampler_cls(initial_structure, sampler_constraints,
    **(config['aso_params']['sampler']['args']))

  for _ in range(N):
    structures.append(sampler.sample())
    last_print[0] = time.time()
    _print_progress(len(structures), N, constraint_names, constraint_counts,
      start_time)

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
