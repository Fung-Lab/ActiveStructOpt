from activestructopt.sampler.base import BaseSampler
from activestructopt.constraint.base import BaseConstraint
from activestructopt.constraint.lennard_jones import LennardJonesRepulsion
from activestructopt.common.registry import registry
from pymatgen.io.res import AirssProvider
from pymatgen.core.structure import IStructure
from itertools import combinations_with_replacement
import numpy as np
import os
import subprocess

@registry.register_sampler("AIRSS")
class AIRSS(BaseSampler):
  def __init__(self, initial_structure: IStructure, 
    constraints: list[BaseConstraint], sampling_dir = None, 
    airss_bin = None,
    perturb_lattice = True) -> None:
    if not perturb_lattice:
      raise NotImplementedError
    self.airss_bin = airss_bin
    self.constraints = constraints
    self.nsampled = 0
    self.sampling_dir = sampling_dir

    build_cell_data = '%BLOCK POSITIONS_FRAC\n'
    comp_dict = initial_structure.composition.as_dict()
    for key, value in comp_dict.items():
      count = round(value)
      if not np.isclose(value, count):
        raise ValueError(f"Non-integer amount for {key}: {value}")
      build_cell_data += f'{key}  0.0  0.0  0.0  # {key} % NUM={int(count)}\n'
    build_cell_data += '%ENDBLOCK POSITIONS_FRAC\n\n'

    for constraint in constraints:
      if isinstance(constraint, LennardJonesRepulsion):
        build_cell_data += '#MINSEP=0.5'
        for key1, key2 in combinations_with_replacement(comp_dict.keys(), 2):
          Z1 = np.argmax(constraint.el_symbols == key1)
          Z2 = np.argmax(constraint.el_symbols == key2)
          rmin = constraint.lj_rmins[Z1 - 1, Z2 - 1]
          build_cell_data += f' {key1}-{key2}={rmin}'
        build_cell_data += '\n'
    
    self.template_path = os.path.join(self.sampling_dir, 'template.cell')
    with open(self.template_path, 'w') as file:
      file.write(build_cell_data)

  def sample(self) -> IStructure:
    rejected = True
    while rejected:
      self.nsampled += 1
      
      cell_path = os.path.join(self.sampling_dir, f'{self.nsampled}.cell')
      res_path = os.path.join(self.sampling_dir, f'{self.nsampled}.res')
      with open(self.template_path, "rb") as stdin, open(cell_path, "wb") as stdout:
        subprocess.run(
          [os.path.join(self.airss_bin, 'buildcell')],
          stdin = stdin,
          stdout = stdout,
          stderr = subprocess.PIPE,
          check = True,
        )
      with open(cell_path, "rb") as stdin, open(res_path, "wb") as stdout:
        subprocess.run(
          [os.path.join(self.airss_bin, 'cabal'), 'cell', 'res'],
          stdin = stdin,
          stdout = stdout,
          stderr = subprocess.PIPE,
          check = True,
        )
      new_structure = AirssProvider.from_file(res_path).structure
            
      rejected = False
      for constraint in self.constraints:
        rejected = rejected or constraint.hard_constraint(new_structure)

      os.remove(cell_path)
      os.remove(res_path)

    return new_structure
