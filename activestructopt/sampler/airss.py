from activestructopt.sampler.base import BaseSampler
from activestructopt.constraint.base import BaseConstraint
from activestructopt.constraint.lennard_jones import LennardJonesRepulsion
from activestructopt.constraint.mlip_burn_in import MLIPBurnIn
from activestructopt.common.registry import registry
from pymatgen.io.res import AirssProvider
from pymatgen.io.ase import AseAtomsAdaptor
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
    perturb_lattice = True, relax_fmax = 0.1, relax_steps = 500,
    relax_cell = True, volume_range = 0.2) -> None:
    if not perturb_lattice:
      raise NotImplementedError
    self.airss_bin = airss_bin
    self.constraints = constraints
    self.nsampled = 0
    self.sampling_dir = sampling_dir

    # Random structures from buildcell are far from any minimum, so they almost
    # never pass the MLIP burn-in. As in AIRSS, they are relaxed first, with
    # the burn-in's own MLIP, whenever that constraint is used.
    self.relax_calculator = None
    for constraint in constraints:
      if isinstance(constraint, MLIPBurnIn):
        self.relax_calculator = constraint.calculator
        break
    self.relax_fmax = relax_fmax
    self.relax_steps = relax_steps
    self.relax_cell = relax_cell

    build_cell_data = '%BLOCK POSITIONS_FRAC\n'
    comp_dict = initial_structure.composition.as_dict()
    for key, value in comp_dict.items():
      count = round(value)
      if not np.isclose(value, count):
        raise ValueError(f"Non-integer amount for {key}: {value}")
      build_cell_data += f'{key}  0.0  0.0  0.0  # {key} % NUM={int(count)}\n'
    build_cell_data += '%ENDBLOCK POSITIONS_FRAC\n\n'

    # Target the starting structure's density, allowing +/- volume_range of it.
    # buildcell's volume is per row of the POSITIONS block, before the rows are
    # expanded by NUM, so the per-atom volume is multiplied by the number of
    # rows (one per species here). See
    # https://zhubonan.com/blog/2026/setting-volume-in-airss-buildcell/
    targvol = initial_structure.volume / len(initial_structure) * len(
      comp_dict)
    if volume_range > 0:
      build_cell_data += (
        f'#TARGVOL={targvol * (1 - volume_range):.4f}' +
        f'-{targvol * (1 + volume_range):.4f}\n')
    else:
      build_cell_data += f'#TARGVOL={targvol:.4f}\n'

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
      os.remove(cell_path)
      os.remove(res_path)

      if self.relax_calculator is not None:
        new_structure = self.relax(new_structure)
        if new_structure is None:
          continue

      rejected = False
      for constraint in self.constraints:
        rejected = rejected or constraint.hard_constraint(new_structure)

    return new_structure

  def relax(self, structure: IStructure):
    """Relaxes the positions (and the cell, if relax_cell) with FIRE. Returns
    None if the relaxation produced a non-finite position or cell."""
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    atoms = AseAtomsAdaptor().get_atoms(structure)
    atoms.calc = self.relax_calculator
    target = FrechetCellFilter(atoms) if self.relax_cell else atoms
    FIRE(target, logfile = None).run(fmax = self.relax_fmax,
      steps = self.relax_steps)
    if not (np.all(np.isfinite(atoms.positions)) and
      np.all(np.isfinite(atoms.cell.array))):
      return None
    atoms.calc = None
    return AseAtomsAdaptor().get_structure(atoms)
