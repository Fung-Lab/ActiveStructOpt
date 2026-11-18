from activestructopt.sampler.base import BaseSampler
from activestructopt.constraint.base import BaseConstraint
from activestructopt.constraint.lennard_jones import LennardJonesRepulsion
from activestructopt.common.registry import registry
from pymatgen.io.res import AirssProvider

@registry.register_sampler("AIRSS")
class AIRSS(BaseSampler):
  def __init__(self, initial_structure: IStructure, 
    constraints: list[BaseConstraint], airss_bin = None,
    perturb_lattice = True) -> None:
    if not perturb_lattice:
      raise NotImplementedError
    self.airss_bin = airss_bin
    self.constraints = constraints
    self.nsampled = 0

    build_cell_data = '%BLOCK POSITIONS_FRAC\n'
    comp_dict = initial_structure.composition.as_dict()
    for key, value in comp_dict:
      build_cell_data += f'{key}  0.0  0.0  0.0  # {key} % NUM={int(value)}\n'
    build_cell_data += '%ENDBLOCK POSITIONS_FRAC\n'

    for constraint in constraints:
      
    comp_dict

    #MINSEP=1.0 Ti-Ti=2.0 Ti-O=1.5 O-O=1.2

  def sample(self) -> IStructure:
    if self.build_cell_file
    

    return new_structure
