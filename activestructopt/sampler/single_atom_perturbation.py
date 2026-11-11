from activestructopt.sampler.base import BaseSampler
from activestructopt.constraint.base import BaseConstraint
from activestructopt.common.registry import registry
from pymatgen.core.structure import IStructure
import numpy as np

@registry.register_sampler("SingleAtomPerturbation")
class SingleAtomPerturbation(BaseSampler):
  def __init__(self, initial_structure: IStructure, 
    constraints: list[BaseConstraint], perturbrmin = 0.1, 
    perturbrmax = 1.0, perturblmax = 0.0, perturbθmax = 0.0, 
    lattice_prob = 0.5) -> None:
    self.initial_structure = initial_structure
    self.perturbrmin = perturbrmin
    self.perturbrmax = perturbrmax
    self.perturblmax = perturblmax
    self.perturbθmax = perturbθmax
    self.lattice_prob = lattice_prob
    self.constraints = constraints

  def sample(self) -> IStructure:
    rejected = True
    while rejected:
      new_structure = self.initial_structure.copy()
      new_structure2 = self.initial_structure.copy()
      new_structure2.perturb(np.random.uniform(
        self.perturbrmin, self.perturbrmax))

      if np.random.rand() > self.lattice_prob:
        rand_index = np.random.randint(0, len(new_structure))

        new_structure.sites[rand_index].frac_coords = new_structure2.sites[
          rand_index].frac_coords
      else:
        to_change = np.random.randint(6)
        new_structure.lattice = new_structure.lattice.from_parameters(
          max(0.0, new_structure.lattice.a * np.random.uniform(
            1 - self.perturblmax, 1 + self.perturblmax)) if to_change == 0 else new_structure.lattice.a,
          max(0.0, new_structure.lattice.b * np.random.uniform(
            1 - self.perturblmax, 1 + self.perturblmax)) if to_change == 1 else new_structure.lattice.b,
          max(0.0, new_structure.lattice.c * np.random.uniform(
            1 - self.perturblmax, 1 + self.perturblmax)) if to_change == 2 else new_structure.lattice.c, 
          min(180.0, max(0.0, new_structure.lattice.alpha + np.random.uniform(
            -self.perturbθmax, self.perturbθmax))) if to_change == 3 else new_structure.lattice.alpha, 
          min(180.0, max(0.0, new_structure.lattice.beta + np.random.uniform(
            -self.perturbθmax, self.perturbθmax))) if to_change == 4 else new_structure.lattice.beta, 
          min(180.0, max(0.0, new_structure.lattice.gamma + np.random.uniform(
            -self.perturbθmax, self.perturbθmax))) if to_change == 5 else new_structure.lattice.gamma
        )
      rejected = False
      for constraint in self.constraints:
        rejected = rejected or constraint.hard_constraint(new_structure)
    return new_structure
