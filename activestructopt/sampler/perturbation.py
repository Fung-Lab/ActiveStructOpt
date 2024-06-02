from activestructopt.sampler.base import BaseSampler
from activestructopt.common.registry import registry
from activestructopt.common.constraints import lj_reject
from pymatgen.core.structure import IStructure
import numpy as np

@registry.register_sampler("Perturbation")
class Perturbation(BaseSampler):
  def __init__(self, initial_structure: IStructure, perturbrmin = 0.1, 
    perturbrmax = 1.0, perturblmax = 0.0, perturbθmax = 0.0) -> None:
    self.initial_structure = initial_structure
    self.perturbrmin = perturbrmin
    self.perturbrmax = perturbrmax
    self.perturblmax = perturblmax
    self.perturbθmax = perturbθmax

  def sample(self) -> IStructure:
    rejected = True
    while rejected:
      new_structure = self.initial_structure.copy()
      new_structure.perturb(np.random.uniform(
        self.perturbrmin, self.perturbrmax))
      new_structure.lattice = new_structure.lattice.from_parameters(
        max(0.0, new_structure.lattice.a + np.random.uniform(
          -self.perturblmax, self.perturblmax)),
        max(0.0, new_structure.lattice.b + np.random.uniform(
          -self.perturblmax, self.perturblmax)),
        max(0.0, new_structure.lattice.c + np.random.uniform(
          -self.perturblmax, self.perturblmax)), 
        min(180.0, max(0.0, new_structure.lattice.alpha + np.random.uniform(
          -self.perturbθmax, self.perturbθmax))), 
        min(180.0, max(0.0, new_structure.lattice.beta + np.random.uniform(
          -self.perturbθmax, self.perturbθmax))), 
        min(180.0, max(0.0, new_structure.lattice.gamma + np.random.uniform(
          -self.perturbθmax, self.perturbθmax)))
      )
      rejected = lj_reject(new_structure)
    return new_structure