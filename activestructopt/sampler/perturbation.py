from activestructopt.sampler.base import BaseSampler
from activestructopt.common.registry import registry
from activestructopt.common.constraints import lj_reject
from pymatgen.core.structure import IStructure, Lattice
import numpy as np

@registry.register_sampler("Perturbation")
class Perturbation(BaseSampler):
  def __init__(self, initial_structure: IStructure, perturbrmin = 0.1, 
    perturbrmax = 1.0, perturblσ = 0.0, perturbl = None, perturbθ = None, 
    constraint_buffer = 0.85) -> None:
    self.initial_structure = initial_structure
    self.perturbrmin = perturbrmin
    self.perturbrmax = perturbrmax
    self.perturblσ = perturblσ
    self.perturbl = perturbl
    self.perturbθ = perturbθ
    self.constraint_buffer = constraint_buffer

  def sample(self) -> IStructure:
    rejected = True
    while rejected:
      try:
        new_structure = self.initial_structure.copy()
        new_structure.perturb(np.random.uniform(
          self.perturbrmin, self.perturbrmax))

        if self.perturbl is None:
          new_structure.lattice = Lattice(new_structure.lattice.matrix + 
            self.perturblσ * np.random.normal(0, 1, (3, 3)))
        else:
          new_structure.lattice = new_structure.lattice.from_parameters(
            max(0.0, new_structure.lattice.a * np.random.uniform(
              1 - self.perturbl, 1 + self.perturbl)),
            max(0.0, new_structure.lattice.b * np.random.uniform(
              1 - self.perturbl, 1 + self.perturbl)),
            max(0.0, new_structure.lattice.c * np.random.uniform(
              1 - self.perturbl, 1 + self.perturbl)), 
            new_structure.lattice.alpha, 
            new_structure.lattice.beta, 
            new_structure.lattice.gamma
          ) if np.random.rand() < 0.5 else new_structure.lattice.from_parameters(
            new_structure.lattice.a,
            new_structure.lattice.b,
            new_structure.lattice.c, 
            min(180.0, max(0.0, new_structure.lattice.alpha + np.random.uniform(
              -self.perturbθ, self.perturbθ))), 
            min(180.0, max(0.0, new_structure.lattice.beta + np.random.uniform(
              -self.perturbθ, self.perturbθ))), 
            min(180.0, max(0.0, new_structure.lattice.gamma + np.random.uniform(
              -self.perturbθ, self.perturbθ))),
          )
        
        rejected = lj_reject(new_structure, buffer = self.constraint_buffer)
      except:
        rejected = True
    return new_structure
