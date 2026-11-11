from activestructopt.common.registry import registry
from activestructopt.constraints.base import BaseConstraint
from pymatgen.core.structure import IStructure
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np

@registry.register_constraint("MLIPForces")
class MLIPForces(BaseConstraint):
  def __init__(self, mlip = None, force_cutoff = 10.0, **kwargs):
    if mlip == "uma-s-1p2p1":
      try:
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
      except ImportError as e:
        raise ImportError("Error loading fairchem.core") from e
      predictor = pretrained_mlip.get_predict_unit(
        "uma-s-1p2p1",
        device="cuda",
        inference_settings="nitro",
      )
      self.calculator = FAIRChemCalculator(predictor, 
        task_name="omat")

    else:
      raise NotImplementedError("MLIP {mlip} not implemented")
    self.adaptor = AseAtomsAdaptor()
    self.force_cutoff = force_cutoff

  @abstractmethod
  def soft_constraint(self, data, data_type = "pmg", **kwargs):
    if data_type == "pmg":
      atoms = self.adaptor.get_atoms(data)
      atoms.calc = self.calculator
      forces = atoms.get_forces(apply_constraint = False)
      force_mags = np.linalg.norm(forces, axis = 1)
      return np.mean(np.maximum(0, (force_mags - self.force_cutoff)) ** 2)
    else:
      raise NotImplementedError("Data type {data_type} not implemented")

  @abstractmethod
  def hard_constraint(self, structure: IStructure, **kwargs):
    atoms = self.adaptor.get_atoms(structure)
    atoms.calc = self.calculator
    forces = atoms.get_forces(apply_constraint = False)
    return np.max(np.linalg.norm(forces, axis = 1)) > self.force_cutoff
