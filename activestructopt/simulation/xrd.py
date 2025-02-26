from activestructopt.simulation.base import BaseSimulation
from activestructopt.common.registry import registry
import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator

@registry.register_simulation("XRD")
class XRD(BaseSimulation):
  def __init__(self, initial_structure, σ = 0.2, δθ = 0.1, maxθ = 90., 
  **kwargs) -> None:
    self.σ = σ
    self.δθ = δθ
    self.maxθ = maxθ
    self.x_2θs = np.arange(0, maxθ + δθ / 2, δθ)
    self.outdim = len(self.x_2θs)
    self.natoms = len(initial_structure)

  def setup_config(self, config):
    config['dataset']['preprocess_params']['prediction_level'] = 'node'
    config['dataset']['preprocess_params']['output_dim'] = self.outdim
    return config

  def get(self, struct, group = False, separator = None):
    self.pattern = XRDCalculator().get_pattern(struct)

  def resolve(self):
    xps = np.repeat(self.x_2θs, len(self.pattern.x)).reshape(len(self.x_2θs), len(self.pattern.x))
    yps = np.repeat((self.pattern.x), len(self.x_2θs)).reshape(len(self.x_2θs), len(self.pattern.x), order = 'F')
    norm_pattern = np.sum(np.exp(-np.square((xps - yps) / (np.sqrt(2) * self.σ))) * self.pattern.y, axis = 1)
    norm_pattern /= np.max(norm_pattern)
    # Note: no per-atom information
    xrd = np.repeat(norm_pattern[np.new_axis, :], self.natoms)

    return xrd

  def garbage_collect(self, is_better):
    return

  def get_mismatch(self, to_compare, target):
    return np.mean((np.mean(to_compare, axis = 0) - target) ** 2)
