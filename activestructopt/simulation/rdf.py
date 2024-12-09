from activestructopt.simulation.base import BaseSimulation
from activestructopt.common.registry import registry
from scipy.stats import norm
import numpy as np
from pymatgen.optimization.neighbors import find_points_in_spheres

@registry.register_simulation("RDF")
class RDF(BaseSimulation):
  def __init__(self, initial_structure, σ = 0.05, dr = 0.01, max_r = 12.0, 
    **kwargs) -> None:
    self.σ = σ
    self.dr = dr
    self.max_r = max_r
    self.rmax = max_r + 3 * σ + dr
    self.rs = np.arange(0.5, self.rmax + dr, dr)
    self.nr = len(self.rs) - 1
    self.mask = [True for _ in initial_structure.species]
    self.outdim = (self.nr - int((3 * σ) / dr) - 1)
    self.natoms = len(initial_structure)
    self.conv_dist = norm.pdf(np.arange(-3 * σ, 3 * σ + dr, dr), 0.0, σ)

  def setup_config(self, config):
    config['dataset']['preprocess_params']['prediction_level'] = 'node'
    config['dataset']['preprocess_params']['output_dim'] = self.outdim
    return config

  def get(self, struct, group = False, separator = None):
    self.normalization = 4 * self.natoms / struct.volume * np.pi * (
      self.rs[:-1]) ** 2
    self.cart_coords = np.array(struct.cart_coords, dtype=float)
    self.lattice_matrix = np.array(struct.lattice.matrix, dtype=float)

  def resolve(self):
    rdf = np.zeros((self.natoms, self.outdim), dtype = float)
    for i in range(self.natoms):
      # based heavily on https://github.com/materialsproject/pymatgen/blob/a850e6972b8addc0ecddfefc6394cbb85588f4e4/pymatgen/core/lattice.py#L1412
      # to faster get the distances from pymatgen
      rdf[i] = np.convolve(np.histogram(find_points_in_spheres(
          all_coords = self.cart_coords,
          center_coords = np.array([self.cart_coords[i]], dtype=float),
          r = self.rmax,
          pbc = np.array([1, 1, 1], dtype=int),
          lattice = self.lattice_matrix,
          tol = 1e-8,
        )[3], self.rs)[0] / self.normalization, self.conv_dist, mode="same"
        )[0:self.outdim]
    
    return rdf

  def garbage_collect(self, is_better):
    return

  def get_mismatch(self, to_compare, target):
    return np.mean((np.mean(to_compare, axis = 0) - target) ** 2)
