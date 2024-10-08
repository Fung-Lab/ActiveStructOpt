from activestructopt.simulation.base import BaseSimulation
from activestructopt.common.registry import registry
from pymatgen.io import feff
from pymatgen.io.feff.sets import MPEXAFSSet
from pymatgen.io.feff.outputs import Xmu
import numpy as np
import os
import time
import subprocess
import shutil

@registry.register_simulation("EXAFS")
class EXAFS(BaseSimulation):
  def __init__(self, initial_structure, feff_location = "", folder = "", 
    absorber = 'Co', edge = 'K', radius = 10.0, kmax = 12.0, 
    scf = '4.5 0 30 .2 1', rgrid = 0.05, **kwargs) -> None:
    self.feff_location = feff_location
    self.parent_folder = folder
    self.absorber = absorber
    self.edge = edge
    self.radius = radius
    self.kmax = kmax
    self.scf = scf
    self.rgrid = rgrid
    self.mask = [x.symbol == self.absorber 
      for x in initial_structure.species]
    self.N = len(self.mask)

  def setup_config(self, config):
    config['dataset']['preprocess_params']['prediction_level'] = 'node'
    config['optim']['loss'] = {
      'loss_type': 'MaskedTorchLossWrapper',
      'loss_args': {
        'loss_fn': 'l1_loss',
        'mask': self.mask,
      }
    }
    config['dataset']['preprocess_params']['output_dim'] = 181
    return config

  def get(self, struct):
    structure = struct.copy()

    # get all indices of the absorber
    absorber_indices = 8 * np.argwhere(
      [x.symbol == self.absorber for x in structure.species]).flatten()

    assert len(absorber_indices) > 0

    # guarantees at least two atoms of the absorber,
    # which is necessary because two different ipots are created
    structure.make_supercell(2)

    subfolders = [int(x) for x in os.listdir(self.parent_folder)]
    new_folder = os.path.join(self.parent_folder, str(np.max(
      subfolders) + 1 if len(subfolders) > 0 else 0))
    os.mkdir(new_folder)
    
    for absorb_ind in absorber_indices:
      new_abs_folder = os.path.join(new_folder, str(absorb_ind))
      os.mkdir(new_abs_folder)

      params = MPEXAFSSet(
        int(absorb_ind),
        structure,
        edge = self.edge,
        radius = self.radius,
        user_tag_settings = {'EXAFS': self.kmax, 'SCF': self.scf, 
          'RGRID': self.rgrid})

      atoms_loc = os.path.join(new_abs_folder, 'ATOMS')
      pot_loc = os.path.join(new_abs_folder, 'POTENTIALS')
      params_loc = os.path.join(new_abs_folder, 'PARAMETERS')

      params.atoms.write_file(atoms_loc)
      params.potential.write_file(pot_loc)
      feff.inputs.Tags(params.tags).write_file(params_loc)
      # https://www.geeksforgeeks.org/python-program-to-merge-two-files-into-a-third-file/
      atoms = pot = tags = ""
      with open(atoms_loc) as fp:
        atoms = fp.read()
      with open(pot_loc) as fp:
        pot = fp.read()
      with open(params_loc) as fp:
        tags = fp.read()
      with open (os.path.join(new_abs_folder, 'feff.inp'), 'w') as fp:
        fp.write(tags + '\n' + pot + '\n' + atoms)
      os.remove(atoms_loc)
      os.remove(pot_loc)
      os.remove(params_loc)

      # run feff.inp and don't wait for the output
      subprocess.Popen(f"cd {new_abs_folder} && {self.feff_location} feff.inp", 
        shell = True)#), stdout = subprocess.PIPE, stderr=subprocess.STDOUT)
    self.folder = new_folder
    self.params = params
    self.inds = absorber_indices 

  def resolve(self):
    chi_ks = np.zeros((self.N, 181))
    for absorb_ind in self.inds:
      new_abs_folder = os.path.join(self.folder, str(absorb_ind))
      opened = False
      while not opened:
        try:
          f = open(os.path.join(new_abs_folder, "xmu.dat"), "r")
          opened = True
          f.close()
        except:
          time.sleep(10)
      time.sleep(1)
      f = open(os.path.join(new_abs_folder, "xmu.dat"), "r")
      start = 0
      i = 0
      while start == 0:
        i += 1
        if f.readline().startswith("#  omega"):
          start = i
      f.close()

      xmu = Xmu(self.params.header, feff.inputs.Tags(self.params.tags), 
        int(absorb_ind), np.genfromtxt(os.path.join(
        new_abs_folder, "xmu.dat"), skip_header = start))
      
      chi_ks[int(np.round(absorb_ind / 8))] = xmu.chi[60:]
    return chi_ks #np.mean(np.array(chi_ks), axis = 0)

  def garbage_collect(self, is_better):
    parent_folder = os.path.dirname(self.folder)
    if is_better:
      subfolders = [int(x) for x in os.listdir(parent_folder)]
      for sf in subfolders:
        to_delete = os.path.join(parent_folder, str(sf))
        if to_delete != self.folder:
          shutil.rmtree(to_delete)
    else:
      shutil.rmtree(self.folder)

  def get_mismatch(self, to_compare, target):
    return np.mean((
      np.mean(to_compare[np.array(self.mask)], axis = 0) - target) ** 2) 
