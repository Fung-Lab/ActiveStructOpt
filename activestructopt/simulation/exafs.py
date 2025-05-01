from activestructopt.simulation.base import BaseSimulation, ASOSimulationException
from activestructopt.common.registry import registry
from pymatgen.io import feff
from pymatgen.io.feff.sets import MPEXAFSSet
from pymatgen.io.feff.outputs import Xmu
import numpy as np
import os
import time
import subprocess
import shutil
import traceback
import stat

@registry.register_simulation("EXAFS")
class EXAFS(BaseSimulation):
  def __init__(self, initial_structure, feff_location = "", folder = "", 
    absorber = 'Co', edge = 'K', radius = 10.0, 
    additional_settings = {'EXAFS': 12.0, 'SCF': '4.5 0 30 .2 1'},
    sh_template = None, 
    sbatch_template = None, sbatch_group_template = None,
    number_absorbers = None, save_sim = True,
    **kwargs) -> None:
    self.feff_location = feff_location
    self.parent_folder = folder
    self.absorber = absorber
    self.edge = edge
    self.radius = radius
    self.additional_settings = additional_settings
    self.mask = [x.symbol == self.absorber 
      for x in initial_structure.species]
    self.N = len(self.mask)
    self.sbatch_template = sbatch_template
    self.sbatch_group_template = sbatch_group_template
    self.sh_template = sh_template
    self.number_absorbers = number_absorbers
    self.save_sim = save_sim

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

  def get(self, struct, group = False, separator = ','):
    structure = struct.copy()

    # get all indices of the absorber
    absorber_indices = 8 * np.argwhere(
      [x.symbol == self.absorber for x in structure.species]).flatten()

    if self.number_absorbers is not None:
      absorber_indices = np.sort(np.random.choice(absorber_indices, 
        self.number_absorbers, replace = False))

      self.mask = [((8 * x) in absorber_indices) for x in range(len(structure))]

    assert len(absorber_indices) > 0

    # guarantees at least two atoms of the absorber,
    # which is necessary because two different ipots are created
    structure.make_supercell(2)

    subfolders = [int(x) for x in os.listdir(self.parent_folder)]
    new_folder = os.path.join(self.parent_folder, str(np.max(
      subfolders) + 1 if len(subfolders) > 0 else 0))
    os.mkdir(new_folder)
    
    for i in range(len(absorber_indices)):
      new_abs_folder = os.path.join(new_folder, str(i))
      os.mkdir(new_abs_folder)

      params = MPEXAFSSet(
        int(absorber_indices[i]),
        structure,
        edge = self.edge,
        radius = self.radius,
        user_tag_settings = self.additional_settings)

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

    if (self.sbatch_template is not None) or (
      self.sbatch_group_template is not None):
      with open(self.sbatch_group_template if group else self.sbatch_template, 
        'r') as file:
        sbatch_data = file.read()
      index_str = str(0)
      for i in range(1, len(absorber_indices)):
        index_str += separator + str(i)
      job_name = int(time.time()) % 604800
      sbatch_data = sbatch_data.replace('##ARRAY_INDS##', index_str)
      sbatch_data = sbatch_data.replace('##DIRECTORY##', new_folder)
      sbatch_data = sbatch_data.replace('##JOB_NAME##', str(job_name))
      new_job_file = os.path.join(new_folder, 'job.sbatch')
      with open(new_job_file, 'w') as file:
        file.write(sbatch_data)
      
      try:
        subprocess.check_output(["sbatch", f"{new_job_file}"])
      except subprocess.CalledProcessError as e:
        print(e.output)

    elif self.sh_template is not None:
      with open(self.sh_template, 'r') as file:
        sh_data = file.read()
      index_str = str(0)
      for i in range(1, len(absorber_indices)):
        index_str += separator + str(i)
      sh_data = sh_data.replace('##ARRAY_INDS##', index_str)
      sh_data = sh_data.replace('##DIRECTORY##', new_folder)
      sh_data = sh_data.replace('##FEFF_DIR##', self.feff_location)
      new_job_file = os.path.join(new_folder, 'feff_job.sh')
      with open(new_job_file, 'w') as file:
        file.write(sh_data)
      time.sleep(10) # Wait for file to write
      file_perms = os.stat(new_job_file).st_mode
      os.chmod(new_job_file, file_perms | stat.S_IXUSR)
      try:
        subprocess.check_output(["nohup", 'feff_job.sh'], cwd = new_folder)
      except subprocess.CalledProcessError as e:
        print(e.output)

    self.folder = new_folder
    self.params = params
    self.inds = absorber_indices 

  def check_done(self):
    for i in range(len(self.inds)):
      new_abs_folder = os.path.join(self.folder, str(i))
      if not os.path.isfile(os.path.join(new_abs_folder, "DONE")):
        return False
    return True

  def resolve(self):
    finished = False
    while not finished:
      finished = self.check_done()
      time.sleep(30)

    chi_ks = np.zeros((self.N, 181))
    for i in range(len(self.inds)):
      absorb_ind = self.inds[i]
      new_abs_folder = os.path.join(self.folder, str(i))
      xmu_file = os.path.join(new_abs_folder, "xmu.dat")
      try:
        f = open(xmu_file, "r")
        start = 0
        i = 0
        while start == 0:
          i += 1
          if f.readline().startswith("#  omega"):
            start = i
        f.close()
      except:
        raise ASOSimulationException(f"Could not open {xmu_file}")

      try:
        xmu = Xmu(self.params.header, feff.inputs.Tags(self.params.tags), 
          int(absorb_ind), np.genfromtxt(xmu_file, skip_header = start))
      except:
        raise ASOSimulationException(f"Could not parse {xmu_file}")
      
      chi_ks[int(np.round(absorb_ind / 8))] = xmu.chi[60:]

      if not self.save_sim:
        shutil.rmtree(new_abs_folder)
    
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
      if os.path.isdir(self.folder):
        shutil.rmtree(self.folder)

  def get_mismatch(self, to_compare, target):
    return np.mean((
      np.mean(to_compare[np.array(self.mask)], axis = 0) - target) ** 2) 
