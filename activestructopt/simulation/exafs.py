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

@registry.register_simulation("EXAFS")
class EXAFS(BaseSimulation):
  def __init__(self, initial_structure, feff_location = "", folder = "", 
    absorber = 'Co', edge = 'K', radius = 10.0, 
    additional_settings = {'EXAFS': 12.0, 'SCF': '4.5 0 30 .2 1'},
    sbatch_template = None, sbatch_group_template = None,
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

    with open(self.sbatch_group_template if group else self.sbatch_template, 
      'r') as file:
      sbatch_data = file.read()
    index_str = str(absorber_indices[0])
    for i in range(1, len(absorber_indices)):
      index_str += separator + str(absorber_indices[i])
    sbatch_data = sbatch_data.replace('##ARRAY_INDS##', index_str)
    sbatch_data = sbatch_data.replace('##DIRECTORY##', new_folder)
    new_job_file = os.path.join(new_folder, 'job.sbatch')
    with open(new_job_file, 'w') as file:
      file.write(sbatch_data)
    slurm_response = subprocess.check_output(["sbatch", f"{new_job_file}"])
    self.slurm_job_number = int(str(slurm_response).split('\\n')[0].split()[-1])
    
    self.folder = new_folder
    self.params = params
    self.inds = absorber_indices 

  def resolve(self):
    finished = False
    while not finished:
      time.sleep(30)
      try:
        sacct_check_output = subprocess.check_output(
          ["sacct", f"--jobs={self.slurm_job_number}", "--format=state"])
        job_status = str(sacct_check_output).split('\\n')[2]
        if 'FAILED' in job_status or 'COMPLETED' in job_status or (
          'CANCELLED' in job_status):
          finished = True
      except:
        print('Probably a temporary slurm issue')
        print(traceback.format_exc())

    chi_ks = np.zeros((self.N, 181))
    for absorb_ind in self.inds:
      new_abs_folder = os.path.join(self.folder, str(absorb_ind))
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
