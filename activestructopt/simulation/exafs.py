from activestructopt.simulation.base import BaseSimulation, ASOSimulationException
from activestructopt.common.registry import registry
from mattertune.backbones import ORBBackboneModule
from mattertune.backbones.orb import ORBBackboneConfig
from pymatgen.io import feff
from pymatgen.io.feff.sets import MPEXAFSSet
from pymatgen.io.feff.outputs import Xmu
from larch.xafs import feffpath, sigma2_debye
from larch.xafs import autobk
from larch import Group
import numpy as np
import scipy.constants as consts
import scipy as sp
from lmfit import Minimizer, Parameters
import os
import time
import subprocess
import shutil
import traceback
import stat

def get_sims(folder, n):
  paths = []
  for i in range(n):
    path_files = np.sort(list(filter(lambda x: x.startswith('feff'
      ) and x.endswith('.dat'), 
      os.listdir(f"{folder}/{i}"))))
    abs_paths = []
    for j in range(len(path_files)):
      try:
          abs_paths.append(feffpath(f'{folder}/{i}/{path_files[j]}'))
      except:
          print(f'Skipping {path_files[j]}')
    if len(abs_paths) < 1:
      raise ASOSimulationException(f"Folder {folder} has no feffXXXX.dat files")
    paths.append(abs_paths)
  return paths

def get_s02(folder, n):
  s02s = []
  for i in range(n):
    f = open(f'{folder}/{i}/xmu.dat')
    lines = f.readlines()
    s02_line = np.where(['S02=' in line for line in lines])[0][0]
    s02s.append(float(lines[s02_line].split('S02=')[1].split()[0]))
  return s02s

def get_debye_predictor(ckpt_path):
  TD_mean = 2.460525700259601
  TD_std = 0.2499737087979493
  unnormalize_TD = lambda x: 10 ** (2 * TD_std * x + TD_mean)

  ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
  hparams = ckpt["hyper_parameters"].copy()
  hparams.pop("pruning_message_passing", None)
  hparams.pop("using_partition", None)
  model = ORBBackboneModule(ORBBackboneConfig.model_validate(hparams))
  model.load_state_dict(ckpt["state_dict"])
  model = model.eval().to("cpu")
  def predict_debye_temp(s):
    return unnormalize_TD(model.model_forward(model.atoms_to_data(
    Atoms(
        numbers=np.array(s.atomic_numbers),
        positions=np.array(s.cart_coords),
        cell=np.array(s.lattice.matrix),
        pbc=True,
    ), has_labels = False).to('cpu'), 
    mode = 'predict')['predicted_properties']['debye_temp'].item())
  return predict_debye_temp

def get_paths_info(sim):
  struct_info = []

  debye_t_x = np.arange(100, 2000, 10)

  for j, abs_paths in enumerate(sim):
    abs_reffs = []
    abs_m_a = []
    abs_m_s = []
    abs_rnorman = []
    abs_scatter_Zs = []
    abs_degens = []
    abs_amp = []
    abs_pha = []
    abs_rep = []
    abs_lam = []
    abs_sigma2_debye = []

    abs_reffs_ms = []
    abs_nlegs_ms = []
    abs_degen_ms = []
    abs_rnorman_ms = []
    abs_amp_ms = []
    abs_pha_ms = []
    abs_rep_ms = []
    abs_lam_ms = []
    abs_atwt_ms = []
    abs_pos_ms = []
    abs_sigma2_debye_ms = []

    for pathi, path in enumerate(abs_paths):
      if len(path._feffdat.geom) == 2:
        reff = np.linalg.norm(np.array(path._feffdat.geom[1][4:7]).astype(float))
        m_a = path._feffdat.geom[0][3]
        m_s = path._feffdat.geom[1][3]
        rnorman = path._feffdat.rnorman
        if len(abs_reffs) > 0 and np.min(np.abs(np.array(abs_reffs) - reff)
              ) < 0.00005 and abs_scatter_Zs[np.argmin(np.abs(np.array(
              abs_reffs) - reff))] == path._feffdat.geom[1][1]:
          abs_degens[np.argmin(np.abs(np.array(
              abs_reffs) - reff))] += path._feffdat.degen
        else:
          abs_reffs.append(reff)
          abs_m_a.append(m_a)
          abs_m_s.append(m_s)
          abs_rnorman.append(rnorman)
          abs_scatter_Zs.append(path._feffdat.geom[1][1])
          abs_degens.append(path._feffdat.degen)
          abs_amp.append(path._feffdat.amp)
          abs_pha.append(path._feffdat.pha)
          abs_rep.append(path._feffdat.rep)
          abs_lam.append(path._feffdat.lam)
          abs_sigma2_debye.append([sigma2_debye(300., debye_t, path) for debye_t in debye_t_x])
      else:
        for leg in path._feffdat.geom:
          abs_atwt_ms.append(leg[3])
          abs_pos_ms.append([float(leg[4]), float(leg[5]), float(leg[6])])
        rnorman = path._feffdat.rnorman
        abs_rnorman_ms.append(rnorman)
        abs_reffs_ms.append(path._feffdat.reff)
        abs_nlegs_ms.append(len(path._feffdat.geom))
        abs_degen_ms.append(path._feffdat.degen)
        abs_amp_ms.append(path._feffdat.amp)
        abs_pha_ms.append(path._feffdat.pha)
        abs_rep_ms.append(path._feffdat.rep)
        abs_lam_ms.append(path._feffdat.lam)
        abs_sigma2_debye_ms.append([sigma2_debye(300., debye_t, path) for debye_t in debye_t_x])
    abs_info = {
      "k_feff": path._feffdat.k,
      "Reffs": abs_reffs,
      "m_a": m_a,
      "m_s": m_s,
      "rnorman": rnorman,
      "degen": abs_degens,
      "scatterer_Zs": abs_scatter_Zs,
      "amp": abs_amp,
      "pha": abs_pha,
      "rep": abs_rep,
      "lam": abs_lam,
      "sigma2_debye": abs_sigma2_debye,
      "Reffs_MS": abs_reffs_ms,
      "nlegs_MS": abs_nlegs_ms,
      "degen_MS": abs_degen_ms,
      "rnorman_MS": abs_rnorman_ms,
      "amp_MS": abs_amp_ms,
      "pha_MS": abs_pha_ms,
      "rep_MS": abs_rep_ms,
      "lam_MS": abs_lam_ms,
      "atwt_MS": abs_atwt_ms,
      "pos_MS": abs_pos_ms,
      "sigma2_debye_MS": abs_sigma2_debye_ms,
    }
    struct_info.append(abs_info)
  return struct_info

def batch_interp_rows(x, xp, ys):
    x = np.clip(x, xp[0], xp[-1])

    right = np.searchsorted(xp, x, side="right")
    right = np.clip(right, 1, len(xp) - 1)
    left = right - 1

    weight = (x - xp[left]) / (xp[right] - xp[left])

    return (ys[:, left] * (1.0 - weight)[None, :]
        + ys[:, right] * weight[None, :])

# https://github.com/xraypy/xraylarch/blob/dafed7db999523d366f482f6a260bc983c4defe4/larch/xafs/feffdat.py#L638
KTOE = 1.e20*consts.hbar**2 / (2*consts.m_e * consts.e) # 3.8099819442818976
ETOK = 1.0/KTOE
def _calc_chi(k, feffk, reff, degen, pha, amp, rep, lam, sigma2, s02 = 1.0, e0 = 0.0, ei = 0.0):
  en = k ** 2 - float(e0) * ETOK
  q = np.sign(en) * np.sqrt(np.abs(en))

  n_paths = len(reff)

  feff_tables = np.concatenate((pha, amp, rep, lam), axis = 0)
  interpolated = batch_interp_rows(q, feffk, feff_tables)
  pha, amp, rep, lam = np.split(interpolated, [n_paths, 2 * n_paths, 3 * n_paths], axis = 0)

  reff = reff[:, None]
  degen = degen[:, None]
  sigma2 = sigma2[:, None]

  pp = ((rep + 1j / lam) ** 2 + 1j * float(ei) * ETOK)
  p = np.sqrt(pp)

  cchi = np.exp(-2.0 * reff * p.imag - 2.0 * pp * sigma2
    + 1j * (2.0 * q[None, :] * reff + pha - 4.0 * p * sigma2 / reff))
  cchi *= (degen * float(s02) * amp / (q[None, :] * reff ** 2))
  cchi[:, 0] = 2.0 * cchi[:, 1] - cchi[:, 2]
  return cchi.imag

def get_absorber_spectra_debye(reffs, degens, 
        phas, amps, lams, reps, sigma2s, e0, ei, s02, k, feffk):
  return np.sum(_calc_chi(k, feffk, reffs, degens, 
    phas, amps, reps, lams, sigma2s, s02 = s02, e0 = e0, ei = ei,
    ), axis = 0)

def get_structure_spectra_debye(reffs, degens, 
        phas, amps, lams, reps, sigma2s, e0s, eis, s02s, k, feffk):
  return np.mean(np.stack([get_absorber_spectra_debye(reffs[i], degens[i], 
    phas[i], amps[i], lams[i], reps[i], sigma2s[i], 
    e0s[i], eis[i], s02s[i], k, feffk) for i in range(len(reffs))]), axis = 0)

def get_aligned_sim(sim, s02s, exp_g, structure, kmin_fit = 4.0, kmax_fit = 15.0, 
  kwfit = 3, vary_TD = True, TD_predictor_path = None, kwout = 3):
  kmini = np.argmin(np.abs(exp_g.k - kmin_fit))
  kmaxi = np.argmin(np.abs(exp_g.k - kmax_fit))
  ks = exp_g.k[kmini:kmaxi]
  exp_chi = exp_g.chi[kmini:kmaxi]

  n = len(sim)
  paths_info = get_paths_info(sim)

  reffs = []
  degens = []
  phas = []
  amps = []
  lams = []
  reps = []
  sigma2_grids = []
  for i in range(n):
    reffs.append(np.concatenate((paths_info[i]['Reffs'], paths_info[i]['Reffs_MS'])))
    degens.append(np.concatenate((paths_info[i]['degen'], paths_info[i]['degen_MS'])))
    phas.append(np.concatenate((paths_info[i]['pha'], paths_info[i]['pha_MS'])))
    amps.append(np.concatenate((paths_info[i]['amp'], paths_info[i]['amp_MS'])))
    lams.append(np.concatenate((paths_info[i]['lam'], paths_info[i]['lam_MS'])))
    reps.append(np.concatenate((paths_info[i]['rep'], paths_info[i]['rep_MS'])))
    sigma2_grids.append(np.concatenate((paths_info[i]['sigma2_debye'], paths_info[i]['sigma2_debye_MS'])))

  def res_fun_lmfit(params):
    sigma2s = []
    for i in range(n):
      sigma2s.append(batch_interp_rows(np.atleast_1d(params[f'θD']), 
        np.arange(100, 2000, 10), sigma2_grids[i]).flatten())
    dev = (ks ** kwfit *  exp_chi) - (
      ks ** kwfit * get_structure_spectra_debye(reffs, degens, 
        phas, amps, lams, reps, sigma2s, 
        [params[f'ΔE0_{i}'] for i in range(n)],
        [params[f'Ei_{i}'] for i in range(n)], s02s, ks, paths_info[0]['k_feff']))
    return dev

  if TD_predictor_path is None:
    predicted_debye_t = 500.0
    debye_lb = 100.0
    debye_ub = 2000.0
  else:
    debye_predictor = get_debye_predictor(TD_predictor_path)
    predicted_debye_t = debye_predictor(structure)
    debye_lb = max(0, predicted_debye_t - 250)
    debye_ub = predicted_debye_t + 250

  params = Parameters()
  for i in range(len(sim)):
      params.add(f'Ei_{i}', value = 0.0, min = -5.0, max = 5.0, vary = False)
      params.add(f'ΔE0_{i}', value = 0.0, min = -30.0, max = 30.0)
  params.add(f'θD', value = predicted_debye_t, min = debye_lb, 
    max = debye_ub, vary = vary_TD)

  minner = Minimizer(res_fun_lmfit, params)
  result = minner.minimize()
  eis = [result.params[f'Ei_{i}'] for i in range(n)]
  e0s = [result.params[f'ΔE0_{i}'] for i in range(n)]
  sigma2s = []
  for i in range(len(sigma2_grids)):
    sigma2s.append(batch_interp_rows(np.atleast_1d(result.params[f'θD']), 
      np.arange(100, 2000, 10), sigma2_grids[i]).flatten())
  chi_spec = np.stack([get_absorber_spectra_debye(reffs[i], degens[i], 
    phas[i], amps[i], lams[i], reps[i], sigma2s[i], 
    e0s[i], eis[i], s02s[i], ks, paths_info[0]['k_feff']) for i in range(n)])
  return ks ** kwout * chi_spec

@registry.register_simulation("EXAFS")
class EXAFS(BaseSimulation):
  def __init__(self, initial_structure, feff_location = "", folder = "", 
    absorber = 'Co', edge = 'K', radius = 10.0, fit_kmin = 3.0, 
    fit_kmax = 12.0, time_limit = 240,
    additional_settings = {'EXAFS': 12.0, 'SCF': '4.5 0 30 .2 1'},
    sh_template = None, 
    sbatch_template = None, sbatch_group_template = None, sbatch_opt_template = None,
    number_absorbers = None, save_sim = True, TD_predictor_path = None,
    vary_TD = True, kwfit = 3,
    **kwargs) -> None:
    self.exp_g = exp_g
    kmini = np.argmin(np.abs(exp_g.k - fit_kmin))
    kmaxi = np.argmin(np.abs(exp_g.k - fit_kmax))
    self.outdim = int(kmaxi - kmini)
    self.fit_kmin = fit_kmin
    self.fit_kmax = fit_kmax
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
    self.sbatch_opt_template = sbatch_opt_template
    self.sh_template = sh_template
    self.number_absorbers = number_absorbers
    self.save_sim = save_sim
    self.time_limit = time_limit
    self.structure = None
    self.TD_predictor_path = TD_predictor_path
    self.vary_TD = vary_TD
    self.kwfit = kwfit

  def setup_config(self, config):
    config['dataset']['preprocess_params']['prediction_level'] = 'node'
    config['optim']['loss'] = {
      'loss_type': 'MaskedTorchLossWrapper',
      'loss_args': {
        'loss_fn': 'l1_loss',
        'mask': self.mask,
      }
    }
    config['dataset']['preprocess_params']['output_dim'] = self.outdim
    return config

  def get(self, struct, group = False, separator = ','):
    structure = struct.copy()
    self.structure = struct.copy()

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

    if self.TD_predictor_path is not None:
      additional_settings.pop('DEBYE', None)
      debye_predictor = get_debye_predictor(TD_predictor_path)
      predicted_debye_t = debye_predictor(self.structure)
      additional_settings['DEBYE'] = f'300 {predicted_debye_t} 0'
    
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

    self.group = group
    self.folder = new_folder
    self.params = params
    self.inds = absorber_indices 

  def check_done(self):
    if not os.path.isdir(self.folder):
      raise ASOSimulationException(f"Folder {self.folder} was deleted")

    for i in range(len(self.inds)):
      new_abs_folder = os.path.join(self.folder, str(i))
      if not os.path.isfile(os.path.join(new_abs_folder, "DONE")):
        #print(f'waiting for {os.path.join(new_abs_folder, "DONE")}')
        return False
    return True

  def check_opt_done(self):
    if not os.path.isdir(self.folder):
      raise ASOSimulationException(f"Folder {self.folder} was deleted")

    return os.path.isfile(os.path.join(self.folder, "DONE"))

  def resolve(self):
    if not self.group:
      finished = False
      for _ in range(2 * self.time_limit):
        finished = self.check_done()
        if finished:
          break
        time.sleep(30)

      if not finished:
        raise ASOSimulationException(f"Simulation not finished in time limit")

      assert self.sbatch_opt_template is not None, "Need opt template for now"

      with open(self.sbatch_opt_template, 'r') as file:
        sbatch_opt_data = file.read()
      job_name = int(time.time()) % 604800
      sbatch_opt_data = sbatch_opt_data.replace('##DIRECTORY##', self.folder)
      sbatch_opt_data = sbatch_opt_data.replace('##JOB_NAME##', str(job_name))
      new_job_file = os.path.join(self.folder, 'opt_job.sbatch')
      with open(new_job_file, 'w') as file:
        file.write(sbatch_opt_data)

      try:
        subprocess.check_output(["sbatch", f"{new_job_file}"])
      except subprocess.CalledProcessError as e:
        print(e.output)

      print(f'Running optimization for {self.folder}')

      finished = False
      for _ in range(self.time_limit):
        finished = self.check_opt_done()
        if finished:
          break
        time.sleep(30)

      if not finished:
        raise ASOSimulationException(f"Optimization not finished in time limit")

      if not os.path.isfile(os.path.join(self.folder, 'chi_k.dat')):
        raise ASOSimulationException(f"Optimization failed")
    else:
      finished = False
      for _ in range(3 * self.time_limit):
        finished = self.check_done()
        if finished:
          break
        time.sleep(30)

      if not finished:
        raise ASOSimulationException(f"Simulation/Optimization not finished in time limit")

      if not os.path.isfile(os.path.join(self.folder, 'chi_k.dat')):
        raise ASOSimulationException(f"Optimization failed")

    for i, absorb_ind in enumerate(self.inds):
      if not self.save_sim:
        shutil.rmtree(os.path.join(self.folder, str(i)))
    
    chi_ks = np.genfromtxt(os.path.join(self.folder, 'chi_k.dat'))
    return chi_ks 

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
