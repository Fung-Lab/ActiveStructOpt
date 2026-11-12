from activestructopt.simulation.base import BaseSimulation, ASOSimulationException
from activestructopt.common.registry import registry
import numpy as np
import time
import os

def run_md(structure, trajectory_path, log_path, timestep = 1.25, 
  temp = 300, friction = 0.002, loginterval = 4, nsteps = 41600,
  burnin_steps = 1600, max_rms_burnin = 0.5)
  try:
    from ase import units
    from ase.build.supercells import make_supercell
    from ase.constraints import FixCom
    from ase.io import Trajectory
    from ase.md import MDLogger
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    from pymatgen.io.ase import AseAtomsAdaptor
  except ImportError as e:
    raise ImportError('Error loading required packages for MD') from e

  atoms = AseAtomsAdaptor().get_atoms(structure)
  minimum_length = 8.4  # Angstrom
  cell_lengths = np.linalg.norm(atoms.cell.array, axis = 1)
  repetitions = np.maximum(1, np.ceil(minimum_length / cell_lengths).astype(int),)
  M = np.diag(repetitions)
  atoms = make_supercell(atoms, M)
  atoms.set_constraint(FixCom())

  predictor = pretrained_mlip.get_predict_unit('uma-s-1p2p1', 
    device = 'cuda', inference_settings = 'nitro')
  atoms.calc = FAIRChemCalculator(predictor, task_name = 'omat')
  rng = np.random.RandomState(0)
  MaxwellBoltzmannDistribution(atoms, temperature_K = temp, rng=rng)
  initial_positions = atoms.get_scaled_positions().copy()

  def check_initial_drift():
    if dyn.nsteps >= burnin_steps:
      return

    disp_frac = (atoms.get_scaled_positions() - 
      initial_positions)
    disp_frac -= np.round(disp_frac)
    disp_cart = disp_frac @ atoms.cell.array
    rms_displacement = np.sqrt(np.mean(np.sum(
      disp_cart ** 2, axis = 1)))

    if rms_displacement > max_rms_burnin:
      raise ASOSimulationException(
        f"RMS displacement exceeded threshold: "
        f"{rms_displacement} Å > {max_rms_burnin} Å "
        f"at step {dyn.nsteps}")

  dyn = Langevin(atoms,
    timestep = timestep * units.fs,
    temperature_K = temp,
    friction = friction / units.fs,
    fixcm = False)
  trajectory = Trajectory(trajectory_path, 'w', atoms)
  logger = MDLogger(dyn, atoms, log_path, header = True, mode = 'w')
  dyn.attach(trajectory.write, interval = loginterval)
  dyn.attach(logger, interval = loginterval)
  dyn.attach(check_initial_drift, interval = loginterval)
  try:
    dyn.run(nsteps)
  finally:
    trajectory.close()
    logger.close()

def autobk_lmfit(energy, mu, ek0, edge_step, std_de = None,
  rbkg=1, nknots=None, kmin=0, kmax=None, kweight=1, dk=0.1,
  win='hanning', k_std=None, chi_std=None, nfft=2048, kstep=0.05,
  nclamp=3, clamp_lo=0, clamp_hi=0,  **kws):
  """
  Essentially https://github.com/xraypy/xraylarch/blob/170d8ee342ad809722c9e6e0cd361fea0579865c/larch/xafs/autobk.py#L330 
  but with wider range of dE0, and outputs chi directly
  """
  try:
    from larch.xafs.autobk import _resid_lmfit, spline_eval, FMT_COEF
    from larch.xafs.xafsutils import ETOK
    from larch.xafs.xafsft import ftwindow
    from larch.math import index_of, index_nearest
  except ImportError as e:
    raise ImportError('Error loading xraylarch') from e
  try:
    from scipy.interpolate import splrep
  except ImportError as e:
    raise ImportError('Error loading scipy') from e
  try:
    from lmfit import Parameters, minimize
  except ImportError as e:
    raise ImportError('Error loading lmfit') from e
  
  # get array indices for rkbg and ek0: irbkg, iek0
  iek0 = index_of(energy, ek0)
  rgrid = np.pi / (kstep * nfft)
  if rbkg < 2 * rgrid: rbkg = 2 * rgrid

  # save ungridded k (kraw) and grided k (kout)
  # and ftwin (*k-weighting) for FT in residual
  enpe = energy[iek0:] - ek0
  kraw = np.sign(enpe) * np.sqrt(ETOK * abs(enpe))
  kmax = max(kraw) if kmax is None else max(0, min(max(kraw), kmax))
  kout  = kstep * np.arange(int(1.01 + kmax / kstep), dtype = 'float64')
  iemax = min(len(energy), 2 + index_of(energy, ek0 + kmax * kmax / ETOK)) - 1

  # pre-load FT window
  ftwin = kout ** kweight * ftwindow(kout, xmin = kmin, 
    xmax = kmax, window = win, dx = dk, dx2 = dk)
  # calc k-value and initial guess for y-values of spline params
  nspl = 1 + int(2 * rbkg * (kmax - kmin) / np.pi)
  irbkg = int(1 + (nspl - 1) * np.pi / (2 * rgrid * (kmax - kmin)))
  if nknots is not None: nspl = nknots
  nspl = max(5, min(128, nspl))
  spl_y, spl_k  = np.ones(nspl), np.zeros(nspl)
  for i in range(nspl):
      q  = kmin + i * (kmax - kmin) / (nspl - 1)
      ik = index_nearest(kraw, q)
      i1 = min(len(kraw) - 1, ik + 5)
      i2 = max(0, ik - 5)
      spl_k[i] = kraw[ik]
      spl_y[i] = (2 * mu[ik + iek0] + mu[i1 + iek0] + mu[i2 + iek0]) / 4.0

  order = 3
  qmin, qmax  = spl_k[0], spl_k[nspl - 1]
  knots = [spl_k[0] - 1.e-4 * (order - i) for i in range(order)]

  for i in range(order, nspl):
    knots.append((i - order) * (qmax - qmin)/(nspl - order + 1))
  qlast = knots[-1]
  for i in range(order + 1):
    knots.append(qlast + 1.e-4 * (i + 1))

  # coefs = [mu[index_nearest(energy, ek0 + q**2/ETOK)] for q in knots]
  knots, coefs, order = splrep(spl_k, spl_y, k=order)
  coefs[nspl:] = coefs[nspl-1]

  # set fit parameters from initial coefficients
  params = Parameters()
  for i in range(len(coefs)):
    params.add(name = FMT_COEF % i, value = coefs[i], vary = i < len(spl_y))

  if chi_std is not None:
    params.add(name='std_amp', value = 1.0, min =  0.05, max = 20.0, vary = True)
    if std_de is not None:
      params.add(name='std_de',  value = std_de, min = -30.0, max = 30.0, vary = False)
    else:
      params.add(name='std_de',  value = 0.0, min = -30.0, max = 30.0, vary = True)

  # do fit
  result = minimize(_resid_lmfit, params, method='leastsq',
    gtol = 1.e-6, ftol = 1.e-6, xtol = 1.e-6, epsfcn = 1.e-6,
    kws = dict(ncoefs = len(coefs), k_std = k_std, chi_std = chi_std,
      knots = knots, order = order,
      kraw = kraw[:iemax - iek0 + 1],
      mu = mu[iek0:iemax + 1], irbkg = irbkg, kout = kout,
      ftwin = ftwin, kweight = kweight,
      nfft = nfft, nclamp = nclamp,
      clamp_lo = clamp_lo, clamp_hi = clamp_hi))

  # write final results
  coefs = [result.params[FMT_COEF % i].value for i in range(len(coefs))]
  bkg, chi = spline_eval(kraw[:iemax - iek0 + 1], mu[iek0:iemax + 1],
    knots, coefs, order, kout)
  return chi / edge_step

def get_paths(sim_dir):
  try:
    from larch.xafs import feffpath
  except ImportError as e:
    raise ImportError('Error loading xraylarch') from e
  subdirs = sorted(
    (s for s in os.listdir(sim_dir)
      if os.path.isdir(os.path.join(sim_dir, s))),
    key=int,
  )
  feffdat_files = [np.sort([s for s in os.listdir(
    f'{sim_dir}/{sub_dir}') if s[:4] == 'feff' and s[-4:] == '.dat']
    ) for sub_dir in subdirs]
  feffpaths = [[feffpath(f'{sim_dir}/{subdirs[i]}/{feffdat_files[i][j
    ]}') for j in range(len(feffdat_files[i])
    )] for i in range(len(feffdat_files))]
  return feffpaths

def _interp_paths(q, feffk, values):
  """
  Vectorized equivalent of: np.interp(q, feffk, values[i])
  ----------
  q : (n_k,) array
  feffk : (n_feff_k,) array
  values : (n_paths, n_feff_k) array
  """
  q_clipped = np.clip(q, feffk[0], feffk[-1])
  inds = np.searchsorted(feffk, q_clipped, side='right')
  inds = np.clip(inds, 1, len(feffk) - 1)
  left = inds - 1
  right = inds
  weight = ((q_clipped - feffk[left]) / (feffk[right] - feffk[left]))
  return values[:, left] * (1.0 - weight)[None, :] + values[:, right] * weight[None, :]


def get_absorber_spectra(feffk, Reffs, degen, deltar, sigma2, third, 
  fourth, pha, amp, rep, lam, e0, ei, s02, k):
  """Calculate the summed chi(k) for all paths simultaneously."""
  try:
    from larch.xafs.xafsutils import ETOK
  except ImportError as e:
    raise ImportError('Error loading xraylarch') from e

  feffk = np.asarray(feffk)
  reff = Reffs[:, None]
  degen = degen[:, None]
  deltar = deltar[:, None]
  sigma2 = sigma2[:, None]
  third = third[:, None]
  fourth =  fourth[:, None]

  # Create the E0-shifted energy and wavenumber.
  en = k * k - e0 * ETOK
  q = np.sign(en) * np.sqrt(np.abs(en))

  # Look up FEFF values for every path and k value.
  pha = _interp_paths(q, feffk, pha)
  amp = _interp_paths(q, feffk, amp)
  rep = _interp_paths(q, feffk, rep)
  lam = _interp_paths(q, feffk, lam)

  # Complex wavenumber.
  pp = (rep + 1j / lam) ** 2 + 1j * ei * ETOK
  p = np.sqrt(pp)

  # XAFS equation. Arrays have shape (n_paths, n_k).
  cchi = np.exp(-2 * reff * p.imag - 2 * pp * (
    sigma2 - pp * fourth / 3) + 1j * (2 * 
    q[None, :] * reff + pha + 2 * p * (deltar - 
    2 * sigma2 / reff - 2 * pp * third / 3)))

  cchi = (degen * s02 * amp * cchi /
    (q[None, :] * (reff + deltar)**2))
  cchi[:, 0] = 2 * cchi[:, 1] - cchi[:, 2]

  return np.sum(cchi.imag, axis=0)

def get_spectra(feffk, Reffs, degen, deltar, sigma2, third, 
  fourth, pha, amp, rep, lam, e0, ei, s02, k):
  return np.mean(np.stack([get_absorber_spectra(feffk, Reffs[i], 
    degen[i], deltar[i], sigma2[i], third[i], fourth[i], 
    pha[i], amp[i], rep[i], lam[i], e0, ei, s02, k) for i in range(
    len(Reffs))]), axis = 0)

def get_kstats_from_traj(traj, sim_data):
  try:
    from scipy.stats import kstat
  except ImportError as e:
    raise ImportError('Error loading scipy') from e
  try:
    from pymatgen.io.ase import AseAtomsAdaptor
  except ImportError as e:
    raise ImportError('Error loading pymatgen') from e
  adaptor = AseAtomsAdaptor()
  starting_struct = adaptor.get_structure(traj[0])
  nabs = len(sim_data)
  sc_sf = np.sum(traj[0].get_atomic_numbers() == 22) // nabs
  nprim = len(traj[0]) // sc_sf
  nsegsmax = np.max(np.concatenate([np.array([len(
    sim_data[i][j]._feffdat.geom) - 1 for j in range(
    len(sim_data[i]))]) for i in range(len(sim_data))]))
  abs_inds = np.where(traj[0].get_atomic_numbers() == 22)[0][:nabs]
  lattice = starting_struct.lattice.matrix
  natoms = len(traj[0])
  nframes = 10000
  md_coords = np.zeros((nframes, natoms, 3))
  for fi, atoms in enumerate(traj[401:]):
    md_coords[fi] = atoms.positions

  Reffs = []
  kstats = []
  for abs_i, abs_ind in enumerate(abs_inds):
    path_inds_abs = np.ones((len(sim_data[abs_i]), sc_sf, 
      nsegsmax + 2), dtype = np.int64)
    for si in range(sc_sf):
      path_inds_abs[:, si, :] *= abs_ind + si * nprim

    path_ims_abs = np.zeros((len(sim_data[abs_i]), sc_sf, 
      nsegsmax + 2, 3))
    for fi, fp in enumerate(sim_data[abs_i]):
      geom = fp._feffdat.geometry
      for i in range(len(geom) - 1):
        for si in range(sc_sf):
          disp = np.asarray(geom[i][3:6], dtype=float)
          center = starting_struct[abs_ind + si * nprim].coords
          target = center + disp
          matches = starting_struct.get_sites_in_sphere(target, 0.1)

          if not matches:
            all_neighbors = starting_struct.get_sites_in_sphere(target, 2.0)
            nearest = all_neighbors[0] if all_neighbors else None
            raise ASOSimulationException(
              f"No FEFF geometry match:\n"
              f"  abs_i       = {abs_i}\n"
              f"  abs_ind     = {abs_ind}\n"
              f"  replica     = {si}\n"
              f"  path        = {fi}\n"
              f"  leg         = {i}\n"
              f"  absorber    = {center}\n"
              f"  displacement= {disp}\n"
              f"  target      = {target}\n"
              f"  nearest     = {nearest}\n"
            )
          nb = matches[0]
          path_inds_abs[fi, si, i + 1] = nb.index
          path_ims_abs[fi, si, i + 1] = nb.image
    path_shifts_abs = path_ims_abs @ lattice
    path_shift_dists_abs = path_shifts_abs[..., 1:, :
        ] - path_shifts_abs[..., :-1, :]

    Reffs_abs = np.zeros(len(sim_data[abs_i]))
    for fi, fp in enumerate(sim_data[abs_i]):
        geom = fp._feffdat.geometry
        Reffs_abs[fi] = np.linalg.norm(np.array(geom[0][3:6]))
        for i in range(len(geom) - 1):
            Reffs_abs[fi] += np.linalg.norm(np.array(
                geom[i+1][3:6]) - np.array(geom[i][3:6]))
        Reffs_abs[fi] /= 2
    dR_abs = md_coords[:, path_inds_abs[..., 1:], :] - md_coords[
        :, path_inds_abs[..., :-1], :] + path_shift_dists_abs[None]
    path_lengths_abs = np.linalg.norm(dR_abs, axis = -1).sum(axis = -1) / 2
    path_lengths_combined_abs = path_lengths_abs.transpose(0, 2, 1
        ).reshape(-1, len(sim_data[abs_i]))

    kstats_abs = np.array([kstat(path_lengths_combined_abs, 
        axis = 0, n = kn) for kn in range(1, 5)])
    kstats_abs[0, :] -= Reffs_abs
    kstats.append(kstats_abs)
    Reffs.append(Reffs_abs)

  deltar = [kstats[i][0, :] for i in range(len(kstats))]
  sigma2 = [kstats[i][1, :] for i in range(len(kstats))]
  third  = [kstats[i][2, :] for i in range(len(kstats))]
  fourth = [kstats[i][3, :] for i in range(len(kstats))]
  return Reffs, deltar, sigma2, third, fourth

def get_sim_spec(params, g, kstart, kstop, feffk, 
  Reffs, degens, deltar, sigma2, third, fourth, 
  pha, amp, rep, lam):
  chi = get_spectra(feffk, Reffs, degens, deltar, sigma2, third, fourth, 
    pha, amp, rep, lam, params['ΔE0'], 0.0, 1.0, g.k)
  
  scalar = np.linalg.norm(g.k[kstart:kstop] ** 3 * g.chi[kstart:kstop]
    ) / np.linalg.norm(g.k[kstart:kstop] ** 3 * chi[kstart:kstop]) 
  chi *= scalar

  fitchi = autobk_lmfit(g.energy, g.mu, g.ek0, g.edge_step, 
    std_de = None, k_std = g.k, chi_std = chi)
  return chi - fitchi + g.chi

def res_fun_lmfit(params, g, kstart, kstop, kw, feffk, Reffs, degens, 
  deltar, sigma2, third, fourth, pha, amp, rep, lam):
  chi = get_sim_spec(params, g, kstart, kstop, 
      feffk, Reffs, degens, deltar, sigma2, third, fourth, 
      pha, amp, rep, lam)
  expnorm = g.k[kstart:kstop] ** kw * g.chi[kstart:kstop]
  simnorm = g.k[kstart:kstop] ** kw * chi[kstart:kstop]
  dev = simnorm - expnorm
  return dev

def get_aligned_sim(sim, exp_g, traj, structure, kmin_fit = 3.0, kmax_fit = 12.0, 
  kwfit = 3, kwout = 3):
  try:
    from larch.math import index_of
  except ImportError as e:
    raise ImportError('Error loading xraylarch') from e
  try:
    from lmfit import Parameters, Minimizer
  except ImportError as e:
    raise ImportError('Error loading lmfit') from e
  degens = [np.array([sim[i][j]._feffdat.degen for j in range(len(sim[i]))]
    ) for i in range(len(sim))]
  feffk = sim[0][0]._feffdat.k
  Reffs, deltar, sigma2, third, fourth = get_kstats_from_traj(traj, sim)
  amp = [np.array([path._feffdat.amp for path in abs_data]) for abs_data in sim]
  pha = [np.array([path._feffdat.pha for path in abs_data]) for abs_data in sim]
  lam = [np.array([path._feffdat.lam for path in abs_data]) for abs_data in sim]
  rep = [np.array([path._feffdat.rep for path in abs_data]) for abs_data in sim]

  kstart = index_of(exp_g.k, kmin_fit)
  kstop = index_of(exp_g.k, kmax_fit)

  params = Parameters()
  params.add('ΔE0', value = 0.0, min = -30.0, max = 30.0, vary = True)

  dE0_grid = np.arange(-30.0, 30.0, 0.5)
  chi2_grid = np.empty_like(dE0_grid)

  for i, dE0 in enumerate(dE0_grid):
    p = params.copy()
    p['ΔE0'].set(value=dE0)
    residual = res_fun_lmfit(p, exp_g, kstart, kstop, kwfit,
      feffk, Reffs, degens, deltar, sigma2, third, fourth, 
      pha, amp, rep, lam)
    chi2_grid[i] = np.dot(residual, residual)

  params['ΔE0'].set(value = dE0_grid[np.argmin(chi2_grid)])
  minner = Minimizer(res_fun_lmfit, params, fcn_args = [
    exp_g, kstart, kstop, kwfit, feffk, Reffs, degens, 
    deltar, sigma2, third, fourth, pha, amp, rep, lam])
  result = minner.minimize()
  
  sim_chi = get_sim_spec(result.params, exp_g, kstart, kstop, 
    feffk, Reffs, degens, deltar, sigma2, third, fourth, 
    pha, amp, rep, lam)
  return sim_chi[kstart:kstop] * exp_g.k[kstart:kstop] ** kwout

@registry.register_simulation("EXAFS")
class EXAFS(BaseSimulation):
  def __init__(self, initial_structure, exp_g, feff_location = "", 
    folder = "", md_folder = "", 
    absorber = 'Co', edge = 'K', radius = 10.0, fit_kmin = 3.0, 
    fit_kmax = 12.0, time_limit = 240,
    additional_settings = {'EXAFS': 12.0, 'SCF': '4.5 0 30 .2 1'},
    sh_template = None, 
    sbatch_template = None, sbatch_group_template = None, 
    sbatch_md_template = None, sbatch_md_group_template = None,
    sbatch_opt_template = None,
    number_absorbers = None, save_sim = True, kwfit = 3,
    **kwargs) -> None:
    self.exp_g = exp_g
    kmini = np.argmin(np.abs(exp_g.k - fit_kmin))
    kmaxi = np.argmin(np.abs(exp_g.k - fit_kmax))
    self.outdim = int(kmaxi - kmini)
    self.fit_kmin = fit_kmin
    self.fit_kmax = fit_kmax
    self.feff_location = feff_location
    self.parent_folder = folder
    self.md_parent_folder = md_folder
    self.absorber = absorber
    self.edge = edge
    self.radius = radius
    self.additional_settings = additional_settings
    self.mask = [x.symbol == self.absorber 
      for x in initial_structure.species]
    self.N = len(self.mask)
    self.sbatch_template = sbatch_template
    self.sbatch_group_template = sbatch_group_template
    self.sbatch_md_template = sbatch_md_template
    self.sbatch_md_group_template = sbatch_md_group_template
    self.sbatch_opt_template = sbatch_opt_template
    self.sh_template = sh_template
    self.number_absorbers = number_absorbers
    self.save_sim = save_sim
    self.time_limit = time_limit
    self.structure = None
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
    try:
      from pymatgen.io import feff
      from pymatgen.io.feff.sets import MPEXAFSSet
    except ImportError as e:
      raise ImportError('Error loading pymatgen') from e
    try:
      from subprocess
    except ImportError as e:
      raise ImportError('Error loading subprocess') from e
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

    if self.md_parent_folder is not None:
      md_subfolders = [int(x) for x in os.listdir(self.md_parent_folder)]
      new_md_folder = os.path.join(self.md_parent_folder, str(np.max(
        md_subfolders) + 1 if len(md_subfolders) > 0 else 0))
      os.mkdir(new_md_folder)
    else:
      new_md_folder = None
    
    # Setup FEFF files
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
  
    # Setup MD Files
    if self.md_parent_folder is not None:
      with open(os.path.join(new_md_folder, 'structure.json'), 'w') as md_struct_file:
        json.dump(structure.as_dict(), md_struct_file)

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
      sbatch_data = sbatch_data.replace('##JOB_NAME##', f'FEFF_{job_name}')
      new_job_file = os.path.join(new_folder, 'job.sbatch')
      with open(new_job_file, 'w') as file:
        file.write(sbatch_data)
      
      try:
        subprocess.check_output(["sbatch", f"{new_job_file}"])
      except subprocess.CalledProcessError as e:
        print(e.output)

    elif self.sh_template is not None:
      try:
        from stat
      except ImportError as e:
        raise ImportError('Error loading stat') from e
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

    if self.sbatch_md_template is not None:
      with open(self.sbatch_md_template, 'r') as file:
        sbatch_data = file.read()
      job_name = int(time.time()) % 604800
      sbatch_data = sbatch_data.replace('##DIRECTORY##', new_md_folder)
      sbatch_data = sbatch_data.replace('##JOB_NAME##', f'MD_{job_name}'))
      new_job_file = os.path.join(new_md_folder, 'job.sbatch')
      with open(new_job_file, 'w') as file:
        file.write(sbatch_data)
      try:
        subprocess.check_output(["sbatch", f"{new_job_file}"])
      except subprocess.CalledProcessError as e:
        print(e.output)

    self.group = group
    self.folder = new_folder
    self.md_folder = new_md_folder
    self.params = params
    self.inds = absorber_indices 

  def check_done(self):
    if not os.path.isdir(self.folder):
      raise ASOSimulationException(f"Folder {self.folder} was deleted")

    if self.group:
      return self.check_opt_done()

    for i in range(len(self.inds)):
      new_abs_folder = os.path.join(self.folder, str(i))
      if not os.path.isfile(os.path.join(new_abs_folder, "DONE")):
        #print(f'waiting for {os.path.join(new_abs_folder, "DONE")}')
        return False

    if self.md_folder is not None:
      if not os.path.isfile(os.path.join(self.md_folder, "DONE")):
        return False
    
    return True

  def check_opt_done(self):
    if not os.path.isdir(self.folder):
      raise ASOSimulationException(f"Folder {self.folder} was deleted")

    return os.path.isfile(os.path.join(self.folder, "DONE"))

  def resolve(self):
    try:
      from shutil
    except ImportError as e:
      raise ImportError('Error loading shutil') from e
    try:
      from subprocess
    except ImportError as e:
      raise ImportError('Error loading subprocess') from e
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
      if not os.path.isfile(os.path.join(self.folder, 'chi_k.dat')):
        raise ASOSimulationException(f"Optimization failed")

    for i, absorb_ind in enumerate(self.inds):
      if not self.save_sim:
        shutil.rmtree(os.path.join(self.folder, str(i)))
    
    chi_ks = np.genfromtxt(os.path.join(self.folder, 'chi_k.dat'))
    return chi_ks 

  def garbage_collect(self, is_better):
    try:
      from shutil
    except ImportError as e:
      raise ImportError('Error loading shutil') from e
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
