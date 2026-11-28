from activestructopt.simulation.base import BaseSimulation, ASOSimulationException
from activestructopt.common.registry import registry
from activestructopt.common.d2min import D2minMonitor, block_frames
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io import feff
from pymatgen.io.feff.sets import MPEXAFSSet
from itertools import product
import numpy as np
import subprocess
import shutil
import stat
import time
import json
import os

def write_group(filename, g):
  items = g.items()
  save_dict = {}
  for key, value in items:
    if type(value) in [float, int, str, tuple]:
      save_dict[key] = value
    elif type(value) is np.ndarray:
      save_dict[key] = value.tolist()
    else:
      print(f'not saving {key} of type {type(value)}')
  with open(filename, "w") as f:
    json.dump(save_dict, f)

def read_group(filename):
  try:
    from larch import Group
  except ImportError as e:
    raise ImportError('Error loading xraylarch') from e
  with open(filename, 'r') as file:
    data = json.load(file)
  for key in data.keys():
    if isinstance(data[key], list):
      data[key] = np.array(data[key])
  return Group(**data)

def _md_predictor(mlip, inference_settings, device):
  try:
    from fairchem.core import pretrained_mlip
  except ImportError as e:
    raise ImportError('Error loading fairchem.core') from e
  return pretrained_mlip.get_predict_unit(mlip, device = device,
    inference_settings = inference_settings)

def _min_repetitions(cell_lengths, minimum_length):
  """Repetitions along each lattice vector so that every supercell length is
  at least minimum_length."""
  return np.maximum(1, np.ceil(minimum_length / cell_lengths).astype(int))

def _grow_repetitions(cell_lengths, repetitions):
  """The next larger supercell: one more repetition along the currently
  shortest supercell length, keeping the supercell as close to isotropic as
  possible (the minimum length never decreases)."""
  repetitions = repetitions.copy()
  repetitions[np.argmin(cell_lengths * repetitions)] += 1
  return repetitions

def _probe_supercell(atoms_prim, repetitions, calc, device):
  """Peak GPU memory (bytes) of a force evaluation of the supercell with the
  calculator calc, which is reused (with its loaded model) across sizes.
  Raises torch.cuda.OutOfMemoryError if the supercell does not fit."""
  import torch
  from ase.build.supercells import make_supercell
  atoms = make_supercell(atoms_prim, np.diag(repetitions))
  calc.reset()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats(device)
  try:
    atoms.calc = calc
    atoms.get_forces()
    torch.cuda.synchronize(device)
  finally:
    # Don't keep the probe's results (or atoms) alive in the calculator
    calc.reset()
  return torch.cuda.max_memory_allocated(device)

def _largest_md_supercell(atoms_prim, minimum_length, calc, device,
  gpu_memory_fraction, max_atoms):
  """The repetitions of the largest supercell (grown from the minimum length
  supercell by _grow_repetitions) whose force evaluation with calc fits in
  gpu_memory_fraction of the GPU memory (with at most max_atoms atoms, if not
  None). The memory is modeled as linear in the number of atoms, from
  evaluations of the minimum supercell and one about twice as large. The
  chosen supercell is then evaluated, falling back to smaller supercells if
  it does not fit."""
  import torch
  cell_lengths = np.linalg.norm(atoms_prim.cell.array, axis = 1)
  nprim = len(atoms_prim)
  min_reps = _min_repetitions(cell_lengths, minimum_length)

  # Candidate supercells, in order of increasing size
  candidates = [min_reps]
  while len(candidates) < 10000:
    reps = _grow_repetitions(cell_lengths, candidates[-1])
    if max_atoms is not None and nprim * np.prod(reps) > max_atoms:
      break
    candidates.append(reps)
  if len(candidates) == 1:
    return min_reps
  natoms = np.array([nprim * np.prod(reps) for reps in candidates])

  # Linear memory model from two evaluations
  budget = gpu_memory_fraction * torch.cuda.get_device_properties(
    device).total_memory
  mem_min = _probe_supercell(atoms_prim, min_reps, calc, device)
  probe_i = min(int(np.searchsorted(natoms, 2 * natoms[0])),
    len(candidates) - 1)
  try:
    mem_probe = _probe_supercell(atoms_prim, candidates[probe_i], calc,
      device)
  except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    return min_reps
  per_atom = (mem_probe - mem_min) / (natoms[probe_i] - natoms[0])
  if per_atom <= 0:
    chosen_i = len(candidates) - 1
  else:
    predicted = mem_min + per_atom * (natoms - natoms[0])
    chosen_i = max(0, int(np.searchsorted(predicted, budget,
      side = 'right')) - 1)

  # Check the chosen supercell, falling back to ~10% fewer atoms on OOM
  while chosen_i > 0:
    try:
      _probe_supercell(atoms_prim, candidates[chosen_i], calc, device)
      return candidates[chosen_i]
    except torch.cuda.OutOfMemoryError:
      torch.cuda.empty_cache()
      chosen_i = min(chosen_i - 1, int(np.searchsorted(natoms,
        0.9 * natoms[chosen_i], side = 'right')) - 1)
  return min_reps

def run_md(structure, trajectory_path = 'md.traj',
  log_path = 'md.log', timestep = 1.25, temp = 300,
  friction = 0.002, loginterval = 4, nsteps = 41600,
  burnin_steps = 1600, d2min_block_ps = 0.5, d2min_cutoff = 3.5,
  d2min_level = 0.05, d2min_max_frac = 0.5, d2min_max_atom = 0.5,
  minimum_length = 8.4,
  maximize_supercell = True, gpu_memory_fraction = 0.8, max_atoms = None,
  min_production_steps = 4000, mlip = 'uma-s-1p2p1', task_name = 'omat',
  inference_settings = 'nitro', device = 'cuda'):
  """Langevin MD of a supercell of structure, writing every loginterval
  steps to trajectory_path, and failing if the MD rearranges the structure at
  any point of the run: if, in any block of d2min_block_ps, the D^2_min of
  the block-averaged positions relative to the starting structure (see
  common/d2min.py) is above d2min_level for more than d2min_max_frac of the
  atoms, or above d2min_max_atom for any atom. The same check is applied to
  the burn-in alone by the MLIPBurnIn constraint.

  The minimum supercell has every length at least minimum_length (Å). If
  maximize_supercell, the supercell is instead the largest (with that
  minimum length preserved) whose force evaluation fits in
  gpu_memory_fraction of the GPU memory, with at most max_atoms atoms if not
  None. The cost per step is roughly flat in the number of atoms, while the
  statistics of the configurational averages scale with atoms times frames,
  so the production steps after the burn-in (nsteps - burnin_steps for the
  minimum supercell) are divided by the number of minimum supercells in the
  supercell, down to min_production_steps, and rounded up to a multiple of
  loginterval. The burn-in is unchanged, so the burn-in frames of the
  trajectory (first_frame in _iter_path_lengths) are too."""
  try:
    from ase import units
    from ase.build.supercells import make_supercell
    from ase.constraints import FixCom
    from ase.io import Trajectory
    from ase.md import MDLogger
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from fairchem.core import FAIRChemCalculator
  except ImportError as e:
    raise ImportError('Error loading required packages for MD') from e

  atoms_prim = AseAtomsAdaptor().get_atoms(structure)
  cell_lengths = np.linalg.norm(atoms_prim.cell.array, axis = 1)
  min_reps = _min_repetitions(cell_lengths, minimum_length)
  # The model is loaded once, and the calculator shared by the supercell
  # probes and the MD
  calc = FAIRChemCalculator(_md_predictor(mlip, inference_settings, device),
    task_name = task_name)
  repetitions = min_reps
  if maximize_supercell:
    repetitions = _largest_md_supercell(atoms_prim, minimum_length, calc,
      device, gpu_memory_fraction, max_atoms)

  # Fewer production steps for a larger supercell
  size_ratio = np.prod(repetitions) / np.prod(min_reps)
  production_steps = max(min_production_steps,
    (nsteps - burnin_steps) / size_ratio)
  production_steps = int(loginterval * np.ceil(
    production_steps / loginterval))
  total_steps = burnin_steps + production_steps
  natoms = len(atoms_prim) * int(np.prod(repetitions))
  print(f'MD supercell {repetitions.tolist()}: {natoms} atoms '
    f'({size_ratio:.2f}x the minimum), {burnin_steps} burn-in + '
    f'{production_steps} production steps', flush = True)

  atoms = make_supercell(atoms_prim, np.diag(repetitions))
  atoms.set_constraint(FixCom())
  atoms.calc = calc
  rng = np.random.RandomState(0)
  MaxwellBoltzmannDistribution(atoms, temperature_K = temp, rng=rng)
  monitor = D2minMonitor(atoms, block_frames(d2min_block_ps, timestep,
    loginterval), cutoff = d2min_cutoff, level = d2min_level,
    max_frac = d2min_max_frac, max_atom = d2min_max_atom)

  def check_d2min():
    reason = monitor.update(atoms)
    if reason is not None:
      raise ASOSimulationException(
        f'Structure rearranged during the MD at step {dyn.nsteps}: {reason}')

  dyn = Langevin(atoms,
    timestep = timestep * units.fs,
    temperature_K = temp,
    friction = friction / units.fs,
    fixcm = False)
  trajectory = Trajectory(trajectory_path, 'w', atoms)
  logger = MDLogger(dyn, atoms, log_path, header = True, mode = 'w')
  dyn.attach(trajectory.write, interval = loginterval)
  dyn.attach(logger, interval = loginterval)
  dyn.attach(check_d2min, interval = loginterval)
  try:
    dyn.run(total_steps)
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
    from larch.xafs.autobk import _lmfit_resid, spline_eval, FMT_COEF
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
  result = minimize(_lmfit_resid, params, method='leastsq',
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

BOHR = 0.529177249  # Angstrom, as in FEFF's rdfbin.f90

class FeffPath:
  """
  The subset of a FEFF path (as in feffNNNN.dat) used in this module,
  with all quantities in Angstrom units.

  positions: (nleg, 3) atom positions along the path, absorber at the origin.
    Rows 0..nleg-2 are the scatterers in order, and the last row is the
    absorber (same ordering as larch's FeffDatFile.geometry).
  k, amp, pha, lam, rep: same as larch's FeffDatFile attributes.
  """
  def __init__(self, index, nleg, degen, reff, abs_Z, positions,
    k, amp, pha, lam, rep):
    self.index = index
    self.nleg = nleg
    self.degen = degen
    self.reff = reff
    self.abs_Z = abs_Z
    self.positions = positions
    self.k = k
    self.amp = amp
    self.pha = pha
    self.lam = lam
    self.rep = rep

def _unpad(s, npack):
  """
  Decode a string of concatenated Packed Ascii Data (PAD) numbers, each
  npack characters long. Vectorized port of unpad from feff10's padlib.f90.
  """
  c = np.frombuffer(s.encode('ascii'), dtype = np.uint8).reshape(
    -1, npack).astype(np.int64) - 37
  iexp = c[:, 0] - 45
  isgn = (c[:, 1] % 2) * 2 - 1
  total = (c[:, 1] // 2) / 90.0 ** 2
  for i in range(npack - 1, 1, -1):
    total += c[:, i] / 90.0 ** (i + 1)
  return 2 * isgn * 90 * total * 10.0 ** iexp

class _FeffBinReader:
  def __init__(self, filename):
    with open(filename, 'r') as f:
      self.lines = [line.strip() for line in f]
    self.i = 0

  def next_line(self):
    line = self.lines[self.i]
    self.i += 1
    return line

  def done(self):
    while self.i < len(self.lines) and self.lines[self.i] == '':
      self.i += 1
    return self.i >= len(self.lines)

  def read_pad(self, npack, npts, is_complex = False):
    """Port of rdpadr/rdpadc from feff10's padlib.f90"""
    marker = '$' if is_complex else '!'
    width = 2 * npack if is_complex else npack
    chunks = []
    nread = 0
    while nread < npts:
      line = self.next_line()
      nline = (len(line) - 1) // width
      if line[:1] != marker or nline <= 0:
        raise ASOSimulationException(f'Bad PAD data in feff.bin: {line}')
      chunks.append(line[1:1 + nline * width])
      nread += nline
    values = _unpad(''.join(chunks), npack)
    if is_complex:
      values = values[0::2] + 1j * values[1::2]
    return values[:npts]

def _get_ne1(abs_dir, xk):
  """
  Number of energy points on the main (real) axis, which is what FEFF
  writes to feffNNNN.dat. Read from phase.bin if present; otherwise, use
  the points before xk stops increasing (where the other contours start).
  """
  phase_bin = os.path.join(abs_dir, 'phase.bin')
  if os.path.isfile(phase_bin):
    with open(phase_bin, 'r') as f:
      return int(f.readline().split()[2])
  nonincreasing = np.flatnonzero(np.diff(xk) <= 0)
  return len(xk) if len(nonincreasing) == 0 else nonincreasing[0] + 1

def read_feff_bin(abs_dir, filename = 'feff.bin'):
  """
  Read the paths in feff.bin, following rdfbin.f90 from feff10, and
  convert them to the quantities feffdt.f90 writes to feffNNNN.dat.
  """
  reader = _FeffBinReader(os.path.join(abs_dir, filename))

  line = reader.next_line()
  if not line.startswith('#_feff.bin v03'):
    raise ASOSimulationException(f'Unsupported feff.bin version: {line}')
  npot, ne, npack = [int(x) for x in reader.next_line()[2:].split()[:3]]
  reader.next_line() # '#=' ihole, iorder, ilinit, rnrmav, xmu, edge
  words = reader.next_line()[2:].split()
  iz = [int(x) for x in words[npot + 1:2 * npot + 2]]

  reader.read_pad(npack, ne, is_complex = True) # central atom phase shifts
  ck = reader.read_pad(npack, ne, is_complex = True)
  xk = reader.read_pad(npack, ne)

  ne1 = _get_ne1(abs_dir, xk)
  ck = ck[:ne1]
  xk = xk[:ne1]

  # Quantities shared by all paths, as in feffdt.f90
  k = xk / BOHR
  rep = ck.real / BOHR
  xlam = np.full(ne1, 1.0e10)
  absorbing = np.abs(ck.imag) > 1.0e-16
  xlam[absorbing] = 1 / ck.imag[absorbing]
  lam = xlam * BOHR

  paths = []
  while not reader.done():
    line = reader.next_line()
    if not line.startswith('##'):
      raise ASOSimulationException(f'Bad path header in feff.bin: {line}')
    words = line[2:].split()
    index = int(words[0])
    nleg = int(words[1])
    degen = float(words[2])
    reff = float(words[3]) # Angstrom
    ipot = [int(x) for x in words[5:5 + nleg]]

    rat = reader.read_pad(npack, 3 * nleg).reshape(nleg, 3) * BOHR
    reader.read_pad(npack, nleg) # beta
    reader.read_pad(npack, nleg) # eta
    reader.read_pad(npack, nleg) # ri
    achi = reader.read_pad(npack, ne)[:ne1]
    phchi = reader.read_pad(npack, ne)[:ne1]

    # amp = mag[feff] * red factor and pha = real[2*phc] + phase[feff],
    # in which the central atom terms cancel
    reff_bohr = reff / BOHR
    amp = (np.abs(achi * xk) * reff_bohr ** 2 *
      np.exp(2 * reff_bohr / xlam) * BOHR)
    # genfmt already removed 2 pi jumps from phchi, so it is continuous and
    # safe to interpolate. Like feffdt, start in (-pi, pi]
    pha = phchi - 2 * np.pi * np.round(phchi[0] / (2 * np.pi))

    paths.append(FeffPath(index, nleg, degen, reff, iz[ipot[-1]], rat,
      k, amp, pha, lam, rep))

  paths.sort(key = lambda p: p.index)
  return paths

def get_paths(sim_dir):
  subdirs = sorted(
    (s for s in os.listdir(sim_dir)
      if s.isdigit()
      and os.path.isdir(os.path.join(sim_dir, s))),
    key=int)
  return [read_feff_bin(os.path.join(sim_dir, sub_dir)
    ) for sub_dir in subdirs]

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


def get_absorber_spectra(feffk, Reffs, degen, deltar, avg_factor,
  pha, amp, rep, lam, e0, ei, s02, k):
  """
  Calculate the summed chi(k) for all paths simultaneously, using the
  configurational average over the MD frames rather than a cumulant
  expansion. For each path,

    chi(k) = Im[degen * s02 * amp / q
      * exp(-2 * reff * Im(p) + i * (2 * q * reff + pha))
      * <exp(2 i p (R - reff)) / R^2>]

  where the average is over the MD distribution of the half path length R,
  and is split as exp(2 i p deltar) * avg_factor, with deltar = <R> - reff
  and avg_factor = <exp(2 i p (R - <R>)) / R^2> (from
  get_path_averages_from_traj, tabulated on feffk). For a Gaussian
  distribution this reduces to the usual cumulant expression, but unlike the
  truncated cumulant expansion it is bounded for any distribution.
  """
  try:
    from larch.xafs.xafsutils import ETOK
  except ImportError as e:
    raise ImportError('Error loading xraylarch') from e

  feffk = np.asarray(feffk)
  reff = Reffs[:, None]
  degen = degen[:, None]
  deltar = deltar[:, None]

  # Create the E0-shifted energy and wavenumber.
  en = k * k - e0 * ETOK
  q = np.sign(en) * np.sqrt(np.abs(en))

  # Look up FEFF values (and the MD averages) for every path and k value.
  pha = _interp_paths(q, feffk, pha)
  amp = _interp_paths(q, feffk, amp)
  rep = _interp_paths(q, feffk, rep)
  lam = _interp_paths(q, feffk, lam)
  avg_factor = _interp_paths(q, feffk, avg_factor)

  # Complex wavenumber.
  pp = (rep + 1j / lam) ** 2 + 1j * ei * ETOK
  p = np.sqrt(pp)

  # XAFS equation. Arrays have shape (n_paths, n_k).
  cchi = np.exp(-2 * reff * p.imag + 1j * (2 * q[None, :] * reff + pha
    + 2 * p * deltar)) * avg_factor

  cchi = degen * s02 * amp * cchi / q[None, :]
  cchi[:, 0] = 2 * cchi[:, 1] - cchi[:, 2]

  return np.sum(cchi.imag, axis=0)

def get_spectra(feffk, Reffs, degen, deltar, avg_factor,
  pha, amp, rep, lam, e0, ei, s02, k, avg_spec = True):
  spectra = np.stack([get_absorber_spectra(feffk, Reffs[i],
    degen[i], deltar[i], avg_factor[i],
    pha[i], amp[i], rep[i], lam[i], e0, ei, s02, k) for i in range(
    len(Reffs))])
  return np.mean(spectra, axis = 0) if avg_spec else spectra

def _iter_path_lengths(traj, sim_data, abs_Z, first_frame = 401,
  frame_batch_size = 250):
  """
  Match each FEFF path of each absorber to atoms of the MD supercell, and
  yield the half path lengths over the MD frames from first_frame on, in
  batches, as (abs_i, Reffs_abs, path_lengths) with path_lengths of shape
  (nframes_batch, npaths, sc_sf), one column per supercell replica of the
  absorber. Batches are yielded absorber by absorber. path_lengths is a view
  of a buffer that is reused, so it is only valid until the next batch.
  """
  try:
    from scipy.spatial import cKDTree
  except ImportError as e:
      raise ImportError('Error loading scipy') from e
  if len(traj) <= first_frame:
    raise ASOSimulationException(
      f'Trajectory has {len(traj)} frames, need more than {first_frame}')
  adaptor = AseAtomsAdaptor()
  starting_struct = adaptor.get_structure(traj[0])
  nabs = len(sim_data)
  sc_sf = np.sum(traj[0].get_atomic_numbers() == abs_Z) // nabs
  nprim = len(traj[0]) // sc_sf
  nsegs = [np.array([sim_data[i][j].nleg - 1 for j in range(
    len(sim_data[i]))]) for i in range(len(sim_data))]
  nsegsmax = np.max(np.concatenate(nsegs))
  abs_inds = np.where(traj[0].get_atomic_numbers() == abs_Z)[0][:nabs]
  lattice = starting_struct.lattice.matrix

  site_coords = np.asarray(
    starting_struct.cart_coords,
    dtype=np.float64,
  )

  natoms = len(starting_struct)

  image_shifts = np.asarray(
    list(product((-1, 0, 1), repeat=3)),
    dtype=np.int64,
  ) @ lattice

  periodic_coords = (
    site_coords[None, :, :]
    + image_shifts[:, None, :]
  ).reshape(-1, 3)

  periodic_site_indices = np.tile(
    np.arange(natoms, dtype=np.int64),
    len(image_shifts),
  )

  periodic_shifts = np.repeat(image_shifts, natoms, axis = 0)

  periodic_tree = cKDTree(periodic_coords)
  md_coords = np.stack([traj[i].positions for i in range(first_frame, len(traj))])
  for abs_i, abs_ind in enumerate(abs_inds):
    path_inds_abs = []
    path_shift_dists_abs = []
    nseg_inds = []
    for nseg in np.arange(1, nsegsmax + 1):
      nsegs_is = np.where(nsegs[abs_i] == nseg)[0]
      path_inds_abs_nseg = np.ones((len(nsegs_is), sc_sf,
        nseg + 2), dtype = np.int64)
      for si in range(sc_sf):
        path_inds_abs_nseg[:, si, :] *= abs_ind + si * nprim

      path_shifts_abs_nseg = np.zeros((len(nsegs_is), sc_sf,
        nseg + 2, 3))

      # This absorber may have no paths with this number of legs (nsegsmax
      # is taken over all absorbers), so keep placeholders to stay aligned
      if len(nsegs_is) == 0:
        path_inds_abs.append(path_inds_abs_nseg)
        path_shift_dists_abs.append(np.diff(path_shifts_abs_nseg, axis=-2))
        nseg_inds.append(nsegs_is)
        continue

      targets = []
      locations = []

      for fi, fp in enumerate(nsegs_is):
        geom = sim_data[abs_i][fp].positions

        for leg_i in range(nseg):
          disp = geom[leg_i]

          for si in range(sc_sf):
            center_index = abs_ind + si * nprim
            target = site_coords[center_index] + disp

            targets.append(target)
            locations.append((fi, si, leg_i + 1))

      targets = np.asarray(targets, dtype=np.float64)
      locations = np.asarray(locations, dtype=np.int64)

      distances, matches = periodic_tree.query(targets, k=1)

      bad = np.flatnonzero(distances > 0.1)
      if bad.size:
        bad_i = bad[0]
        fi, si, stored_leg_i = locations[bad_i]
        match = matches[bad_i]

        raise ASOSimulationException(
          f"No FEFF geometry match:\n"
          f"  abs_i         = {abs_i}\n"
          f"  abs_ind       = {abs_ind}\n"
          f"  replica       = {si}\n"
          f"  path          = {fi}\n"
          f"  leg           = {stored_leg_i - 1}\n"
          f"  target        = {targets[bad_i]}\n"
          f"  nearest site  = {periodic_site_indices[match]}\n"
          f"  nearest shift = {periodic_shifts[match]}\n"
          f"  distance      = {distances[bad_i]} Å"
        )

      path_i = locations[:, 0]
      replica_i = locations[:, 1]
      leg_i = locations[:, 2]

      path_inds_abs_nseg[path_i, replica_i, leg_i] = (
        periodic_site_indices[matches]
      )

      path_shifts_abs_nseg[path_i, replica_i, leg_i] = (
        periodic_shifts[matches]
      )

      path_inds_abs.append(path_inds_abs_nseg)
      path_shift_dists_abs.append(np.diff(path_shifts_abs_nseg, axis=-2))
      nseg_inds.append(nsegs_is)

    Reffs_abs = np.zeros(len(sim_data[abs_i]))
    for fi, fp in enumerate(sim_data[abs_i]):
      geom = fp.positions
      Reffs_abs[fi] = np.linalg.norm(geom[0])
      for i in range(len(geom) - 1):
        Reffs_abs[fi] += np.linalg.norm(geom[i+1] - geom[i])
      Reffs_abs[fi] /= 2

    npaths = len(sim_data[abs_i])
    path_lengths = np.zeros((frame_batch_size, npaths, sc_sf))

    for frame_start in range(first_frame, len(traj), frame_batch_size):
      frame_stop = min(frame_start + frame_batch_size, len(traj))
      nframes_batch = frame_stop - frame_start

      positions = md_coords[(frame_start - first_frame):(frame_stop - first_frame)]

      for nseg in np.arange(1, nsegsmax + 1):
        if len(nseg_inds[nseg - 1]) == 0:
          continue
        path_coords = positions[:, path_inds_abs[nseg - 1], :]
        dR = np.diff(path_coords, axis=-2)
        del path_coords
        dR += path_shift_dists_abs[nseg - 1][None]

        # Shape: (nframes_batch, npaths, sc_sf)
        path_lengths[:nframes_batch, nseg_inds[nseg - 1], :] = (
          np.sqrt(np.einsum("...i,...i->...", dR, dR))
          .sum(axis=-1)
          / 2
        )

      yield abs_i, Reffs_abs, path_lengths[:nframes_batch]

def _merge_hists(keys_a, counts_a, keys_b, counts_b):
  """Merge two sparse histograms given as (sorted unique keys, counts)"""
  keys, inverse = np.unique(np.concatenate([keys_a, keys_b]),
    return_inverse = True)
  counts = np.bincount(inverse.ravel(),
    weights = np.concatenate([counts_a, counts_b]))
  return keys, counts

def get_path_averages_from_traj(traj, sim_data, abs_Z, bin_width = 0.005,
  chunk_elements = 4_000_000):
  """
  Configurational averages over the MD for each path, used in place of the
  cumulant expansion. Returns (Reffs, deltar, avg_factor), each a list over
  absorbers:
    Reffs: (npaths,) half path lengths of the FEFF geometry
    deltar: (npaths,) <R> - Reffs
    avg_factor: (npaths, n_feff_k) complex, <exp(2 i p (R - <R>)) / R^2>,
      on the FEFF k grid of the paths, with p = rep + i / lam the complex
      wavenumber at that k (ei = 0)
  where R is the half path length, averaged over frames and supercell
  replicas of the absorber. The mean phase exp(2 i p deltar) is left out of
  avg_factor so that what is interpolated in k is smooth.

  The half path lengths are binned into a histogram of width bin_width (Å)
  per path, which is exact up to a relative error of about
  (2 k bin_width)^2 / 24 (about 0.1% at k = 16 Å^-1 for 0.005 Å).
  """
  # Bins are keyed as path * key_stride + (bin + key_offset), so one sparse
  # histogram holds all paths of an absorber without knowing their ranges.
  # Only this running histogram (whose size is set by the number of occupied
  # bins, not the number of frames) is kept between batches.
  key_stride = np.int64(2 ** 32)
  key_offset = np.int64(2 ** 31)
  nabs = len(sim_data)
  Reffs = [None] * nabs
  deltar = [None] * nabs
  avg_factor = [None] * nabs

  def finalize(abs_i, keys, hist_counts, sums, count):
    """Average over the histogram of absorber abs_i"""
    Reffs_abs = Reffs[abs_i]
    npaths = len(Reffs_abs)
    xbar = sums / count
    weights = hist_counts / count
    # Sorted by key, so the bins of each path are contiguous
    path_ids = keys // key_stride
    x = (keys % key_stride - key_offset + 0.5) * bin_width
    R = Reffs_abs[path_ids] + x
    if np.any(R <= 0):
      raise ASOSimulationException('Non-positive path length in MD')

    p = np.array([path.rep + 1j / path.lam for path in sim_data[abs_i]])
    nk = p.shape[1]
    avg_abs = np.zeros((npaths, nk), dtype = np.complex128)
    chunk = max(1, chunk_elements // nk)
    for start in range(0, len(keys), chunk):
      stop = min(start + chunk, len(keys))
      ids = path_ids[start:stop]
      vals = (weights[start:stop] / R[start:stop] ** 2)[:, None] * np.exp(
        2j * (x[start:stop] - xbar[ids])[:, None] * p[ids])
      uniq, seg_starts = np.unique(ids, return_index = True)
      avg_abs[uniq] += np.add.reduceat(vals, seg_starts, axis = 0)

    deltar[abs_i] = xbar
    avg_factor[abs_i] = avg_abs

  # Batches come absorber by absorber, so each absorber is finalized as soon
  # as its batches end
  current = None
  for abs_i, Reffs_abs, path_lengths in _iter_path_lengths(
    traj, sim_data, abs_Z):
    if current is None or current[0] != abs_i:
      if current is not None:
        finalize(*current)
      npaths = len(Reffs_abs)
      Reffs[abs_i] = Reffs_abs
      current = [abs_i, np.zeros(0, dtype = np.int64), np.zeros(0),
        np.zeros(npaths), 0]
    x = path_lengths - Reffs_abs[None, :, None]
    current[3] += x.sum(axis = (0, 2))
    current[4] += x.shape[0] * x.shape[2]
    bins = np.floor(x / bin_width).astype(np.int64)
    keys = (np.arange(npaths, dtype = np.int64)[None, :, None] * key_stride
      + bins + key_offset).ravel()
    batch_keys, batch_counts = np.unique(keys, return_counts = True)
    current[1], current[2] = _merge_hists(current[1], current[2],
      batch_keys, batch_counts.astype(np.float64))
  if current is not None:
    finalize(*current)

  return Reffs, deltar, avg_factor

def get_sim_spec(params, g, kstart, kstop, feffk,
  Reffs, degens, deltar, avg_factor,
  pha, amp, rep, lam, avg_spec = True):
  chi = get_spectra(feffk, Reffs, degens, deltar, avg_factor,
    pha, amp, rep, lam, params['ΔE0'], 0.0, 1.0, g.k)
  
  scalar = np.linalg.norm(g.k[kstart:kstop] ** 3 * g.chi[kstart:kstop]
    ) / np.linalg.norm(g.k[kstart:kstop] ** 3 * chi[kstart:kstop]) 
  chi *= scalar

  fitchi = autobk_lmfit(g.energy, g.mu, g.ek0, g.edge_step, 
    std_de = None, k_std = g.k, chi_std = chi)

  if not avg_spec:
    chi = get_spectra(feffk, Reffs, degens, deltar, avg_factor,
      pha, amp, rep, lam, params['ΔE0'], 0.0, 1.0, g.k, avg_spec = False)
    chi *= scalar

  return chi - fitchi + g.chi

def res_fun_lmfit(params, g, kstart, kstop, kw, feffk, Reffs, degens, 
  deltar, avg_factor, pha, amp, rep, lam):
  chi = get_sim_spec(params, g, kstart, kstop,
      feffk, Reffs, degens, deltar, avg_factor,
      pha, amp, rep, lam)
  expnorm = g.k[kstart:kstop] ** kw * g.chi[kstart:kstop]
  simnorm = g.k[kstart:kstop] ** kw * chi[kstart:kstop]
  dev = simnorm - expnorm
  return dev

def get_aligned_sim(sim, exp_g, traj, kmin_fit = 3.0, kmax_fit = 12.0, 
  kwfit = 3, kwout = 3, bin_width = 0.005):
  try:
    from larch.math import index_of
  except ImportError as e:
    raise ImportError('Error loading xraylarch') from e
  try:
    from lmfit import Parameters, Minimizer
  except ImportError as e:
    raise ImportError('Error loading lmfit') from e
  degens = [np.array([sim[i][j].degen for j in range(len(sim[i]))]
    ) for i in range(len(sim))]
  feffk = sim[0][0].k
  abs_Z = sim[0][0].abs_Z
  Reffs, deltar, avg_factor = get_path_averages_from_traj(traj, sim, abs_Z,
    bin_width = bin_width)
  amp = [np.array([path.amp for path in abs_data]) for abs_data in sim]
  pha = [np.array([path.pha for path in abs_data]) for abs_data in sim]
  lam = [np.array([path.lam for path in abs_data]) for abs_data in sim]
  rep = [np.array([path.rep for path in abs_data]) for abs_data in sim]

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
      feffk, Reffs, degens, deltar, avg_factor,
      pha, amp, rep, lam)
    chi2_grid[i] = np.dot(residual, residual)

  params['ΔE0'].set(value = dE0_grid[np.argmin(chi2_grid)])
  minner = Minimizer(res_fun_lmfit, params, fcn_args = [
    exp_g, kstart, kstop, kwfit, feffk, Reffs, degens, 
    deltar, avg_factor, pha, amp, rep, lam])
  result = minner.minimize()
  
  sim_chi = get_sim_spec(result.params, exp_g, kstart, kstop, 
    feffk, Reffs, degens, deltar, avg_factor,
    pha, amp, rep, lam, avg_spec = False)
  return sim_chi[:, kstart:kstop] * exp_g.k[None, kstart:kstop] ** kwout

@registry.register_simulation("EXAFS")
class EXAFS(BaseSimulation):
  def __init__(self, initial_structure, exp_g_filename, outdim, feff_location = "", 
    folder = "",  
    absorber = 'Co', edge = 'K', radius = 10.0, fit_kmin = 3.0, 
    fit_kmax = 12.0, time_limit = 240,
    additional_settings = {'EXAFS': 12.0, 'SCF': '4.5 0 30 .2 1'},
    sh_template = None, 
    sbatch_template = None, sbatch_group_template = None, 
    sbatch_md_template = None, sbatch_md_group_template = None,
    sbatch_opt_template = None,
    save_sim = True, kwfit = 3, kwout = 3,
    **kwargs) -> None:
    self.exp_g_filename = exp_g_filename
    self.outdim = outdim
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
    self.sbatch_md_template = sbatch_md_template
    self.sbatch_md_group_template = sbatch_md_group_template
    self.sbatch_opt_template = sbatch_opt_template
    self.sh_template = sh_template
    self.save_sim = save_sim
    self.time_limit = time_limit
    self.structure = None
    self.fit_kmin = fit_kmin
    self.fit_kmax = fit_kmax
    self.kwfit = kwfit
    self.kwout = kwout

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

    assert len(absorber_indices) > 0

    # guarantees at least two atoms of the absorber,
    # which is necessary because two different ipots are created
    structure.make_supercell(2)

    subfolders = [int(x) for x in os.listdir(self.parent_folder)]
    new_folder = os.path.join(self.parent_folder, str(np.max(
      subfolders) + 1 if len(subfolders) > 0 else 0))
    os.mkdir(new_folder)
    
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
    if self.sbatch_md_template is not None:
      with open(os.path.join(new_folder, 'structure.json'), 
        'w', encoding = 'utf-8') as md_struct_file:
        json.dump(struct.as_dict(), md_struct_file)

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
      new_job_file = os.path.join(new_folder, 'feff_job.sbatch')
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

    if self.sbatch_md_template is not None:
      with open(self.sbatch_md_template, 'r') as file:
        sbatch_data = file.read()
      job_name = int(time.time()) % 604800
      sbatch_data = sbatch_data.replace('##DIRECTORY##', new_folder)
      sbatch_data = sbatch_data.replace('##STRUCTURE_FILE##', 'structure.json')
      sbatch_data = sbatch_data.replace('##JOB_NAME##', f'MD_{job_name}')
      new_job_file = os.path.join(new_folder, 'md_job.sbatch')
      with open(new_job_file, 'w') as file:
        file.write(sbatch_data)
      try:
        subprocess.check_output(["sbatch", f"{new_job_file}"])
      except subprocess.CalledProcessError as e:
        print(e.output)

    self.group = group
    self.folder = new_folder
    self.params = params
    self.inds = absorber_indices 

  def check_done(self):
    if not os.path.isdir(self.folder):
      raise ASOSimulationException(f"Folder {self.folder} was deleted")

    abs_folders = [os.path.join(self.folder, str(i))
      for i in range(len(self.inds))]

    # Check every failure marker first, so a failure is reported even while
    # other jobs for this structure are still running
    if self.sbatch_md_template is not None and os.path.isfile(
      os.path.join(self.folder, "MD_FAILED")):
      raise ASOSimulationException(f"MD failed in {self.folder}")
    for abs_folder in abs_folders:
      if os.path.isfile(os.path.join(abs_folder, "FEFF_FAILED")):
        raise ASOSimulationException(f"FEFF failed in {abs_folder}")

    if not all(os.path.isfile(os.path.join(abs_folder, "DONE"))
      for abs_folder in abs_folders):
      return False
    if self.sbatch_md_template is not None and not os.path.isfile(
      os.path.join(self.folder, "MD_DONE")):
      return False

    return True

  def check_opt_done(self):
    if not os.path.isdir(self.folder):
      raise ASOSimulationException(f"Folder {self.folder} was deleted")

    return os.path.isfile(os.path.join(self.folder, "DONE"))

  def resolve(self):
    assert self.sbatch_opt_template is not None, "Need opt template for now"

    with open(self.sbatch_opt_template, 'r') as file:
      sbatch_opt_data = file.read()
    job_name = int(time.time()) % 604800
    sbatch_opt_data = sbatch_opt_data.replace('##JOB_INDEX##', self.folder)
    sbatch_opt_data = sbatch_opt_data.replace('##DIRECTORY##', self.folder)
    sbatch_opt_data = sbatch_opt_data.replace('##GROUP_FILE##', self.exp_g_filename)
    sbatch_opt_data = sbatch_opt_data.replace('##MD_FILE##', 'md.traj')
    sbatch_opt_data = sbatch_opt_data.replace('##KMIN_FIT##', str(self.fit_kmin))
    sbatch_opt_data = sbatch_opt_data.replace('##KMAX_FIT##', str(self.fit_kmax))
    sbatch_opt_data = sbatch_opt_data.replace('##KWFIT##', str(self.kwfit))
    sbatch_opt_data = sbatch_opt_data.replace('##KWOUT##', str(self.kwout))
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
    for i, absorb_ind in enumerate(self.inds):
      if not self.save_sim:
        shutil.rmtree(os.path.join(self.folder, str(i)))
    
    chi_ks = np.atleast_2d(np.genfromtxt(os.path.join(self.folder, 'chi_k.dat')))
    spectra = np.zeros((len(self.mask), chi_ks.shape[1]))
    spectra[np.array(self.mask)] = chi_ks

    return spectra 

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
