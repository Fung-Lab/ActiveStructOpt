"""The Falk-Langer non-affine displacement D^2_min of an MD trajectory
relative to its starting structure, as a check that the MD only vibrates
about the starting structure rather than rearranging into a different one.

For atom i with neighbors N_i (the atoms within cutoff of it in the starting
structure),

  D^2_min(i) = min_J (1 / |N_i|) sum_{j in N_i} |d_ij - J d_ij(0)|^2

where d_ij are the separation vectors (followed continuously from the
starting structure) and J is the best local affine map. Since the residual of
the least-squares fit is P_i (u_j - u_i), with u the displacements from the
starting structure and P_i the projector onto the complement of the column
space of the starting separations, D^2_min only depends on the relative
displacements of the neighbors.

D^2_min is computed on block-averaged displacements, which removes most of
the thermal motion: at 300 K the instantaneous D^2_min of the experimental
Ti oxide references is ~0.02 Å^2 (99th percentile ~0.1 Å^2), but at most
0.07 Å^2 on 0.5-1 ps block averages. The defaults reject a block if more than
max_frac of the atoms have D^2_min above level, which catches rearrangements
of the whole structure, or if any atom has D^2_min above max_atom, which
catches local rearrangements that a fraction averages away. On 21 UMA MD runs
of experimental Ti compounds at 300 K, the worst well-behaved structure
(Fe2TiO5, which relaxes by ~0.17 Å RMS to the UMA minimum) has 27% of its
atoms above 0.05 Å^2 and a largest D^2_min of 0.07 Å^2.
"""
import numpy as np

class D2minMonitor:
  """Checks block-averaged D^2_min relative to the structure atoms has when
  the monitor is created. Call update(atoms) at every sampled MD step; every
  block_frames calls, the displacements averaged over the block are checked,
  and update returns a description of the failure, or None.

  Diagnostics: nblocks (blocks checked), last_frac and last_max (the fraction
  of atoms above level and the largest D^2_min of the last block), and
  max_frac_seen and max_atom_seen (the largest of each over all blocks)."""
  def __init__(self, atoms, block_frames, cutoff = 3.5, level = 0.05,
    max_frac = 0.5, max_atom = 0.5):
    if block_frames < 1:
      raise ValueError('block_frames must be at least 1')
    self.block_frames = int(block_frames)
    self.level = level
    self.max_frac = max_frac
    self.max_atom = max_atom
    self.cell = atoms.cell.array.copy()
    self.inv_cell = np.linalg.inv(self.cell)
    self.frac0 = atoms.get_scaled_positions(wrap = False).copy()
    cart0 = self.frac0 @ self.cell
    n = len(atoms)

    # Neighbors in the starting structure, and the projectors P_i
    self.neighbors = []
    self.projectors = []
    for i in range(n):
      d = self._mic(cart0 - cart0[i])
      r = np.linalg.norm(d, axis = 1)
      j = np.where((r > 1e-8) & (r < cutoff))[0]
      self.neighbors.append(j)
      if len(j) == 0:
        self.projectors.append(None)
        continue
      A = d[j]
      self.projectors.append(np.eye(len(j)) - A @ np.linalg.pinv(A))

    self._sum = np.zeros((n, 3))
    self._count = 0
    self.nblocks = 0
    self.last_frac = None
    self.last_max = None
    self.max_frac_seen = 0.0
    self.max_atom_seen = 0.0

  def _mic(self, d):
    f = d @ self.inv_cell
    f -= np.round(f)
    return f @ self.cell

  def displacements(self, atoms):
    """Displacements (natoms, 3) from the starting structure, as minimum
    image vectors."""
    df = atoms.get_scaled_positions(wrap = False) - self.frac0
    df -= np.round(df)
    return df @ self.cell

  def d2min(self, u):
    """D^2_min (natoms,) of displacements u (natoms, 3) from the starting
    structure."""
    out = np.zeros(len(u))
    for i, (j, P) in enumerate(zip(self.neighbors, self.projectors)):
      if P is None:
        continue
      res = P @ (u[j] - u[i])
      out[i] = np.sum(res ** 2) / len(j)
    return out

  def update(self, atoms):
    self._sum += self.displacements(atoms)
    self._count += 1
    if self._count < self.block_frames:
      return None
    d2 = self.d2min(self._sum / self._count)
    self._sum[:] = 0.0
    self._count = 0
    self.nblocks += 1
    self.last_frac = float(np.mean(d2 > self.level))
    self.last_max = float(d2.max())
    self.max_frac_seen = max(self.max_frac_seen, self.last_frac)
    self.max_atom_seen = max(self.max_atom_seen, self.last_max)
    if self.last_frac > self.max_frac:
      return (f'{100 * self.last_frac:.0f}% of atoms have block-averaged '
        f'D2min above {self.level} Å^2 (limit {100 * self.max_frac:.0f}%) '
        f'in block {self.nblocks}')
    if self.last_max > self.max_atom:
      return (f'an atom has block-averaged D2min {self.last_max:.3f} Å^2 '
        f'(limit {self.max_atom} Å^2) in block {self.nblocks}')
    return None

def block_frames(block_ps, timestep_fs, interval_steps):
  """Number of sampled steps (every interval_steps MD steps of timestep_fs)
  in a block of block_ps picoseconds."""
  return max(1, int(round(1000 * block_ps / (timestep_fs * interval_steps))))
