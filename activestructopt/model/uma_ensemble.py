from activestructopt.model.base import BaseModel
from activestructopt.dataset.kfolds import KFoldsDataset
from activestructopt.common.registry import registry
import numpy as np
import torch
import copy

# Pretrained backbones kept on the CPU, keyed by model name, so the checkpoint
# is read from the Hugging Face cache once per process rather than per training
_PRETRAINED_BACKBONES = {}

def _pretrained_backbone(model_name):
  """Returns a fresh copy of the pretrained UMA backbone (EMA weights), with
  the pretrained energy/force/stress heads discarded.

  fairchem is an optional dependency, and every file in this directory is
  imported by setup_imports, so its imports must stay lazy."""
  if model_name not in _PRETRAINED_BACKBONES:
    try:
      from fairchem.core.calculate.pretrained_mlip import (
        pretrained_checkpoint_path_from_name)
      from fairchem.core.units.mlip_unit.utils import load_inference_model
    except ImportError as e:
      raise ImportError('Error loading fairchem.core') from e

    overrides = {'backbone': {
      # The graph is built by batch_pos_cell so that the edge vectors stay
      # differentiable with respect to the positions and cell
      'otf_graph': False,
      'always_use_pbc': False,
      # Only the node embeddings are used, so no autograd forces or stress
      'regress_forces': False,
      'regress_stress': False,
      # The fast backends are inference only
      'execution_mode': 'general',
    }}
    model = load_inference_model(pretrained_checkpoint_path_from_name(
      model_name), overrides = overrides, use_ema = True,
      return_checkpoint = False)
    # use_ema wraps the model in an AveragedModel
    model = getattr(model, 'module', model)
    _PRETRAINED_BACKBONES[model_name] = model.backbone.cpu()
  return copy.deepcopy(_PRETRAINED_BACKBONES[model_name])

def _cpu_state_dict(module):
  return {key: value.detach().to('cpu', copy = True)
    for key, value in module.state_dict().items()}

class NodeVectorHead(torch.nn.Module):
  """MLP from the invariant (l = 0) node embedding to a per-atom vector, added
  to a fixed offset (e.g. the mean training target), so that the MLP only
  learns the deviation from it."""
  def __init__(self, in_channels, size, num_layers = 1, hidden_channels = 256):
    super().__init__()
    layers = []
    channels = in_channels
    for _ in range(num_layers):
      layers += [torch.nn.Linear(channels, hidden_channels), torch.nn.SiLU()]
      channels = hidden_channels
    layers.append(torch.nn.Linear(channels, size))
    self.mlp = torch.nn.Sequential(*layers)
    # A buffer, so it is saved with the state dict but not trained
    self.register_buffer('offset', torch.zeros(size))

  def forward(self, emb):
    return self.mlp(emb['node_embedding'][:, 0, :]) + self.offset

class UMABatch:
  """A batch of structures for the UMA backbone.

  Also exposes the attributes that the 'mt' data style of the constraints
  reads (see constraint/lennard_jones.py), so it can be used wherever the
  MatterTune ORB AtomGraphs batch is used."""
  def __init__(self, atomic_data, edge_vectors):
    self.atomic_data = atomic_data
    self.device = atomic_data.pos.device
    self.atomic_numbers = atomic_data.atomic_numbers
    self.senders = atomic_data.edge_index[0]
    self.receivers = atomic_data.edge_index[1]
    self.n_edge = atomic_data.nedges
    self.n_node = atomic_data.natoms
    self.edge_features = {'vectors': edge_vectors}

@registry.register_model("UMAEnsemble")
class UMAEnsemble(BaseModel):
  """An ensemble of UMA backbones, each fine-tuned on one of the k folds with
  a new per-atom output head of dimension output_dim. Training and prediction
  are done in memory, without the fairchem trainer and its files."""
  def __init__(self, config, k = 5, model_name = 'uma-s-1p2p1',
    task_name = 'omat', **kwargs):
    self.k = k
    self.config = config
    self.model_name = model_name
    self.task_name = task_name
    self.scalar = 1.0
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.sim_index = None
    self.backbones = []
    self.heads = []
    self.cutoff = None
    self.max_neighbors = None

  def train(self, dataset: KFoldsDataset, sim_index, iterations = 250,
    lr = 0.001, batch_size = 16, freeze_backbone = False,
    backbone_weight_decay = 0.0, head_weight_decay = 0.1,
    head_layers = 1, head_hidden_channels = 256, grad_clip = 1.0,
    lr_patience = 5, lr_factor = 0.5, early_stopping_patience = 15,
    seed = 0, merge_mole = True, subtract_mean = True, epoch_callback = None,
    **kwargs):
    """subtract_mean: whether each member's head predicts the deviation from
    the mean spectrum of the masked atoms of its training structures, rather
    than the spectrum itself, so that it starts from that mean.
    merge_mole: whether to merge the backbone's mixture of linear experts
    into plain linear layers for the composition of the dataset before
    fine-tuning (see _base_backbone).
    epoch_callback: optional function, for monitoring, called after each
    epoch's validation as epoch_callback(i, epoch, backbone, head, val_loss)
    for ensemble member i, with the backbone and head in eval mode."""
    self.sim_index = sim_index
    out_dim = self.config['dataset']['preprocess_params']['output_dim']
    mask = torch.tensor(dataset.simfuncs[sim_index].mask, dtype = torch.bool,
      device = self.device)

    trainval_indices = np.setxor1d(np.arange(len(dataset.structures)),
      dataset.test_indices)

    # Free the previous ensemble before building the new one
    self.backbones = []
    self.heads = []

    base_backbone = self._base_backbone(dataset, merge_mole)

    gen = torch.Generator().manual_seed(seed)
    for i in range(self.k):
      backbone = copy.deepcopy(base_backbone).to(self.device)
      head = NodeVectorHead(backbone.sphere_channels, out_dim,
        num_layers = head_layers,
        hidden_channels = head_hidden_channels).to(self.device)

      if freeze_backbone:
        backbone.requires_grad_(False)
      param_groups = [{'params': head.parameters(),
        'weight_decay': head_weight_decay}]
      if not freeze_backbone:
        # Weight decay pulls the weights toward zero rather than toward their
        # pretrained values, so it defaults to zero for the backbone
        param_groups.append({'params': backbone.parameters(),
          'weight_decay': backbone_weight_decay})
      optimizer = torch.optim.AdamW(param_groups, lr = lr,
        betas = (0.9, 0.95), eps = 1e-8)
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        mode = 'min', factor = lr_factor, patience = lr_patience,
        threshold = 1e-4)

      train_indices = np.setxor1d(trainval_indices, dataset.kfolds[i])
      val_indices = np.array(dataset.kfolds[i])
      train_batches = self._dataset_batches(dataset, train_indices)
      val_batches = self._dataset_batches(dataset, val_indices)
      if subtract_mean:
        # From this member's training structures only, not its validation fold
        head.offset.copy_(torch.cat([target[mask] for _, _, _, target
          in train_batches], dim = 0).mean(dim = 0))

      best_val = float('inf')
      best_state = None
      epochs_since_best = 0
      for epoch in range(iterations):
        backbone.train(not freeze_backbone)
        head.train()
        order = torch.randperm(len(train_indices), generator = gen).tolist()
        for start in range(0, len(order), batch_size):
          idx = order[start:(start + batch_size)]
          optimizer.zero_grad()
          loss = self._loss(backbone, head, train_batches, idx, mask)
          loss.backward()
          if grad_clip is not None:
            torch.nn.utils.clip_grad_value_(
              [p for g in param_groups for p in g['params']], grad_clip)
          optimizer.step()

        backbone.eval()
        head.eval()
        with torch.no_grad():
          val_loss = self._loss(backbone, head, val_batches,
            list(range(len(val_indices))), mask).item()
        scheduler.step(val_loss)
        if epoch_callback is not None:
          epoch_callback(i, epoch, backbone, head, val_loss)

        if val_loss < best_val - 1e-8:
          best_val = val_loss
          epochs_since_best = 0
          # Kept on the CPU so that the GPU does not hold a second model
          best_state = (_cpu_state_dict(backbone), _cpu_state_dict(head))
        else:
          epochs_since_best += 1
          if epochs_since_best >= early_stopping_patience:
            break

      backbone.load_state_dict(best_state[0])
      head.load_state_dict(best_state[1])
      backbone.eval()
      head.eval()
      backbone.requires_grad_(False)
      head.requires_grad_(False)
      # Kept on the CPU while the remaining members train
      self.backbones.append(backbone.cpu())
      self.heads.append(head.cpu())
      del (backbone, head, param_groups, optimizer, scheduler, best_state,
        train_batches, val_batches)
      if self.device == 'cuda':
        torch.cuda.empty_cache()

    del base_backbone
    self.backbones = [b.to(self.device) for b in self.backbones]
    self.heads = [h.to(self.device) for h in self.heads]

    gnn_mae, _, _ = self.set_scalar_calibration(dataset)

    return gnn_mae, [], [{} for _ in range(self.k)]

  def _base_backbone(self, dataset, merge_mole):
    """The pretrained backbone, on the CPU, that each member is fine-tuned
    from.

    The mixture of linear experts (MoLE) layers blend their experts with
    coefficients that depend only on the composition, charge, spin and task,
    and they do so separately for each structure in a batch. Every structure
    here has the same atoms (batch_pos_cell uses those of the starting
    structure), so with merge_mole the experts are blended once into plain
    linear layers. The predictions of the pretrained backbone are unchanged,
    while the parameters, their gradients and optimizer state, and the
    per-structure blended weights are reduced by about the number of experts.
    Fine-tuning then updates the blended weights rather than each expert."""
    base = _pretrained_backbone(self.model_name)
    self.cutoff = base.cutoff
    self.max_neighbors = base.max_neighbors
    if not merge_mole or getattr(base, 'num_experts', 0) == 0:
      return base

    # The blend also depends on the number of atoms for some models (UMA-S-1.2
    # includes a zero in its mean composition), so the atoms must match
    atoms = {tuple(sorted(s.atomic_numbers)) for s in dataset.structures}
    if len(atoms) > 1:
      raise ValueError('merge_mole requires every structure to have the same '
        'atoms, so train with merge_mole = False for this dataset')

    base = base.to(self.device).eval()
    data = self.batch_structures([dataset.structures[0]]).atomic_data
    with torch.no_grad():
      merged = base.merge_MOLE_model(data)
    del base, data
    return merged.cpu()

  def _dataset_batches(self, dataset, indices):
    """Positions, cells, atomic numbers and per-atom targets for the given
    structures, kept on the device for batching during training."""
    data = []
    for j in indices:
      s = dataset.structures[j]
      data.append((
        torch.tensor(s.cart_coords, dtype = torch.float32,
          device = self.device),
        torch.tensor(s.lattice.matrix, dtype = torch.float32,
          device = self.device),
        torch.tensor(s.atomic_numbers, dtype = torch.long,
          device = self.device),
        torch.tensor(np.array(dataset.ys[self.sim_index][j]),
          dtype = torch.float32, device = self.device),
      ))
    return data

  def _loss(self, backbone, head, data, idx, mask):
    """MAE between the predicted and target spectra of the masked atoms."""
    batch = self._build_batch([data[i][0] for i in idx],
      [data[i][1] for i in idx], [data[i][2] for i in idx])
    prediction = head(backbone(batch.atomic_data))
    target = torch.cat([data[i][3] for i in idx], dim = 0)
    atom_mask = mask.repeat(len(idx))
    return torch.mean(torch.abs(prediction[atom_mask] - target[atom_mask]))

  def batch_structures(self, structures):
    pos = [torch.tensor(s.cart_coords, dtype = torch.float32,
      device = self.device) for s in structures]
    cell = [torch.tensor(s.lattice.matrix, dtype = torch.float32,
      device = self.device) for s in structures]
    numbers = [torch.tensor(s.atomic_numbers, dtype = torch.long,
      device = self.device) for s in structures]
    return self._build_batch(pos, cell, numbers)

  def batch_pos_cell(self, pos, cell, starting_struct):
    numbers = torch.tensor(starting_struct.atomic_numbers, dtype = torch.long,
      device = self.device)
    return self._build_batch(pos, cell, [numbers] * len(pos))

  def _build_batch(self, pos, cell, numbers):
    """Builds a batched AtomicData with its radius graph. The graph
    connectivity is found without gradients, while the positions, and so the
    edge vectors that the backbone computes from them, keep their gradients
    with respect to the positions and cell."""
    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.graph.radius_graph_pbc import radius_graph_pbc_v2

    n = len(pos)
    natoms = torch.tensor([len(p) for p in pos], dtype = torch.long,
      device = self.device)
    batch = torch.repeat_interleave(torch.arange(n, device = self.device),
      natoms)

    # The radius graph assumes the atoms are within the unit cell. The floor
    # has no gradient, so the wrapped positions keep the gradient of pos.
    wrapped = []
    for p, c in zip(pos, cell):
      frac = torch.linalg.solve(c.T, p.T).T
      wrapped.append(p - torch.floor(frac).detach() @ c)
    pos_t = torch.cat(wrapped, dim = 0)
    cell_t = torch.stack(cell, dim = 0)

    with torch.no_grad():
      graph_input = AtomicData(
        pos = pos_t.detach(),
        atomic_numbers = torch.cat(numbers),
        cell = cell_t.detach(),
        pbc = torch.ones((n, 3), dtype = torch.bool, device = self.device),
        natoms = natoms,
        edge_index = torch.empty((2, 0), dtype = torch.long,
          device = self.device),
        cell_offsets = torch.empty((0, 3), dtype = torch.float32,
          device = self.device),
        nedges = torch.zeros(n, dtype = torch.long, device = self.device),
        charge = torch.zeros(n, dtype = torch.long, device = self.device),
        spin = torch.zeros(n, dtype = torch.long, device = self.device),
        fixed = torch.zeros(len(pos_t), dtype = torch.long,
          device = self.device),
        tags = torch.zeros(len(pos_t), dtype = torch.long,
          device = self.device),
        batch = batch,
        sid = [''] * n,
        dataset = [self.task_name] * n,
      )
      edge_index, cell_offsets, _ = radius_graph_pbc_v2(graph_input,
        self.cutoff, self.max_neighbors, False,
        pbc = graph_input.pbc)
      cell_offsets = cell_offsets.to(torch.float32)
      # The backbone expects the edges grouped by structure, in order
      edge_batch = batch[edge_index[1]]
      order = torch.argsort(edge_batch, stable = True)
      edge_index = edge_index[:, order]
      cell_offsets = cell_offsets[order]
      edge_batch = edge_batch[order]
      nedges = torch.bincount(edge_batch, minlength = n)

    # Same convention as the backbone with otf_graph = False
    shifts = torch.einsum('ij,ijk->ik', cell_offsets,
      cell_t[edge_batch])
    edge_vectors = pos_t[edge_index[0]] - pos_t[edge_index[1]] + shifts

    atomic_data = AtomicData(
      pos = pos_t,
      atomic_numbers = graph_input.atomic_numbers,
      cell = cell_t,
      pbc = graph_input.pbc,
      natoms = natoms,
      edge_index = edge_index,
      cell_offsets = cell_offsets,
      nedges = nedges,
      charge = graph_input.charge,
      spin = graph_input.spin,
      fixed = graph_input.fixed,
      tags = graph_input.tags,
      batch = batch,
      sid = [''] * n,
      dataset = [self.task_name] * n,
    )
    return UMABatch(atomic_data, edge_vectors)

  def predict(self, structure, prepared = False, mask = None, **kwargs):
    if not prepared:
      raise NotImplementedError

    atomic_data = structure.atomic_data
    nstructures = len(atomic_data.natoms)
    mask = torch.tensor(mask, dtype = torch.bool, device = self.device)

    # The MoLE backbones store their routing coefficients on the module
    # during the forward pass, so they are looped over rather than vmapped
    prediction = []
    for backbone, head in zip(self.backbones, self.heads):
      out = head(backbone(atomic_data))
      out = out.view(nstructures, -1, out.shape[-1])  # (nstructures, natoms, outdim)
      prediction.append(torch.mean(out[:, mask, :], dim = 1)) # node level masking
    prediction = torch.stack(prediction)  # (k, nstructures, outdim)

    mean = torch.mean(prediction, dim = 0)
    # last term to remove Bessel correction and match numpy behavior
    # https://github.com/pytorch/pytorch/issues/1082
    std = self.scalar * torch.std(prediction, dim = 0) * np.sqrt(
      (self.k - 1) / self.k)

    return torch.stack((mean, std))

  def set_scalar_calibration(self, dataset: KFoldsDataset):
    try:
      from scipy.stats import norm
      from scipy.optimize import minimize
    except ImportError as e:
      raise ImportError('Error loading scipy') from e
    self.scalar = 1.0

    test_data = self.batch_structures(
      [dataset.structures[i] for i in dataset.test_indices])
    test_targets = [dataset.ys[self.sim_index][i] for i in dataset.test_indices]

    with torch.inference_mode():
      test_res = self.predict(test_data, prepared = True,
        mask = dataset.simfuncs[self.sim_index].mask)
    aes = []
    zscores = []
    for i in range(len(test_targets)):
      target = np.mean(test_targets[i][np.array(
        dataset.simfuncs[self.sim_index].mask)], axis = 0)
      for j in range(len(target)):
        zscores.append((
          test_res[0][i][j].item() - target[j]) / test_res[1][i][j].item())
        aes.append(np.abs(test_res[0][i][j].item() - target[j]))
    zscores = np.sort(zscores)
    normdist = norm()
    f = lambda x: np.trapz(np.abs(np.cumsum(np.ones(len(zscores))) / len(
      zscores) - normdist.cdf(zscores / x[0])), normdist.cdf(zscores / x[0]))
    self.scalar = minimize(f, [1.0]).x[0]
    return np.mean(aes), normdist.cdf(np.sort(zscores) /
      self.scalar), np.cumsum(np.ones(len(zscores))) / len(zscores)

  def load(self, dataset: KFoldsDataset, params, scalar, **kwargs):
    raise NotImplementedError
