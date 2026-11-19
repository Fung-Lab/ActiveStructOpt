from activestructopt.model.base import BaseModel
from activestructopt.dataset.kfolds import KFoldsDataset
from activestructopt.common.registry import registry
import numpy as np
import torch
from ase import Atoms
from pymatgen.core import Structure
from torch.func import stack_module_state, functional_call, vmap
from collections.abc import Sequence
from functools import cache
from types import SimpleNamespace
from typing import Literal
import copy

@cache
def _mt_extensions():
  """Adds node-level vector outputs to an unmodified MatterTune ORB backbone.

  MatterTune and orb_models are optional dependencies, and every file in this
  directory is imported by setup_imports, so these definitions must stay lazy.
  The result is cached because backbone_registry rejects duplicate names.
  """
  import nshconfig as C
  import mattertune as mt
  from mattertune.backbones.orb.model import (ORBBackboneConfig,
    ORBBackboneModule)
  from mattertune.finetune.properties import PropertyConfigBase
  from orb_models.forcefield import featurization_utilities as feat_util
  from orb_models.forcefield import segment_ops
  from orb_models.forcefield.nn_util import build_mlp

  # The in-place scatter_add_ in orb_models fails under vmap because it writes
  # batched values into an unbatched tensor of zeros.
  def scatter_sum(src, index, dim = -1, out = None, dim_size = None,
    reduce = "sum"):
    assert reduce == "sum"
    index = segment_ops._broadcast(index, src, dim)
    if out is None:
      size = list(src.size())
      if dim_size is not None:
        size[dim] = dim_size
      elif index.numel() == 0:
        size[dim] = 0
      else:
        size[dim] = int(index.max()) + 1
      out = torch.zeros(size, dtype = src.dtype, device = src.device)
    return out.scatter_add(dim, index, src)

  segment_ops.scatter_sum = scatter_sum

  # orb_models raises an error for cutoffs above 6 unless N is larger than 4.
  find_translations = feat_util.find_minimal_supercell_translations

  def find_minimal_supercell_translations(cell, cutoff, N = None):
    if N is None:
      N = 4 if cutoff <= 6.0 else 5
    return find_translations(cell, cutoff, N = N)

  feat_util.find_minimal_supercell_translations = (
    find_minimal_supercell_translations)

  class NodeVectorPropertyConfig(PropertyConfigBase):
    type: Literal["node_vector"] = "node_vector"
    size: C.PositiveInt
    mask: list[bool]
    num_layers: int = 1
    hidden_channels: int = 256

    def from_ase_atoms(self, atoms):
      return atoms.info[self.name]

    def ase_calculator_property_name(self):
      return None

    def property_type(self):
      return "atom"

  class NodeVectorHead(torch.nn.Module):
    def __init__(self, size, num_layers, hidden_channels):
      super().__init__()
      self.mlp = build_mlp(
        input_size = 256,
        hidden_layer_sizes = [hidden_channels] * num_layers,
        output_size = size,
        activation = 'silu',
        dropout = None,
        checkpoint = None,
      )

    def forward(self, node_features, batch):
      return self.mlp(node_features)

  class ORBNodeVectorBackboneModule(ORBBackboneModule):
    @classmethod
    def hparams_cls(cls):
      return ORBNodeVectorBackboneConfig

    def _create_output_head(self, prop, pretrained_model):
      return NodeVectorHead(prop.size, prop.num_layers, prop.hidden_channels)

    def _compute_loss(self, predictions, labels, log = True, log_prefix = ""):
      masked_predictions = {}
      masked_labels = {}
      for prop in self.hparams.properties:
        prediction = predictions[prop.name]
        mask = torch.tensor(prop.mask, dtype = torch.bool,
          device = prediction.device).repeat(
          prediction.shape[0] // len(prop.mask))
        masked_predictions[prop.name] = prediction[mask]
        masked_labels[prop.name] = labels[prop.name][mask]
      return super()._compute_loss(masked_predictions, masked_labels,
        log = log, log_prefix = log_prefix)

  @mt.backbone_registry.register
  class ORBNodeVectorBackboneConfig(ORBBackboneConfig):
    name: Literal["orb_node_vector"] = "orb_node_vector"
    properties: Sequence[NodeVectorPropertyConfig]

    def create_model(self):
      return ORBNodeVectorBackboneModule(self)

  return SimpleNamespace(
    NodeVectorPropertyConfig = NodeVectorPropertyConfig,
    ORBNodeVectorBackboneConfig = ORBNodeVectorBackboneConfig,
  )

def torch_cell_to_cellpar(cell):
  cellpar = torch.zeros(6, dtype = cell.dtype, device = cell.device)
  cellpar[:3] += torch.norm(cell, dim =  1)
  for i in range(3):
    j = i - 1
    k = i - 2
    ll = cellpar[j % 3] * cellpar[k % 3]
    if ll > 1e-16:
      x = torch.dot(cell[j % 3], cell[k % 3]) / ll
      cellpar[3+i] = 180.0 / np.pi * torch.acos(x)
    else:
      cellpar[3+i] = 90.0
  return cellpar

def hparams(data, num_epochs, out_dim, mask, start_lr = 0.001,
  radius = 10.0, max_num_neighbors = 250, pretrained = True):

  import mattertune.configs as MC
  import mattertune as mt

  extensions = _mt_extensions()

  hparams = MC.MatterTunerConfig.draft()

  # Model hparams
  hparams.model = extensions.ORBNodeVectorBackboneConfig.draft()
  hparams.model.system = mt.backbones.orb.model.ORBSystemConfig(
    radius = radius, max_num_neighbors = max_num_neighbors)
  hparams.model.pretrained_model = "orb-v3-direct-inf-omat"
  #hparams.model.pretrained_model = "orb-v3-conservative-inf-omat"

  hparams.model.ignore_gpu_batch_transform_error = True
  hparams.model.freeze_backbone = not pretrained

  hparams.model.optimizer = MC.AdamWConfig(
    lr = start_lr,
    amsgrad = False,
    betas = (0.9, 0.95),
    eps = 1.0e-8,
    weight_decay = 0.1,
    per_parameter_hparams = None,
  )
  hparams.model.lr_scheduler = MC.ReduceOnPlateauConfig(
    mode = "min",
    monitor = "val/total_loss",
    factor = 0.5,
    patience = 5,
    min_lr = 0,
    threshold = 1e-4,
  )
  hparams.model.reset_output_heads = True
  hparams.model.reset_backbone = False

  hparams.model.properties = []
  spectra = extensions.NodeVectorPropertyConfig(name = 'spectra',
    dtype = 'float', size = out_dim, loss = MC.MAELossConfig(),
    mask = [bool(m) for m in mask], num_layers = 1, hidden_channels = 256)
  hparams.model.properties.append(spectra)

  # Data hparams
  hparams.data = data
  hparams.data.num_workers = 0

  hparams.trainer = MC.TrainerConfig.draft()
  hparams.trainer.max_epochs = num_epochs
  hparams.trainer.accelerator = "gpu"
  hparams.trainer.check_val_every_n_epoch = 1
  hparams.trainer.gradient_clip_algorithm = "value"
  hparams.trainer.gradient_clip_val = 1.0
  #hparams.trainer.ema = MC.EMAConfig(decay=0.99)
  hparams.trainer.precision = "32"
  torch.set_float32_matmul_precision('high')

  hparams.trainer.early_stopping = MC.EarlyStoppingConfig(
    monitor = 'val/total_loss',
    patience = 15,
    mode = "min",
    min_delta = 1e-8,
  )

  hparams.trainer.additional_trainer_kwargs = {
    "inference_mode": False,
    "enable_checkpointing": False,
    "logger": False,
  }

  hparams = hparams.finalize()
  return hparams

@registry.register_model("MTEnsemble")
class MTEnsemble(BaseModel):
  def __init__(self, config, k = 5, **kwargs):
    self.k = k
    self.config = config
    self.scalar = 1.0
    self.device = 'cpu'
    self.updates = 0
    self.hp = None
    self.sim_index = None
  
  def train(self, dataset: KFoldsDataset, sim_index, iterations = 250, lr = 0.001, 
    from_scratch = False, transfer = 1.0, prev_params = None, radius = 10.0, 
    max_num_neighbors = 250, batch_size = 64, pretrained = True, **kwargs):
    import mattertune as mt
    from mattertune import MatterTuner

    self.sim_index = sim_index

    trainval_indices = np.setxor1d(np.arange(len(dataset.structures)), 
        dataset.test_indices)

    backbones = [None for _ in range(self.k)]
    output_heads = [None for _ in range(self.k)]

    for i in range(self.k):
      train_set = [Atoms(
          numbers = np.array([s.specie.Z for s in dataset.structures[j].sites]),
          positions = np.array([s.coords.tolist(
            ) for s in dataset.structures[j].sites]),
          cell = np.array(dataset.structures[j].lattice.matrix.tolist()),
          pbc = True,
          info = {'spectra': dataset.ys[sim_index][j]},
      ) for j in np.setxor1d(trainval_indices, dataset.kfolds[i])]

      val_set = [Atoms(
          numbers = np.array([s.specie.Z for s in dataset.structures[j].sites]),
          positions = np.array([s.coords.tolist(
            ) for s in dataset.structures[j].sites]),
          cell = np.array(dataset.structures[j].lattice.matrix.tolist()),
          pbc = True,
          info = {'spectra': dataset.ys[sim_index][j]},
      ) for j in dataset.kfolds[i]]

      # AtomsListDataset shuffles its atoms, so the folds are passed separately
      mt_dataset = mt.configs.ManualSplitDataModuleConfig(
        train = mt.configs.AtomsListDatasetConfig(atoms_list = train_set),
        validation = mt.configs.AtomsListDatasetConfig(atoms_list = val_set),
        batch_size = batch_size,
      )

      self.hp = hparams(mt_dataset, iterations,
        self.config['dataset']['preprocess_params']['output_dim'],
        dataset.simfuncs[sim_index].mask, lr,
        radius = radius, max_num_neighbors = max_num_neighbors,
        pretrained = pretrained)

      # That shuffle uses the global NumPy RNG, which the samplers also use
      rng_state = np.random.get_state()
      tune_output = MatterTuner(self.hp).tune()
      np.random.set_state(rng_state)

      model = tune_output.model.to('cuda')
      self.device = model.device
      backbones[i] = model.backbone
      output_heads[i] = model.output_heads['spectra']
      self.collate_fn = model.collate_fn

    self.backbone_params, self.backbone_buffers = stack_module_state(backbones)
    self.backbone_base = copy.deepcopy(backbones[0]).to('meta')
    self.output_params, self.output_buffers = stack_module_state(output_heads)
    self.output_head_base = copy.deepcopy(output_heads[0]).to('meta')
    
    gnn_mae, _, _ = self.set_scalar_calibration(dataset)

    return gnn_mae, [], [{} for _ in range(self.k)]
    

  def batch_structures(self, structures):
    pos = [torch.Tensor([site.coords.tolist() for site in struct.sites]).to(
      self.device) for struct in structures]
    cell = [torch.Tensor(struct.lattice.matrix.tolist()).to(
      self.device) for struct in structures]
    return self.batch_pos_cell(pos, cell, structures[0])

  def batch_pos_cell(self, pos, cell, starting_struct):
    atom_graphs = [self.pos_cell_to_atom_graphs(pos[i], cell[i], 
      starting_struct) for i in range(len(pos))]
    return self.collate_fn(atom_graphs)

  def pos_cell_to_atom_graphs(self, positions, cell, starting_struct: Structure, 
      wrap = True, edge_method = None, half_supercell = None,):
    from orb_models.forcefield import featurization_utilities as feat_util
    from orb_models.forcefield.base import AtomGraphs
    output_dtype = torch.get_default_dtype()
    graph_construction_dtype = torch.get_default_dtype()
    max_num_neighbors = self.hp.model.system.max_num_neighbors
    atomic_numbers = torch.from_numpy(np.array(starting_struct.atomic_numbers)
      ).to(torch.long)
    atomic_numbers_embedding = torch.nn.functional.one_hot(
        atomic_numbers, num_classes=118).to(torch.get_default_dtype())
    pbc = torch.from_numpy(np.array([True, True, True])).to(self.device)

    lattice = torch_cell_to_cellpar(cell)
    if wrap and (torch.any(cell != 0) and torch.any(pbc)):
      positions = feat_util.map_to_pbc_cell(positions, cell)

    edge_index, edge_vectors, unit_shifts = feat_util.compute_pbc_radius_graph(
      positions = positions,
      cell = cell,
      pbc = pbc,
      radius = self.hp.model.system.radius,
      max_number_neighbors = max_num_neighbors,
      edge_method = edge_method,
      half_supercell = half_supercell,
      float_dtype = graph_construction_dtype,
      device = self.device,
    )
    senders, receivers = edge_index[0], edge_index[1]

    node_feats = {
      # NOTE: positions are stored as features on the AtomGraphs,
      # but not actually used as input features to the model.
      "positions": positions,
      "atomic_numbers": atomic_numbers.to(torch.long),
      "atomic_numbers_embedding": atomic_numbers_embedding,
      "atom_identity": torch.arange(len(starting_struct)).to(torch.long),
    }
    edge_feats = {
      "vectors": edge_vectors,
      "unit_shifts": unit_shifts,
    }
    graph_feats = {
      "cell": cell.unsqueeze(0), # Add a batch dimension to non-scalar graph features/targets
      "pbc": pbc.unsqueeze(0),
      "lattice": lattice.unsqueeze(0),
    }

    atom_graphs = AtomGraphs(
      senders = senders,
      receivers = receivers,
      n_node = torch.tensor([len(positions)]),
      n_edge = torch.tensor([len(senders)]),
      node_features = node_feats,
      edge_features = edge_feats,
      system_features = graph_feats,
      node_targets = {},
      edge_targets = {},
      system_targets = {},
      fix_atoms = None,
      tags = torch.zeros(len(starting_struct)),
      radius = self.hp.model.system.radius,
      max_num_neighbors = torch.tensor([max_num_neighbors]),
      system_id = None,
    ).to(device = self.device, dtype = output_dtype)

    atom_types_onehot = torch.nn.functional.one_hot(
      atom_graphs.atomic_numbers.view(-1).long(), num_classes = 120)
    # ^ (n_atoms, 120)
    # Now we need to sum this up to get the composition vector
    composition = atom_types_onehot.sum(dim = 0, keepdim = True)
    # ^ (1, 120)
    atom_graphs.system_features["norm_composition"] = composition

    return atom_graphs
  
  def predict(self, structure, prepared = False, mask = None, **kwargs):
    if prepared:
      def fbackbone(params, buffers, x):
        return functional_call(self.backbone_base, (params, buffers), (x,)
          )['node_features']
      def foutput(params, buffers, node_features, x):
        return functional_call(self.output_head_base, (params, buffers), 
          (node_features, x))

      bb_prediction = vmap(fbackbone, in_dims = (0, 0, None))(
        self.backbone_params, self.backbone_buffers, structure)
      out_prediction = vmap(foutput, in_dims = (0, 0, 0, None))(
        self.output_params, self.output_buffers, bb_prediction, structure)

      res = torch.transpose(torch.stack(torch.split(out_prediction, 
        len(mask), dim = 1)), 0, 1)  # (k, nstructures, natoms, outdim)
      prediction = torch.mean(res[:, :, torch.tensor(mask, dtype = torch.bool), 
        :], dim = 2) # node level masking

      mean = torch.mean(prediction, dim = 0)
      # last term to remove Bessel correction and match numpy behavior
      # https://github.com/pytorch/pytorch/issues/1082
      std = self.scalar * torch.std(prediction, dim = 0) * np.sqrt(
        (self.k - 1) / self.k)

      return torch.stack((mean, std))
      
    else:
      raise NotImplementedError

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
      target = np.mean(test_targets[i][np.array(dataset.simfuncs[self.sim_index].mask)], 
        axis = 0)
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
