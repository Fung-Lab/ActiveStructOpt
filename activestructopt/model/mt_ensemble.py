from activestructopt.model.base import BaseModel
from activestructopt.dataset.kfolds import KFoldsDataset
from activestructopt.common.registry import registry
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import torch
from ase import Atoms
from pymatgen.core import Structure
from torch.func import stack_module_state, functional_call, vmap
import copy

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

def hparams(data, num_epochs, out_dim, start_lr = 0.001, radius = 10.0, 
  max_num_neighbors = 250, pretrained = True):
  import mattertune.configs as MC
  import mattertune as mt

  hparams = MC.MatterTunerConfig.draft()

  # Model hparams
  hparams.model = MC.ORBBackboneConfig.draft()
  hparams.model.system = mt.backbones.orb.model.ORBSystemConfig(
    radius = radius, max_num_neighbors = max_num_neighbors)
  hparams.model.pretrained_model = "orb-v3-direct-inf-omat"
  #hparams.model.pretrained_model = "orb-v3-conservative-inf-omat"

  hparams.model.ignore_gpu_batch_transform_error = True
  hparams.model.freeze_backbone = False

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
  hparams.model.reset_backbone = not pretrained

  hparams.model.properties = []
  spectra = MC.AtomInvariantVectorPropertyConfig(name = 'spectra', 
    dtype = 'float', size = out_dim, loss = MC.MAELossConfig(),
    additional_head_settings = {'num_layers': 1, 'hidden_channels': 256})
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
  
  def train(self, dataset: KFoldsDataset, iterations = 250, lr = 0.001, 
    from_scratch = False, transfer = 1.0, prev_params = None, radius = 10.0, 
    max_num_neighbors = 250, batch_size = 64, pretrained = True, **kwargs):
    import mattertune as mt
    from mattertune import MatterTuner

    trainval_indices = np.setxor1d(np.arange(len(dataset.structures)), 
        dataset.test_indices)

    backbones = [None for _ in range(self.k)]
    output_heads = [None for _ in range(self.k)]

    for i in range(self.k):
      atoms_dataset = [Atoms(
          numbers = np.array([s.specie.Z for s in dataset.structures[j].sites]),
          positions = np.array([s.coords.tolist(
            ) for s in dataset.structures[j].sites]),
          cell = np.array(dataset.structures[j].lattice.matrix.tolist()),
          pbc = True,
          info = {'spectra': dataset.ys[j]},
      ) for j in np.setxor1d(trainval_indices, dataset.kfolds[i])]

      val_set = [Atoms(
          numbers = np.array([s.specie.Z for s in dataset.structures[j].sites]),
          positions = np.array([s.coords.tolist(
            ) for s in dataset.structures[j].sites]),
          cell = np.array(dataset.structures[j].lattice.matrix.tolist()),
          pbc = True,
          info = {'spectra': dataset.ys[j]},
      ) for j in dataset.kfolds[i]]

      train_split = len(atoms_dataset) / (len(atoms_dataset) + len(val_set))
      atoms_dataset.extend(val_set)

      mt_dataset = mt.configs.AutoSplitDataModuleConfig(
          dataset = mt.configs.AtomsListDatasetConfig(atoms_list=atoms_dataset),
          train_split = train_split,
          batch_size = batch_size,
          shuffle=False,
      )

      self.hp = hparams(mt_dataset, iterations, 
        self.config['dataset']['preprocess_params']['output_dim'], lr, 
        radius = radius, max_num_neighbors = max_num_neighbors, 
        pretrained = pretrained)
      tune_output = MatterTuner(self.hp).tune()
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
    self.scalar = 1.0
    
    test_data = self.batch_structures(
      [dataset.structures[i] for i in dataset.test_indices])
    test_targets = [dataset.ys[i] for i in dataset.test_indices]

    with torch.inference_mode():
      test_res = self.predict(test_data, prepared = True, 
        mask = dataset.simfunc.mask)
    aes = []
    zscores = []
    for i in range(len(test_targets)):
      target = np.mean(test_targets[i][np.array(dataset.simfunc.mask)], 
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
