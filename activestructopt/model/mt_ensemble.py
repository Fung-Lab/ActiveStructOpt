from activestructopt.common.dataloader import prepare_data_pmg
from activestructopt.model.base import BaseModel, Runner, ConfigSetup
from activestructopt.dataset.kfolds import KFoldsDataset
from activestructopt.common.registry import registry
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import torch
from torch.func import stack_module_state, functional_call, vmap
import copy
from torch_geometric.loader import DataLoader
import mattertune.configs as MC
from mattertune import MatterTuner
import mattertune as mt
from pathlib import Path
from ase import Atoms

def hparams(data):
    hparams = MC.MatterTunerConfig.draft()

    # Model hparams
    hparams.model = MC.ORBBackboneConfig.draft()
    hparams.model.system = mt.backbones.orb.model.ORBSystemConfig(radius=10.0, max_num_neighbors=250)
    hparams.model.pretrained_model = "orb-v3-direct-inf-omat"
    #hparams.model.pretrained_model = "orb-v3-conservative-inf-omat"

    hparams.model.ignore_gpu_batch_transform_error = True
    hparams.model.freeze_backbone = False

    hparams.model.optimizer = MC.AdamWConfig(
        lr=1e-3,
        amsgrad=False,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.1,
        per_parameter_hparams=None,
    )
    hparams.model.lr_scheduler = MC.ReduceOnPlateauConfig(
        mode="min",
        monitor="val/total_loss",
        factor=0.5,
        patience=5,
        min_lr=0,
        threshold=1e-4,
    )
    hparams.model.reset_output_heads = True
    hparams.model.reset_backbone = False

    hparams.model.properties = []
    spectra = MC.AtomInvariantVectorPropertyConfig(name = 'spectra', dtype='float', size=900, loss=MC.MAELossConfig(),
                                                additional_head_settings={'num_layers': 1,
                                                                          'hidden_channels': 256})
    hparams.model.properties.append(spectra)

    # Data hparams
    hparams.data = data
    hparams.data.num_workers = 0

    hparams.trainer = MC.TrainerConfig.draft()
    hparams.trainer.max_epochs = 250
    hparams.trainer.accelerator = "gpu"
    hparams.trainer.check_val_every_n_epoch = 1
    hparams.trainer.gradient_clip_algorithm = "value"
    hparams.trainer.gradient_clip_val = 1.0
    #hparams.trainer.ema = MC.EMAConfig(decay=0.99)
    hparams.trainer.precision = "32"
    torch.set_float32_matmul_precision('high')

    hparams.trainer.early_stopping = MC.EarlyStoppingConfig(
        monitor='val/total_loss',
        patience=15,
        mode="min",
        min_delta=1e-8,
    )

    hparams.trainer.additional_trainer_kwargs = {
        "inference_mode": False,
    }

    hparams = hparams.finalize()
    #rich.print(hparams)
    return hparams

@registry.register_model("MTEnsemble")
class GNNEnsemble(BaseModel):
  def __init__(self, config, k = 5, **kwargs):
    self.k = k
    self.config = config
    self.ensemble = [None for _ in range(k)]
    self.scalar = 1.0
    self.device = 'cpu'
    self.updates = 0
  
  def train(self, dataset: KFoldsDataset, iterations = 500, lr = 0.001, 
    from_scratch = False, transfer = 1.0, prev_params = None, **kwargs):

    batch_size = 64

    trainval_indices = np.setxor1d(np.arange(len(dataset.structures)), 
        dataset.test_indices)
    for i in range(self.k):

      atoms_dataset = [Atoms(
          numbers=np.array([s.specie.Z for s in dataset.structures[j].sites]),
          positions=np.array([s.coords.tolist() for s in dataset.structures[j].sites]),
          cell=np.array(dataset.structures[j].lattice.matrix.tolist()),
          pbc=True,
          info={'spectra': dataset.ys[j]},
      ) for j in np.setxor1d(trainval_indices, dataset.kfolds[i])]

      val_set = [Atoms(
          numbers=np.array([s.specie.Z for s in dataset.structures[j].sites]),
          positions=np.array([s.coords.tolist() for s in dataset.structures[j].sites]),
          cell=np.array(dataset.structures[j].lattice.matrix.tolist()),
          pbc=True,
          info={'spectra': dataset.ys[j]},
      ) for j in dataset.kfolds[i]]

      train_split = len(atoms_dataset) / (len(atoms_dataset) + len(val_set))
      atoms_dataset.extend(val_set)


      dataset = mt.configs.AutoSplitDataModuleConfig(
          dataset = mt.configs.AtomsListDatasetConfig(atoms_list=atoms_dataset),
          train_split = train_split,
          batch_size = batch_size,
          shuffle=False,
      )

      hp = hparams(dataset)
      tune_output = MatterTuner(hp).tune()
      model, trainer = tune_output.model, tune_output.trainer

      lightning_kwargs = {'accelerator': 'gpu',
        'strategy': 'auto',
        'devices': 'auto',
        'num_nodes': 1,
        'precision': '32',
        'deterministic': None,
        'gradient_clip_val': 1.0,
        'gradient_clip_algorithm': 'value',
        'logger': None,
        'inference_mode': True,
        'fast_dev_run': False
      }
      property_predictor = model.property_predictor(lightning_kwargs)


    raise NotImplementedError

  def predict(self, structure, prepared = False, mask = None, **kwargs):
    raise NotImplementedError

  def set_scalar_calibration(self, dataset: KFoldsDataset):
    raise NotImplementedError

  def load(self, dataset: KFoldsDataset, params, scalar, **kwargs):
    raise NotImplementedError
