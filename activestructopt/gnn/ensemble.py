from matdeeplearn.common.trainer_context import new_trainer_context
from matdeeplearn.modules.scheduler import LRScheduler
from activestructopt.gnn.dataloader import prepare_data
import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import minimize
from torch_geometric import compile
import torch
from torch.func import stack_module_state, functional_call, vmap
import torch.optim as optim
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch import distributed as dist
import copy
import os
from torch_geometric.loader import DataLoader

class Runner:
  def __init__(self):
    self.config = None

  def __call__(self, config, args):
    with new_trainer_context(args = args, config = config) as ctx:
      self.config = ctx.config
      self.task = ctx.task
      self.trainer = ctx.trainer
      self.task.setup(self.trainer)
      #self.task.run()

  def checkpoint(self, *args, **kwargs):
    self.trainer.save(checkpoint_file="checkpoint.pt", training_state=True)
    self.config["checkpoint"] = self.task.chkpt_path
    self.config["timestamp_id"] = self.trainer.timestamp_id

class ConfigSetup:
  def __init__(self, run_mode):
    self.run_mode = run_mode
    self.seed = None
    self.submit = None

class Ensemble:
  def __init__(self, k, config):
    self.k = k
    self.config = config
    self.ensemble = [Runner() for _ in range(k)]
    self.scalar = 1.0
    self.loss_fn = getattr(F, config['optim']['loss']['loss_args']['loss_fn'])
    for i in range(self.k):
      self.ensemble[i](self.config, ConfigSetup('train'))
    base_model = copy.deepcopy(self.ensemble[0].trainer.model[0])
    self.base_model = base_model.to('meta')
    params, buffers = stack_module_state(
        [self.ensemble[j].trainer.model[0] for j in range(self.k)])
    self.params = params
    self.buffers = buffers
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", None)
      ) if self.config["task"]["parallel"] else 1
    if world_size > 1:
      self.config["optim"]["lr"] = self.config["optim"]["lr"] * world_size
    self.device = self.ensemble[0].trainer.rank

  def train(self, kfolds, trainval, trainval_targets):
    def fmodel(params, buffers, x):
      return functional_call(self.base_model, (params, buffers), 
        (x,))['output']
    try:
      kfolds_tensors = [torch.tensor(np.array(kfolds[i]), device = self.device
        ) for i in range(len(kfolds))]
      train_inds = [torch.cat([kfolds_tensors[i] for i in range(
        self.k) if i != j]) for j in range(self.k)]
      val_inds = [kfolds_tensors[j] for j in range(self.k)]

      start_epoch = int(self.ensemble[0].trainer.epoch)
      end_epoch = (
        self.ensemble[0].trainer.max_checkpoint_epochs + start_epoch
        if self.ensemble[0].trainer.max_checkpoint_epochs
        else self.ensemble[0].trainer.max_epochs
      )

      clip_grad_norm = self.ensemble[0].trainer.clip_grad_norm

      if str(self.device) not in ("cpu", "cuda"):
        dist.barrier()

      best_vals = [torch.inf for _ in range(self.k)]

      params = copy.deepcopy(self.params)
      buffers = copy.deepcopy(self.buffers)

      optimizers = [getattr(optim, 
        self.config["optim"]["optimizer"]["optimizer_type"])(
        list(params.values()) + list(buffers.values()),
        lr = self.config["optim"]["lr"],
        **self.config["optim"]["optimizer"].get("optimizer_args", {}),
      ) for _ in range(self.k)]


      scheduler = [LRScheduler(optimizers[j], 
        self.config["optim"]["scheduler"]["scheduler_type"], 
        self.config["optim"]["scheduler"]["scheduler_args"]) for j in range(
        self.k)]
      
     
      for epoch in range(start_epoch, end_epoch):
        # Based on https://github.com/Fung-Lab/MatDeepLearn_dev/blob/main/matdeeplearn/trainers/property_trainer.py
        # Start training over epochs loop
        epoch_start_time = time.time()

        self.base_model.train()

        out_lists = vmap(fmodel, in_dims = (0, 0, None), randomness = 'same')(
          params, buffers, next(iter(DataLoader(trainval, 
          batch_size = len(trainval)))))
        
        train_losses = [self.loss_fn(out_lists[j, train_inds[j], :], 
          trainval_targets[train_inds[j], :]) for j in range(self.k)]
        
        for j in range(self.k): # Compute backward 
          optimizers[j].zero_grad(set_to_none=True)
          train_losses[j].backward(retain_graph = True)
          if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(
              list(params.values()) + list(buffers.values()),
              max_norm = clip_grad_norm,
            )
          optimizers[j].step()

        for j in range(self.k):
          self.ensemble[j].trainer.epoch = epoch + 1

        if str(self.device) not in ("cpu", "cuda"):
          dist.barrier()

        self.base_model.eval()

        with torch.no_grad():
          out_lists = vmap(fmodel, in_dims = (0, 0, None), randomness = 'same')(
            params, buffers, next(iter(DataLoader(trainval, 
            batch_size = len(trainval)))))
          
          for j in range(self.k): # update prediction model if beats val losses
            if scheduler[j].scheduler_type == "ReduceLROnPlateau":
              scheduler[j].step(metrics = train_losses[j])
            else:
              scheduler[j].step()
            self.ensemble[j].trainer.epoch_time = time.time() - epoch_start_time
            vloss = self.loss_fn(out_lists[j, val_inds[j], :], 
              trainval_targets[val_inds[j], :]).item()
            if vloss < best_vals[j]:
              best_vals[j] = vloss
              for key in params.keys():
                self.params[key][j] = params[key][j]
              for key in buffers.keys():
                self.buffers[key][j] = buffers[key][j]
               
    except RuntimeError as e:
      self.ensemble[0].task._process_error(e)
      raise e
    
    torch.cuda.empty_cache()


  def predict(self, structure, prepared = False):
    def fmodel(params, buffers, x):
      return functional_call(self.base_model, (params, buffers), (x,))['output']
    data = structure if prepared else [prepare_data(
      structure, self.config['dataset']).to(self.device)]
    prediction = vmap(fmodel, in_dims = (0, 0, None))(
      self.params, self.buffers, next(iter(DataLoader(data, batch_size = len(data)))))

    mean = torch.mean(prediction, dim = 0)
    # last term to remove Bessel correction and match numpy behavior
    # https://github.com/pytorch/pytorch/issues/1082
    std = self.scalar * torch.std(prediction, dim = 0) * np.sqrt(
      (self.k - 1) / self.k)

    return torch.stack((mean, std))

  def set_scalar_calibration(self, test_data, test_targets):
    self.scalar = 1.0
    test_res = self.predict(test_data, prepared = True)
    zscores = []
    for i in range(len(test_targets)):
      for j in range(len(test_targets[0])):
        zscores.append((
          test_res[0][i][j].item() - test_targets[i][j]
          ) / test_res[1][i][j].item())
    zscores = np.sort(zscores)
    normdist = norm()
    f = lambda x: np.trapz(np.abs(np.cumsum(np.ones(len(zscores))) / len(
      zscores) - normdist.cdf(zscores / x[0])), normdist.cdf(zscores / x[0]))
    self.scalar = minimize(f, [1.0]).x[0]
    return normdist.cdf(np.sort(zscores) / self.scalar), np.cumsum(
      np.ones(len(zscores))) / len(zscores)
