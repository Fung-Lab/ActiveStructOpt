from matdeeplearn.common.trainer_context import new_trainer_context
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
    self.device = 'cpu'
    self.loss_fn = getattr(F, config['optim']['loss']['loss_args']['loss_fn'])
    for i in range(self.k):
      self.ensemble[i](self.config, ConfigSetup('train'))
    base_model = copy.deepcopy(self.ensemble[0].trainer.model[0])
    self.base_model = base_model.to('meta')
    rank = self.ensemble[0].trainer.rank
  
  def train(self, kfolds, trainval, trainval_targets):
    def fmodel(params, buffers, x):
      return functional_call(self.base_model, (params, buffers), 
        (x,))['output']
    try:
      kfolds_tensors = [torch.tensor(kfolds[i], device = self.device) for i in range(len(kfolds))]
      
      params, buffers = stack_module_state(
          [self.ensemble[j].trainer.model[0] for j in range(self.k)])

      world_size = int(os.environ.get("LOCAL_WORLD_SIZE", None)
        ) if self.config["task"]["parallel"] else 1

      if world_size > 1:
        self.config["optim"]["lr"] = self.config["optim"]["lr"] * world_size

      optimizer = getattr(optim, 
        self.config["optim"]["optimizer"]["optimizer_type"])(
        params.values() + buffers.values(),
        lr = self.config["optim"]["lr"],
        **self.config["optim"]["optimizer"].get("optimizer_args", {}),
      )

      start_epoch = int(self.ensemble[0].trainer.epoch)
      end_epoch = (
        self.ensemble[0].trainer.max_checkpoint_epochs + start_epoch
        if self.ensemble[0].trainer.max_checkpoint_epochs
        else self.ensemble[0].trainer.max_epochs
      )

      rank = self.ensemble[0].trainer.rank
      model_save_frequency = self.ensemble[0].trainer.model_save_frequency
      train_verbosity = self.ensemble[0].trainer.train_verbosity
      output_frequency = self.ensemble[0].trainer.output_frequency
      write_output = self.ensemble[0].trainer.write_output
      use_amp = self.ensemble[0].trainer.use_amp

      if str(rank) not in ("cpu", "cuda"):
        dist.barrier()

      for epoch in range(start_epoch, end_epoch):
        # Based on https://github.com/Fung-Lab/MatDeepLearn_dev/blob/main/matdeeplearn/trainers/property_trainer.py
        # Start training over epochs loop
        epoch_start_time = time.time()
        _metrics = [{} for _ in range(self.k)] # metrics for every epoch

        for j in range(self.k):
          self.ensemble[j].trainer.model[0].train()

        #with autocast(enabled = use_amp): # Compute forward  
        out_lists = vmap(fmodel, in_dims = (0, 0, None), randomness = 'same')(
          params, buffers, next(iter(DataLoader(trainval, 
          batch_size = len(trainval)))))
        train_inds = [torch.cat([kfolds_tensors[j] for i in range(
          self.k) if i != j]) for j in range(self.k)]
        loss = torch.mean(out_lists)
        #losses = [self.loss_fn(out_lists[j, :, :], 
        #  trainval_targets[:, :]) for j in range(self.k)]
        #print(losses)
        
        for j in range(self.k): # Compute backward 
          print(j)
          optimizer.zero_grad(set_to_none=True)
          loss.backward(retain_graph = True)
          #print(params.grad)
          #self.ensemble[j].trainer.scaler.scale(losses[j]).backward()
          if self.ensemble[j].trainer.clip_grad_norm:
            grad_norm = torch.nn.utils.clip_grad_norm_(
              params,
              max_norm=self.ensemble[j].trainer.clip_grad_norm,
            )
            print(grad_norm)
          optimizer.step()
          #self.ensemble[j].trainer.scaler.step(self.ensemble[j].trainer.optimizer[0])
          #self.ensemble[j].trainer.scaler.update()

        for j in range(self.k): # Compute metrics
          _metrics[j] = self.ensemble[j].trainer._compute_metrics(
            out_lists[j][0], batches[j][0], _metrics[j])
          self.ensemble[j].trainer.metrics[0] = self.ensemble[
            j].trainer.evaluator.update("loss", losses[j].item(), 
            out_lists[j][0]["output"].shape[0], _metrics[j])

        for j in range(self.k):
          self.ensemble[j].trainer.epoch = epoch + 1

        if str(rank) not in ("cpu", "cuda"):
          dist.barrier()

        # Save current model
        if model_save_frequency == 1:
          for j in range(self.k):
            self.ensemble[j].trainer.save_model(
              checkpoint_file = "checkpoint.pt", training_state = True)

        # Evaluate on validation set if it exists
        metrics = [self.ensemble[j].trainer.validate(
          "val") for j in range(self.k)]

        for j in range(self.k): # Train loop timings and log metrics
          self.ensemble[j].trainer.epoch_time = time.time() - epoch_start_time
          if epoch % train_verbosity == 0:
            self.ensemble[j].trainer._log_metrics(metrics[j])

          # Update best val metric and model, and save best model and predicted outputs
          if metrics[j][0][type(self.ensemble[j].trainer.loss_fn).__name__][
            "metric"] < self.ensemble[j].trainer.best_metric[0]:
            if output_frequency == 0:
              self.ensemble[j].trainer.update_best_model(metrics[j][0], 0, 
                write_model = model_save_frequency == 1, write_csv = False)
            elif output_frequency == 1:
              self.ensemble[j].trainer.update_best_model(metrics[j][0], 0, 
                write_model = model_save_frequency == 1, write_csv = True)
            
        for j in range(self.k):
          self.ensemble[j].trainer._scheduler_step()       
        
        for j in range(self.k):
          if self.ensemble[j].trainer.best_model_state:
            if str(self.ensemble[j].trainer.rank) in "0":
              self.ensemble[j].trainer.model[0].module.load_state_dict(self.ensemble[j].trainer.best_model_state[0])
            elif str(self.ensemble[j].trainer.rank) in ("cpu", "cuda"):
              self.ensemble[j].trainer.model[0].load_state_dict(self.ensemble[j].trainer.best_model_state[0])

            if model_save_frequency != -1:
              self.ensemble[j].trainer.save_model("best_checkpoint.pt", index=None, metric=metrics[j], training_state=True)
                
            if "train" in write_output:
              self.ensemble[j].trainer.predict(self.ensemble[j].trainer.data_loader[0]["train_loader"], "train")
            if "val" in write_output:
              self.ensemble[j].trainer.predict(self.ensemble[j].trainer.data_loader[0]["val_loader"], "val")
            if "test" in write_output and self.ensemble[j].trainer.data_loader[0].get("test_loader"):
              self.ensemble[j].trainer.predict(self.ensemble[j].trainer.data_loader[0]["test_loader"], "test") 

        torch.cuda.empty_cache()
               
    except RuntimeError as e:
      self.ensemble[0].task._process_error(e)
      raise e
    
    for i in range(self.k):
      self.ensemble[i].trainer.model[0].eval()
      #self.ensemble[i].trainer.model[0] = compile(self.ensemble[i].trainer.model)
    device = next(iter(self.ensemble[0].trainer.model[0].state_dict().values(
      ))).get_device()
    device = 'cpu' if device == -1 else 'cuda:' + str(device)
    self.device = device
    #https://pytorch.org/tutorials/intermediate/ensembling.html
    models = [self.ensemble[i].trainer.model[0] for i in range(self.k)]
    self.params, self.buffers = stack_module_state(models)
    base_model = copy.deepcopy(models[0])
    self.base_model = base_model.to('meta')

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
