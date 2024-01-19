from matdeeplearn.common.trainer_context import new_trainer_context
from activestructopt.gnn.dataloader import prepare_data
import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import minimize
from torch_geometric import compile
import torch
from torch.func import stack_module_state, functional_call, vmap
from torch.cuda.amp import autocast
from torch import distributed as dist
import copy
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
  def __init__(self, run_mode, train_data, val_data):
      self.run_mode = run_mode
      self.seed = None
      self.submit = None
      self.datasets = {
        'train': train_data, 
        'val': val_data, 
      }

class Ensemble:
  def __init__(self, k, config, datasets):
    self.k = k
    self.config = config
    self.datasets = datasets
    self.ensemble = [Runner() for _ in range(k)]
    self.scalar = 1.0
    self.device = 'cpu'
  
  def train(self):
    for i in range(self.k):
      self.ensemble[i](self.config, 
        ConfigSetup('train', self.datasets[i][0], self.datasets[i][1]))
    try:
      start_epoch = int(self.ensemble[0].trainer.epoch)

      end_epoch = (
        self.ensemble[0].trainer.max_checkpoint_epochs + start_epoch
        if self.ensemble[0].trainer.max_checkpoint_epochs
        else self.ensemble[0].trainer.max_epochs
      )

      if str(self.ensemble[0].trainer.rank) not in ("cpu", "cuda"):
        dist.barrier()
      
      for j in range(self.k):
        # Based on https://github.com/Fung-Lab/MatDeepLearn_dev/blob/main/matdeeplearn/trainers/property_trainer.py
        # Start training over epochs loop
        for epoch in range(start_epoch, end_epoch):
          epoch_start_time = time.time()
          if self.ensemble[j].trainer.train_sampler:
            self.ensemble[j].trainer.train_sampler.set_epoch(epoch)
          train_loader_iter = []
          for i in range(len(self.ensemble[j].trainer.model)):
            train_loader_iter.append(iter(self.ensemble[j].trainer.data_loader[i]["train_loader"]))
          
          # metrics for every epoch
          _metrics = {}
          
          for i in range(len(self.ensemble[j].trainer.data_loader[0]["train_loader"])):
            self.ensemble[j].trainer.model[0].train()
            batch = [next(train_loader_iter[0]).to(self.ensemble[j].trainer.rank)]

            # Compute forward, loss, backward    
            with autocast(enabled=self.ensemble[j].trainer.use_amp):
              out_list = self.ensemble[j].trainer._forward(batch)                                            
              loss = self.ensemble[j].trainer._compute_loss(out_list, batch) 
            grad_norm = []
            for i in range(len(self.ensemble[j].trainer.model)):
              grad_norm.append(self.ensemble[j].trainer._backward(loss[i], i))

            # Compute metrics
            _metrics = self.ensemble[j].trainer._compute_metrics(out_list[0], batch[0], _metrics)
            self.ensemble[j].trainer.metrics[0] = self.ensemble[j].trainer.evaluator.update("loss", loss[0].item(), out_list[0]["output"].shape[0], _metrics)

          self.ensemble[j].trainer.epoch = epoch + 1

          if str(self.ensemble[j].trainer.rank) not in ("cpu", "cuda"):
            dist.barrier()

          # Save current model      
          torch.cuda.empty_cache()                 
          if str(self.ensemble[j].trainer.rank) in ("0", "cpu", "cuda"):
            if self.ensemble[j].trainer.model_save_frequency == 1:
              self.ensemble[j].trainer.save_model(checkpoint_file="checkpoint.pt", training_state=True)

            # Evaluate on validation set if it exists
            if self.ensemble[j].trainer.data_loader[0].get("val_loader"):
              metric = self.ensemble[j].trainer.validate("val") 
            else:
              metric = self.ensemble[j].trainer.metrics

            # Train loop timings
            self.ensemble[j].trainer.epoch_time = time.time() - epoch_start_time
            # Log metrics
            if epoch % self.ensemble[j].trainer.train_verbosity == 0:
              if self.ensemble[j].trainer.data_loader[0].get("val_loader"):
                self.ensemble[j].trainer._log_metrics(metric)
              else:
                self.ensemble[j].trainer._log_metrics()

            # Update best val metric and model, and save best model and predicted outputs
            if metric[0][type(self.ensemble[j].trainer.loss_fn).__name__]["metric"] < self.ensemble[j].trainer.best_metric[0]:
              if self.ensemble[j].trainer.output_frequency == 0:
                if self.ensemble[j].trainer.model_save_frequency == 1:
                  self.ensemble[j].trainer.update_best_model(metric[0], 0, write_model=True, write_csv=False)
                else:
                  self.ensemble[j].trainer.update_best_model(metric[0], 0, write_model=False, write_csv=False)
              elif self.ensemble[j].trainer.output_frequency == 1:
                if self.ensemble[j].trainer.model_save_frequency == 1:
                  self.ensemble[j].trainer.update_best_model(metric[0], 0, write_model=True, write_csv=True)
                else:
                  self.ensemble[j].trainer.update_best_model(metric[0], 0, write_model=False, write_csv=True)
                
            self.ensemble[j].trainer._scheduler_step()

          torch.cuda.empty_cache()        
        
        if self.ensemble[j].trainer.best_model_state:
          if str(self.ensemble[j].trainer.rank) in "0":
            self.ensemble[j].trainer.model[0].module.load_state_dict(self.ensemble[j].trainer.best_model_state[0])
          elif str(self.ensemble[j].trainer.rank) in ("cpu", "cuda"):
            self.ensemble[j].trainer.model[0].load_state_dict(self.ensemble[j].trainer.best_model_state[0])

          if self.ensemble[j].trainer.model_save_frequency != -1:
            self.ensemble[j].trainer.save_model("best_checkpoint.pt", index=None, metric=metric, training_state=True)
              
          if "train" in self.ensemble[j].trainer.write_output:
            self.ensemble[j].trainer.predict(self.ensemble[j].trainer.data_loader[0]["train_loader"], "train")
          if "val" in self.ensemble[j].trainer.write_output and self.ensemble[j].trainer.data_loader[0].get("val_loader"):
            self.ensemble[j].trainer.predict(self.ensemble[j].trainer.data_loader[0]["val_loader"], "val")
          if "test" in self.ensemble[j].trainer.write_output and self.ensemble[j].trainer.data_loader[0].get("test_loader"):
            self.ensemble[j].trainer.predict(self.ensemble[j].trainer.data_loader[0]["test_loader"], "test") 
               
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
