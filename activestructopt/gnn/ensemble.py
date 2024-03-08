from matdeeplearn.trainers.base_trainer import BaseTrainer
from matdeeplearn.modules.scheduler import LRScheduler
from activestructopt.gnn.dataloader import prepare_data
import numpy as np
import torch
from torch.func import stack_module_state, functional_call, vmap
import torch.optim as optim
import torch.nn.functional as F
from torch import distributed as dist
import copy
import os
from torch_geometric.loader import DataLoader
from scipy.stats import norm
from scipy.optimize import minimize

class Ensemble:
  def __init__(self, k, config):
    self.k = k
    self.config = config
    self.scalar = 1.0
    self.loss_fn = getattr(F, config['optim']['loss']['loss_args']['loss_fn'])
    # https://github.com/Fung-Lab/MatDeepLearn_dev/blob/main/matdeeplearn/trainers/base_trainer.py#L136
    if config["task"]["parallel"] == True:
      world_size = os.environ.get("LOCAL_WORLD_SIZE", None)
      world_size = int(world_size)
      self.config["optim"]["lr"] = config["optim"]["lr"] * world_size
      dist.init_process_group(
          "nccl", world_size=world_size, init_method="env://"
      )
      self.device = int(dist.get_rank())
    else:
      world_size = 1
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [BaseTrainer._load_model(
      config['model'], config['dataset']['preprocess_params'], 
      None, world_size, self.device)[0] for _ in range(self.k)]
    params, buffers = stack_module_state(models)
    self.base_model = models[0].to('meta')
    self.params = params
    self.buffers = buffers
  
  def train(self, kfolds, trainval, trainval_targets):
    def fmodel(params, buffers, x):
      return functional_call(self.base_model, (params, buffers), 
        (x,))['output']

    trainval_batch = next(iter(DataLoader(trainval, 
          batch_size = len(trainval))))
        
    kfolds_tensors = [torch.tensor(np.array(kfolds[i]), device = self.device,
        dtype = torch.int) for i in range(len(kfolds))]
    train_inds = [torch.cat([kfolds_tensors[i] for i in range(
      self.k) if i != j]) for j in range(self.k)]
    val_inds = [kfolds_tensors[j] for j in range(self.k)]

    clip_grad_norm = self.config["optim"]["clip_grad_norm"]

    best_vals = [torch.inf for _ in range(self.k)]

    try:
      if str(self.device) not in ("cpu", "cuda"):
        dist.barrier()

      params = copy.deepcopy(self.params)
      buffers = copy.deepcopy(self.buffers)

      optimizer = getattr(optim, 
        self.config["optim"]["optimizer"]["optimizer_type"])(
        list(params.values()) + list(buffers.values()),
        lr = self.config["optim"]["lr"],
        **self.config["optim"]["optimizer"].get("optimizer_args", {}),
      )

      scheduler = LRScheduler(optimizer, 
        self.config["optim"]["scheduler"]["scheduler_type"], 
        self.config["optim"]["scheduler"]["scheduler_args"])
      
     
      for _ in range(self.config["optim"]["max_epochs"]):
        # Based on https://github.com/Fung-Lab/MatDeepLearn_dev/blob/main/matdeeplearn/trainers/property_trainer.py
        # Start training over epochs loop
        self.base_model.train()

        out_lists = vmap(fmodel, in_dims = (0, 0, None), 
          randomness = 'different')(params, buffers, trainval_batch)
        
        train_loss_total = torch.tensor([0.0], device = self.device)
        for j in range(self.k):
          train_loss_total += self.loss_fn(out_lists[j, train_inds[j], :], 
            trainval_targets[train_inds[j], :])

        optimizer.zero_grad(set_to_none=True)
        train_loss_total.backward()
        if clip_grad_norm:
          torch.nn.utils.clip_grad_norm_(
            list(params.values()) + list(buffers.values()),
            max_norm = clip_grad_norm,
          )
        optimizer.step()

        del out_lists

        if str(self.device) not in ("cpu", "cuda"):
          dist.barrier()

        self.base_model.eval()

        with torch.inference_mode():
          out_lists = vmap(fmodel, in_dims = (0, 0, None), randomness = 'same')(
            params, buffers, trainval_batch)
          if scheduler.scheduler_type == "ReduceLROnPlateau":
            scheduler.step(metrics = train_loss_total)
          else:
            scheduler.step()
          
          for j in range(self.k): # update prediction model if beats val losses
            vloss = self.loss_fn(out_lists[j, val_inds[j], :], 
              trainval_targets[val_inds[j], :]).item()
            if vloss < best_vals[j]:
              best_vals[j] = vloss
              for key in params.keys():
                self.params[key][j] = params[key][j]
              for key in buffers.keys():
                self.buffers[key][j] = buffers[key][j]

          del out_lists, vloss, train_loss_total
      del params, buffers
               
    except RuntimeError as e:
      # TODO: re-implement error processing checking the model as this uses it
      raise e
    
    del kfolds_tensors, train_inds, val_inds, best_vals

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
    with torch.inference_mode():
      test_res = self.predict(test_data, prepared = True)
    zscores = []
    for i in range(len(test_targets)):
      for j in range(len(test_targets[0])):
        zscores.append((
          test_res[0][i][j].item() - test_targets[i][j].item()
          ) / test_res[1][i][j].item())
    zscores = np.sort(zscores)
    normdist = norm()
    f = lambda x: np.trapz(np.abs(np.cumsum(np.ones(len(zscores))) / len(
      zscores) - normdist.cdf(zscores / x[0])), normdist.cdf(zscores / x[0]))
    self.scalar = minimize(f, [1.0]).x[0]
    return normdist.cdf(np.sort(zscores) / self.scalar), np.cumsum(
      np.ones(len(zscores))) / len(zscores)
