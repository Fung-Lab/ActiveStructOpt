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

def split_param_buffers(param_buffers):
  new_param_buffers = []
  can_split = False
  for i in range(len(param_buffers)):
    p, b = param_buffers[i]
    pkeys = list(p.keys())
    bkeys = list(b.keys())
    if p[pkeys[0]].size()[0] > 1:
      chunk_size = int(np.ceil(p[pkeys[0]].size()[0] / 2))
      can_split = True
      p_split_1 = {}
      p_split_2 = {}
      b_split_1 = {}
      b_split_2 = {}
      for j in range(len(pkeys)):
        split_tensor = torch.split(p[pkeys[j]], chunk_size)
        p_split_1[pkeys[j]] = split_tensor[0].detach().clone().requires_grad_()
        p_split_2[pkeys[j]] = split_tensor[1].detach().clone().requires_grad_()
      for j in range(len(bkeys)):
        split_tensor = torch.split(b[bkeys[j]], chunk_size)
        b_split_1[bkeys[j]] = split_tensor[0].detach().clone().requires_grad_()
        b_split_2[bkeys[j]] = split_tensor[1].detach().clone().requires_grad_()
      new_param_buffers.append((p_split_1, b_split_1))
      new_param_buffers.append((p_split_2, b_split_2))
    else:
      new_param_buffers.append(param_buffers[i])
  assert can_split, "Out of memory with only one model in vmap dimension"
  return new_param_buffers

def recombine_param_buffers(param_buffers):
  params, buffers = param_buffers[0]
  pkeys = list(params.keys())
  bkeys = list(buffers.keys())
  for i in range(1, len(param_buffers)):
    for j in range(len(pkeys)):
      params[pkeys[j]] = torch.cat((params[pkeys[j]], 
        param_buffers[i][0][pkeys[j]]))
    for j in range(len(bkeys)):
      buffers[bkeys[j]] = torch.cat((buffers[bkeys[j]], 
        param_buffers[i][1][bkeys[j]]))
  for j in range(len(pkeys)):
    # https://stackoverflow.com/questions/75875801/why-is-tensor-not-a-leaf-tensor
    params[pkeys[j]] = params[pkeys[j]].detach().clone().requires_grad_()
  for j in range(len(bkeys)):
    params[bkeys[j]] = params[bkeys[j]].detach().clone().requires_grad_()
  return [(params, buffers)]

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

    self.base_model = BaseTrainer._load_model(config['model'], 
      config['dataset']['preprocess_params'], None, world_size, self.device)[0]
    self.param_buffers = [stack_module_state([BaseTrainer._load_model(
      config['model'], config['dataset']['preprocess_params'], None, world_size,
      self.device)[0] for _ in range(self.k)])]

  
  def train(self, kfolds, trainval, trainval_targets):
    trained = False

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

    while not trained:
      try:
        if str(self.device) not in ("cpu", "cuda"):
          dist.barrier()

        param_buffers = copy.deepcopy(self.param_buffers)
        model_splits = len(param_buffers)

        optimizers = [getattr(optim, 
          self.config["optim"]["optimizer"]["optimizer_type"])(
          list(param_buffers[i][0].values()) + list(param_buffers[i][1].values()),
          lr = self.config["optim"]["lr"],
          **self.config["optim"]["optimizer"].get("optimizer_args", {}),
        ) for i in range(model_splits)]

        schedulers = [LRScheduler(optimizers[i], 
          self.config["optim"]["scheduler"]["scheduler_type"], 
          self.config["optim"]["scheduler"]["scheduler_args"]) for i in range(
            model_splits)]
        
      
        for _ in range(self.config["optim"]["max_epochs"]):
          # Based on https://github.com/Fung-Lab/MatDeepLearn_dev/blob/main/matdeeplearn/trainers/property_trainer.py
          # Start training over epochs loop
          self.base_model.train()

          j_so_far = 0
          for i in range(model_splits):
            out_lists = vmap(fmodel, in_dims = (0, 0, None), 
              randomness = 'different')(param_buffers[i][0], 
              param_buffers[i][1], trainval_batch)
            
            train_loss_total = torch.tensor([0.0], device = self.device)
            for j in range(out_lists.size()[0]):
              train_loss_total += self.loss_fn(out_lists[j, 
                train_inds[j_so_far + j], :], 
                trainval_targets[train_inds[j_so_far + j], :])
            j_so_far += out_lists.size()[0]

            optimizers[i].zero_grad(set_to_none=True)
            train_loss_total.backward()
            if clip_grad_norm:
              torch.nn.utils.clip_grad_norm_(
                list(param_buffers[i][0].values()) + list(
                  param_buffers[i][1].values()),
                max_norm = clip_grad_norm,
              )
            optimizers[i].step()
            if schedulers[i].scheduler_type == "ReduceLROnPlateau":
              schedulers[i].step(metrics = train_loss_total)
            else:
              schedulers[i].step()

            del out_lists, train_loss_total

          if str(self.device) not in ("cpu", "cuda"):
            dist.barrier()
          self.base_model.eval()

          with torch.no_grad():
            j_so_far = 0
            for i in range(model_splits):
              out_lists = vmap(fmodel, in_dims = (0, 0, None), 
                randomness = 'different')(param_buffers[i][0], 
                param_buffers[i][1], trainval_batch)
              
              for j in range(out_lists.size()[0]): # update prediction model if beats val losses
                vloss = self.loss_fn(out_lists[j, val_inds[j_so_far + j], :], 
                  trainval_targets[val_inds[j_so_far + j], :]).item()
                if vloss < best_vals[j_so_far + j]:
                  best_vals[j_so_far + j] = vloss
                  for key in param_buffers[i][0].keys():
                    self.param_buffers[i][0][key][j] = param_buffers[i][0][key][j]
                  for key in param_buffers[i][1].keys():
                    self.param_buffers[i][1][key][j] = param_buffers[i][1][key][j]
              j_so_far += out_lists.size()[0]

              del out_lists, vloss
        del param_buffers
        trained = True

      except torch.cuda.OutOfMemoryError: # TODO: re-implement error processing checking the model as this uses it
        torch.cuda.empty_cache()
        self.param_buffers = split_param_buffers(self.param_buffers)

    self.param_buffers = recombine_param_buffers(self.param_buffers)
      
    del kfolds_tensors, train_inds, val_inds, best_vals

    torch.cuda.empty_cache()


  def predict(self, structure, prepared = False):
    params, buffers = self.param_buffers[0]
    def fmodel(params, buffers, x):
      return functional_call(self.base_model, (params, buffers), (x,))['output']
    data = structure if prepared else [prepare_data(
      structure, self.config['dataset']).to(self.device)]
    prediction = vmap(fmodel, in_dims = (0, 0, None), chunk_size = 1)(
      params, buffers, next(iter(DataLoader(data, batch_size = len(data)))))

    mean = torch.mean(prediction, dim = 0)
    # last term to remove Bessel correction and match numpy behavior
    # https://github.com/pytorch/pytorch/issues/1082
    std = self.scalar * torch.std(prediction, dim = 0) * np.sqrt(
      (self.k - 1) / self.k)

    del prediction
    torch.cuda.empty_cache()

    return torch.stack((mean, std))

  def set_scalar_calibration(self, test_data, test_targets, lr = 0.001, 
    n_iters = 1000):
    self.scalar = 1.0
    with torch.no_grad():
      test_res = self.predict(test_data, prepared = True)
    zscores = ((test_res[0, :, :] - test_targets) / 
      test_res[1, :, :]).flatten()
    zscores, _ = torch.sort(zscores)
    observed = torch.cumsum(torch.ones(zscores.size(), dtype = zscores.dtype, 
      device = self.device), 0) / zscores.size()[0]

    scalar = torch.tensor([1.0], device = self.device)
    optimizer = torch.optim.Adam([scalar], lr = lr)
    for _ in range(n_iters):
      optimizer.zero_grad(set_to_none=True)
      scalar.requires_grad_()
      # https://pytorch.org/docs/master/_modules/torch/distributions/normal.html#Normal.cdf
      expected = 0.5 * (1 + torch.erf((zscores / scalar) / np.sqrt(2)))
      area_diff = torch.trapezoid(torch.abs(observed - expected), expected)
      area_diff.backward()
      optimizer.step()

    self.scalar = scalar.item()
    return expected, observed
