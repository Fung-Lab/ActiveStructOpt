from activestructopt.common.dataloader import prepare_data, reduced_one_hot
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
from matdeeplearn.preprocessor.helpers import (
    generate_edge_features,
    generate_node_features,
    calculate_edges_master,
)

@registry.register_model("GNNEnsemble")
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
    self.config['optim']['max_epochs'] = iterations
    self.config['optim']['lr'] = lr
    metrics = [{'epoch': [], 'lr': [], 'train_err': [], 'val_error': [], 
      'time': []} for _ in range(self.k)]

    fold = self.k - 1
    for i in range(self.k - 1):
      if len(dataset.datasets[i][1]) < len(dataset.datasets[i + 1][1]):
        fold = i
        break
    fold = (fold + 1) % self.k

    for i in range(self.k):
      #if i == fold and self.ensemble[i] is not None:
      #  break

      # Create new runner, with config and datasets
      new_runner = Runner()
      self.config['task']['seed'] = self.k * self.updates + i
      self.config['model']['gradient'] = False
      new_runner(self.config, ConfigSetup('train'), 
                            dataset.datasets[i][0], dataset.datasets[i][1])

      device = next(iter(new_runner.trainer.model[0].state_dict().values(
        ))).get_device()
      device = 'cpu' if device == -1 else 'cuda:' + str(device)
      self.device = device

      # If applicable, use the old model
      if prev_params is not None and not from_scratch:
        prev_state_dict = prev_params[i]
        rand_state_dict = new_runner.trainer.model[0].state_dict()
        for param_tensor in prev_state_dict:
          prev_state_dict[param_tensor] = (transfer * 
            prev_state_dict[param_tensor].to(device)) + ((1 - transfer) * 
            rand_state_dict[param_tensor])
        new_runner.trainer.model[0].load_state_dict(prev_state_dict)
      self.ensemble[i] = new_runner
      
      # Train
      self.ensemble[i].train()
      
      # Set to evaluation mode
      self.ensemble[i].trainer.model[0].eval()

      # Collect metrics from logger
      for l in self.ensemble[i].logstream.getvalue().split('\n'):
        if l.startswith('Epoch: '):
          metric_tokens = l.split()
          metrics[i]['epoch'].append(int(metric_tokens[1][:-1]))
          metrics[i]['lr'].append(float(metric_tokens[4][:-1]))
          metrics[i]['train_err'].append(float(metric_tokens[7][:-1]))
          metrics[i]['val_error'].append(float(metric_tokens[10][:-1]))
          metrics[i]['time'].append(float(metric_tokens[15][:-1]))
      
      # Erase logger
      # https://stackoverflow.com/questions/4330812/how-do-i-clear-a-stringio-object
      self.ensemble[i].logstream.seek(0)
      self.ensemble[i].logstream.truncate(0)
      
      #self.ensemble[i].trainer.model[0] = compile(self.ensemble[i].trainer.model)
    
    new_params = [self.ensemble[i].trainer.model[0].state_dict(
      ) for i in range(self.k)]

    gnn_mae, _, _ = self.set_scalar_calibration(dataset)
    self.updates = self.updates + 1
    
    return gnn_mae, metrics, new_params

  def predict(self, structure, prepared = False, mask = None, **kwargs):
    data = structure if prepared else [prepare_data(
      structure, self.config['dataset']).to(self.device)]

    N = len(data)
    mask = torch.tensor(mask, dtype = torch.bool)

    data = next(iter(DataLoader(data, batch_size = N)))

    self.ensemble[0].trainer.model[0].gradient = True
    data.edge_index, data.edge_weight, data.edge_vec, _, _, _ = self.ensemble[0].trainer.model[0].generate_graph(
      data, self.ensemble[0].trainer.model[0].cutoff_radius, self.ensemble[0].trainer.model[0].n_neighbors)
    self.ensemble[0].trainer.model[0].gradient = False

    out = [self.ensemble[i].trainer.model[0].forward(data)['output'] for i in range(self.k)]
    collated = torch.stack([torch.stack([torch.mean(out[j][torch.where(
      data.batch == i)][torch.where(mask)], dim = 0) for i in range(
      N)]) for j in range(self.k)])
  
    mean = torch.mean(collated, dim = 0)
    # last term to remove Bessel correction and match numpy behavior
    # https://github.com/pytorch/pytorch/issues/1082
    std = self.scalar * torch.std(collated, dim = 0) * np.sqrt(
      (self.k - 1) / self.k)

    return torch.stack((mean, std))

  def set_scalar_calibration(self, dataset: KFoldsDataset):
    self.scalar = 1.0
    #with torch.inference_mode():
    test_res = self.predict(dataset.test_data, prepared = True, 
      mask = dataset.simfunc.mask)
    aes = []
    zscores = []
    for i in range(len(dataset.test_targets)):
      target = np.mean(dataset.test_targets[i][np.array(dataset.simfunc.mask)], 
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
