from matdeeplearn.common.trainer_context import new_trainer_context
from activestructopt.gnn.dataloader import prepare_data
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from torch_geometric import compile
import torch
from torch.func import stack_module_state, functional_call, vmap
import copy
from torch_geometric.loader import DataLoader
from matdeeplearn.trainers.base_trainer import BaseTrainer

class Runner:
  def __init__(self):
    self.config = None

  def __call__(self, config, args, train_data, val_data):
    with new_trainer_context(args = args, config = config) as ctx:
      if config["task"]["parallel"] == True:
        local_world_size = os.environ.get("LOCAL_WORLD_SIZE", None)
        local_world_size = int(local_world_size)
        dist.init_process_group(
          "nccl", world_size=local_world_size, init_method="env://"
        )
        rank = int(dist.get_rank())
      else:
        rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_world_size = 1
      self.config = ctx.config
      self.task = ctx.task
      self.trainer = ctx.trainer
      self.trainer.dataset = {
        'train': train_data, 
        'val': val_data, 
      }
      self.trainer.sampler = BaseTrainer._load_sampler(config["optim"], self.trainer.dataset, local_world_size, rank)
      self.trainer.data_loader = BaseTrainer._load_dataloader(
        config["optim"],
        config["dataset"],
        self.trainer.dataset,
        self.trainer.sampler,
        config["task"]["run_mode"],
        config["model"]
      )
      self.task.setup(self.trainer)
      self.task.run()

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
        ConfigSetup('train'), self.datasets[i][0], self.datasets[i][1])
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
    with torch.inference_mode():
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
