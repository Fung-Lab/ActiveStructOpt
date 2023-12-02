from matdeeplearn.common.config.build_config import build_config
from matdeeplearn.common.trainer_context import new_trainer_context
from matdeeplearn.preprocessor.processor import process_data
import activestructopt.gnn.dataloader
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from torch_geometric import compile

class Runner:
  def __init__(self):
    self.config = None

  def __call__(self, config, args):
    with new_trainer_context(args = args, config = config) as ctx:
        self.config = ctx.config
        self.task = ctx.task
        self.trainer = ctx.trainer
        self.task.setup(self.trainer)
        self.task.run()

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
      self.ensemble[i].trainer.model.eval()
      #self.ensemble[i].trainer.model = compile(self.ensemble[i].trainer.model)
    device = next(iter(self.ensemble[0].trainer.model.state_dict().values(
      ))).get_device()
    device = 'cpu' if device == -1 else 'cuda:' + str(device)
    self.device = device

  def predict(self, structure, prepared = False):
    ensemble_results = []
    if not prepared:
      data = activestructopt.gnn.dataloader.prepare_data(
        structure, self.config['dataset']).to(self.device)
    else:
      data = structure
    for i in range(self.k):
      ensemble_results.append(
        self.ensemble[i].trainer.model._forward(
        data).cpu().detach().numpy()[0])
    return np.mean(np.array(ensemble_results), 0), np.std(
      np.array(ensemble_results), 0) * self.scalar

  def set_scalar_calibration(self, test_data, test_targets):
    self.scalar = 1.0
    test_res = [self.predict(s, prepared = True) for s in test_data]
    zscores = []
    for i in range(len(test_targets)):
      for j in range(len(test_targets[0])):
        zscores.append((
          test_res[i][0][j] - test_targets[i][j]) / test_res[i][1][j])
    zscores = np.sort(zscores)
    normdist = norm()
    f = lambda x: np.trapz(np.abs(np.cumsum(np.ones(len(zscores))) / len(
      zscores) - normdist.cdf(zscores / x[0])), normdist.cdf(zscores / x[0]))
    self.scalar = minimize(f, [1.0]).x[0]
    return normdist.cdf(np.sort(zscores) / self.scalar), np.cumsum(
      np.ones(len(zscores))) / len(zscores)
