from abc import ABC, abstractmethod
from activestructopt.dataset.base import BaseDataset
import torch
import os
import logging
from io import StringIO

class BaseModel(ABC):
  @abstractmethod
  def __init__(self, config, **kwargs):
    pass
  
  @abstractmethod
  def train(self, dataset: BaseDataset, **kwargs):
    pass

  @abstractmethod
  def predict(self, structure, **kwargs):
    pass

class Runner:
  def __init__(self):
    self.config = None
    self.logstream = StringIO()
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(self.logstream)
    sh.setLevel(logging.DEBUG)                               
    root_logger.addHandler(sh)

  def __call__(self, config, args, train_data, val_data):
    from matdeeplearn.common.trainer_context import new_trainer_context
    from matdeeplearn.trainers.base_trainer import BaseTrainer
    from torch import distributed as dist

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
      self.trainer.sampler = BaseTrainer._load_sampler(config["optim"], 
        self.trainer.dataset, local_world_size, rank)
      self.trainer.data_loader = BaseTrainer._load_dataloader(
        config["optim"],
        config["dataset"],
        self.trainer.dataset,
        self.trainer.sampler,
        config["task"]["run_mode"],
        config["model"]
      )
      self.task.setup(self.trainer)

  def train(self):
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
