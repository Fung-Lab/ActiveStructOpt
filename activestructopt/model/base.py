from abc import ABC, abstractmethod
from activestructopt.dataset.base import BaseDataset
from pymatgen.core.structure import IStructure
from matdeeplearn.common.trainer_context import new_trainer_context
from matdeeplearn.common.trainer_context import setup_imports
from matdeeplearn.common.registry import registry
from matdeeplearn.trainers.base_trainer import BaseTrainer
from torch import distributed as dist
import torch
import os
import logging
import sys
from io import StringIO
import copy

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

  def __call__(self, config, args, train_data, val_data, local_world_size, rank):
    # https://github.com/Fung-Lab/MatDeepLearn_dev/blob/masked_node_loss/matdeeplearn/trainers/base_trainer.py
    # https://github.com/Fung-Lab/MatDeepLearn_dev/blob/masked_node_loss/matdeeplearn/common/trainer_context.py
    config = copy.deepcopy(config)
    setup_imports()
    trainer_cls = registry.get_trainer_class(config.get("trainer", "property"))
    assert trainer_cls is not None, "Trainer not found"

    task_cls = registry.get_task_class(config["task"]["run_mode"])
    assert task_cls is not None, "Task not found"
    self.task = task_cls(config)

    self.config = config

    dataset = {
      'train': train_data, 
      'val': val_data, 
    }
    model = trainer_cls._load_model(config["model"], 
      config["dataset"]["preprocess_params"], dataset, local_world_size, rank)
    optimizer = trainer_cls._load_optimizer(config["optim"], model, 
      local_world_size)
    sampler = BaseTrainer._load_sampler(config["optim"], 
      dataset, local_world_size, rank)
    data_loader = BaseTrainer._load_dataloader(
      config["optim"],
      config["dataset"],
      dataset,
      sampler,
      config["task"]["run_mode"],
      config["model"]
    )

    scheduler = trainer_cls._load_scheduler(config["optim"]["scheduler"], 
      optimizer)
    loss = trainer_cls._load_loss(config["optim"]["loss"])
    max_epochs = config["optim"]["max_epochs"]
    clip_grad_norm = config["optim"].get("clip_grad_norm", None)
    verbosity = config["optim"].get("verbosity", None)
    batch_tqdm = config["optim"].get("batch_tqdm", False)
    write_output = config["task"].get("write_output", [])
    output_frequency = config["task"].get("output_frequency", 0)
    model_save_frequency = config["task"].get("model_save_frequency", 0)
    max_checkpoint_epochs = config["optim"].get("max_checkpoint_epochs", None)
    identifier = config["task"].get("identifier", None)

    # pass in custom results home dir and load in prev checkpoint dir
    save_dir = config["task"].get("save_dir", None)
    checkpoint_path = config["task"].get("checkpoint_path", None)

    if local_world_size > 1:
        dist.barrier()
        
    self.trainer = trainer_cls(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        sampler=sampler,
        scheduler=scheduler,
        data_loader=data_loader,
        loss=loss,
        max_epochs=max_epochs,
        clip_grad_norm=clip_grad_norm,
        max_checkpoint_epochs=max_checkpoint_epochs,
        identifier=identifier,
        verbosity=verbosity,
        batch_tqdm=batch_tqdm,
        write_output=write_output,
        output_frequency=output_frequency,
        model_save_frequency=model_save_frequency,
        save_dir=save_dir,
        checkpoint_path=checkpoint_path,
        use_amp=config["task"].get("use_amp", False),
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
