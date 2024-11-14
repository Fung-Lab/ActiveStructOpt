from activestructopt.common.registry import registry, setup_imports
from torch.cuda import empty_cache
from torch import inference_mode
import numpy as np
from gc import collect
from pickle import dump, load
from os.path import join as pathjoin
from os.path import exists as pathexists
from os import remove
from copy import deepcopy
from traceback import format_exc
from collections import OrderedDict
import json
import torch
import os

class ActiveLearning():
  def __init__(self, simfunc, target, config, initial_structure, 
    index = -1, target_structure = None, progress_file = None, verbosity = 2):
    setup_imports()

    self.simfunc = simfunc
    self.config = simfunc.setup_config(config)
    self.index = index
    self.iteration = 0
    self.verbosity = verbosity

    self.model_params = None
    self.model_errs = []
    self.model_metrics = []
    self.opt_obj_values = []
    self.new_structure_predictions = []
    self.target_structure = target_structure
    if not (target_structure is None):
      self.target_predictions = []

    sampler_cls = registry.get_sampler_class(
      self.config['aso_params']['sampler']['name'])
    self.sampler = sampler_cls(initial_structure, 
      **(self.config['aso_params']['sampler']['args']))

    if progress_file is not None:
      if progress_file.split(".")[-1] == 'pkl':
        with open(progress_file, 'rb') as f:
          progress = load(f)
        self.dataset = progress['dataset']
        self.model_params = progress['model_params']
        self.iteration = progress['dataset'].N - progress['dataset'].start_N - 1
      elif progress_file.split(".")[-1] == 'json':
        with open(progress_file, 'rb') as f:
          progress_dict = json.load(f)
          dataset_cls = registry.get_dataset_class(
            self.config['aso_params']['dataset']['name'])
          self.dataset = dataset_cls(simfunc, self.sampler, initial_structure, 
            target, self.config['dataset'], 
            progress_dict = progress_dict['dataset'], **(
            self.config['aso_params']['dataset']['args']))
          self.model_params = []
          for i in range(len(progress_dict['model_params'])):
            kparams = OrderedDict()
            for key, value in progress_dict['model_params'][i].items():
              kparams[key] = torch.tensor(value)
            self.model_params.append(kparams)
      else:
        raise Exception("Progress file should be .pkl or .json") 
    else:
      dataset_cls = registry.get_dataset_class(
        self.config['aso_params']['dataset']['name'])
      self.dataset = dataset_cls(simfunc, self.sampler, initial_structure, 
        target, self.config['dataset'], **(
        self.config['aso_params']['dataset']['args']))

    model_cls = registry.get_model_class(
      self.config['aso_params']['model']['name'])
    if self.config['aso_params']['model']['name'] == "GroundTruth":
      self.model = model_cls(self.config, self.simfunc,
        **(self.config['aso_params']['model']['args']))
    else:
      self.model = model_cls(self.config, 
        **(self.config['aso_params']['model']['args']))

    self.traceback = None
    self.error = None
  
  def optimize(self, print_mismatches = True, save_progress_dir = None, 
    predict_target = False, new_structure_predict = False):
    try:
      if print_mismatches:
        print(self.dataset.mismatches)

      for i in range(len(self.dataset.mismatches), 
        self.config['aso_params']['max_forward_calls']):
        train_profile = self.config['aso_params']['model']['profiles'][
          np.searchsorted(-np.array(
            self.config['aso_params']['model']['switch_profiles']), 
            -(self.config['aso_params']['max_forward_calls'] - i))]
        opt_profile = self.config['aso_params']['optimizer']['profiles'][
          np.searchsorted(-np.array(
            self.config['aso_params']['optimizer']['switch_profiles']), 
            -(self.config['aso_params']['max_forward_calls'] - i))]
        
        model_err, metrics, self.model_params = self.model.train(
          self.dataset, **(train_profile))
        self.model_errs.append(model_err)
        self.model_metrics.append(metrics)

        if not (self.target_structure is None) and predict_target:
          with inference_mode():
            self.target_predictions.append(self.model.predict(
              self.target_structure, 
              mask = self.dataset.simfunc.mask).cpu().numpy())

        objective_cls = registry.get_objective_class(opt_profile['name'])
        objective = objective_cls(**(opt_profile['args']))

        optimizer_cls = registry.get_optimizer_class(
          self.config['aso_params']['optimizer']['name'])

        new_structure, obj_values = optimizer_cls().run(self.model, 
          self.dataset, objective, self.sampler, 
          **(self.config['aso_params']['optimizer']['args']))
        self.opt_obj_values.append(obj_values)
        
        #print(new_structure)
        #for ensemble_i in range(len(metrics)):
        #  print(metrics[ensemble_i]['val_error'])
        self.dataset.update(new_structure)
        if new_structure_predict:
          with inference_mode():
            self.new_structure_predictions.append(self.model.predict(
              new_structure, 
              mask = self.dataset.simfunc.mask).cpu().numpy())

        if print_mismatches:
          print(self.dataset.mismatches[-1])

        collect()
        empty_cache()
        
        if save_progress_dir is not None:
          if self.verbosity == 0 or self.verbosity == 0.5:
            self.save(pathjoin(save_progress_dir, str(self.index) + "_" + str(
              i) + ".json"))
            prev_progress_file = pathjoin(save_progress_dir, str(self.index
              ) + "_" + str(i - 1) + ".json")
          else:
            self.save(pathjoin(save_progress_dir, str(self.index) + "_" + str(
              i) + ".pkl"))
            prev_progress_file = pathjoin(save_progress_dir, str(self.index
              ) + "_" + str(i - 1) + ".pkl")
          if pathexists(prev_progress_file):
            remove(prev_progress_file)
    except Exception as err:
      self.traceback = format_exc()
      self.error = err
      print(self.traceback)
      print(self.error)

  def save(self, filename, additional_data = {}):
    cpu_model_params = deepcopy(self.model_params)
    for i in range(len(cpu_model_params)):
      for param_tensor in cpu_model_params[i]:
        cpu_model_params[i][param_tensor] = cpu_model_params[i][
          param_tensor].detach().cpu()

    if self.verbosity == 0:
      model_params = []
      for i in range(len(self.model_params)):
        model_dict = {}
        state_dict = self.model_params[i]
        for param_tensor in state_dict:
          model_dict[param_tensor] = state_dict[param_tensor].detach().cpu(
            ).tolist()
        model_params.append(model_dict)

      res = {'index': self.index,
            'dataset': self.dataset.toJSONDict(),
            'model_params': model_params,
      }
      with open(filename, "w") as file: 
        json.dump(res, file)
    if self.verbosity == 0.5:
      res = {'index': self.index,
            'ys': [y.tolist() for y in self.dataset.ys],
            'target': self.dataset.target.tolist(),
            'mismatches': self.dataset.mismatches,
            'structures': [s.as_dict() for s in self.dataset.structures],
            'obj_values': [x.tolist() for x in self.opt_obj_values],
      }
      with open(filename, "w") as file: 
        json.dump(res, file)
    elif self.verbosity == 1:
      res = {'index': self.index,
            'dataset': self.dataset.toJSONDict(),
            'model_params': self.model_params, # this probably doesn't work as of now
            'error': self.error,
            'traceback': self.traceback}
      with open(filename, "w") as file:
        json.dump(res, file)
    elif self.verbosity == 2:
      res = {'index': self.index,
            'dataset': self.dataset,
            'model_errs': self.model_errs,
            'model_metrics': self.model_metrics,
            'model_params': self.model_params,
            'opt_obj_values': self.opt_obj_values,
            'new_structure_predictions': self.new_structure_predictions,
            'error': self.error,
            'traceback': self.traceback}
      if not (self.target_structure is None):
        res['target_predictions'] = self.target_predictions
      for k, v in additional_data.items():
        res[k] = v
      with open(filename, "wb") as file:
        dump(res, file)

  def train_model_and_save(self, save_progress_dir = None):
    try:
      train_profile = self.config['aso_params']['model']['profiles'][0]
      
      _, _, self.model_params = self.model.train(self.dataset, **(
        train_profile))

      out = {
        'model_params': self.model_params, 
        'model_scalar': self.model.scalar
      }

      torch.save(out, save_progress_dir + '/{}.pth'.format(self.index))

    except Exception as err:
      self.traceback = format_exc()
      self.error = err
      print(self.traceback)
      print(self.error)

  def load_model_and_optimize(self, model_params_dir, print_mismatches = True):
    params_file = model_params_dir + "/" + list(filter(
      lambda x: x.startswith("{}.".format(self.index)), os.listdir(
      model_params_dir)))[0]
    
    model_params = torch.load(params_file, weights_only=False)

    self.model.load(self.dataset, model_params['model_params'], 
      model_params['model_scalar'])

    opt_profile = self.config['aso_params']['optimizer']['profiles'][0]

    objective_cls = registry.get_objective_class(opt_profile['name'])
    objective = objective_cls(**(opt_profile['args']))

    optimizer_cls = registry.get_optimizer_class(
      self.config['aso_params']['optimizer']['name'])

    new_structure, obj_values = optimizer_cls().run(self.model, 
      self.dataset, objective, self.sampler, 
      **(self.config['aso_params']['optimizer']['args']))
    self.opt_obj_values.append(obj_values)
    
    self.dataset.update(new_structure)

    if print_mismatches:
      print(self.dataset.mismatches[-1])
