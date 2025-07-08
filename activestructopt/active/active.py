from activestructopt.common.registry import registry, setup_imports
from activestructopt.simulation.base import ASOSimulationException
from pymatgen.core.structure import Structure
from collections import OrderedDict
from traceback import format_exc
from pickle import dump, load
import numpy as np
import subprocess
import torch
import json
import time
import os
import gc
import copy

class ActiveLearning():
  def __init__(self, simfuncs, targets, initial_structure, index = -1, 
    config = None, target_structure = None, progress_file = None, 
    model_params_file = None, verbosity = 2, save_structures = True,
    save_progress_dir = None, save_initialization = False, 
    override_config = False):

    setup_imports()

    self.simfuncs = simfuncs
    self.index = index
    self.verbosity = verbosity

    self.last_prog_file = progress_file
    self.model_params = [None for _ in range(len(simfuncs))]
    self.model_errs = [[] for _ in range(len(simfuncs))]
    self.model_metrics = [[] for _ in range(len(simfuncs))]
    self.opt_obj_values = []
    self.new_structure_predictions = []
    self.target_structure = target_structure
    self.save_structures = save_structures
    if not (target_structure is None):
      self.target_predictions = []

    if progress_file is not None:
      if progress_file.split(".")[-1] == 'pkl':
        with open(progress_file, 'rb') as f:
          progress = load(f)
        self.configs = progress['configs']
        self.dataset = progress['dataset']
        self.model_params = progress['model_params']
        self.iteration = progress['dataset'].N - progress['dataset'].start_N - 1
      elif progress_file.split(".")[-1] == 'json':
        with open(progress_file, 'rb') as f:
          progress_dict = json.load(f)
          self.configs = progress_dict['configs'] if not override_config else [
            simfunc.setup_config(config) for simfunc in simfuncs]
          sampler_cls = registry.get_sampler_class(
            self.configs[0]['aso_params']['sampler']['name'])
          self.sampler = sampler_cls(initial_structure, 
            **(self.configs[0]['aso_params']['sampler']['args']))
          dataset_cls = registry.get_dataset_class(
            self.configs[0]['aso_params']['dataset']['name'])
          self.dataset = dataset_cls(simfuncs, self.sampler, initial_structure, 
            targets, self.configs[0]['dataset'], 
            progress_dict = progress_dict['dataset'], **(
            self.configs[0]['aso_params']['dataset']['args']))
          if model_params_file is not None:
            with open(progress_file, 'rb') as f2:
              model_params_dict = json.load(f2)
              self.model_params = []
              for i in range(len(model_params_dict['model_params'])):
                kparams = OrderedDict()
                for key, value in model_params_dict['model_params'][i].items():
                  kparams[key] = torch.tensor(value)
                self.model_params.append(kparams)
          else:
            self.model_params = []
            for i in range(len(progress_dict['model_params'])):
              kparams = OrderedDict()
              for key, value in progress_dict['model_params'][i].items():
                kparams[key] = torch.tensor(value)
              self.model_params.append(kparams)
      else:
        raise Exception("Progress file should be .pkl or .json") 

    else:
      self.iteration = 0
      self.configs = [simfunc.setup_config(copy.deepcopy(config)
        ) for simfunc in simfuncs]
      sampler_cls = registry.get_sampler_class(
        self.configs[0]['aso_params']['sampler']['name'])
      self.sampler = sampler_cls(initial_structure, 
        **(self.configs[0]['aso_params']['sampler']['args']))
      dataset_cls = registry.get_dataset_class(
        self.configs[0]['aso_params']['dataset']['name'])
      self.dataset = dataset_cls(simfuncs, self.sampler, initial_structure, 
        targets, self.configs[0]['dataset'], **(
        self.configs[0]['aso_params']['dataset']['args']))

    model_cls = registry.get_model_class(
      self.configs[0]['aso_params']['model']['name'])

    self.models = [model_cls(self.configs[i], self.simfuncs[i],
          **(self.configs[i]['aso_params']['model']['args'])
          ) if self.configs[i]['aso_params']['model']['name'
          ] == "GroundTruth" else model_cls(self.configs[i], 
          **(self.configs[i]['aso_params']['model']['args'])) for i in range(
            len(self.simfuncs))]

    self.traceback = None
    self.error = None
    self.model_params_file = 'None'

    if save_progress_dir is not None and save_initialization:
      if self.verbosity == 0 or self.verbosity == 0.5:
        self.save(os.path.join(save_progress_dir, str(self.index) + "_0.json"))
        self.last_prog_file = os.path.join(save_progress_dir, 
          str(self.index) + "_0.json")
      else:
        self.save(os.path.join(save_progress_dir, str(self.index) + "_0.pkl"))
        self.last_prog_file = os.path.join(save_progress_dir, 
          str(self.index) + "_0.pkl")
  
  def optimize(self, print_mismatches = True, save_progress_dir = None, 
    predict_target = False, new_structure_predict = False, 
    sbatch_template = None, max_sim_calls = 5, model_params_file = None):
    try:
      if model_params_file != None:
        self.model_params_file = model_params_file

      if print_mismatches:
        print(self.dataset.mismatches)

      for i in range(len(self.dataset.structures), 
        self.configs[0]['aso_params']['max_forward_calls']):
        
        if self.model_params_file == 'None' or int(
          self.model_params_file.split('.')[0].split('_')[-1]) < len(
          self.dataset.structures):
          if sbatch_template is None:
            new_structure = self.opt_step(predict_target = predict_target, 
              save_file = None)
          else:
            new_structure = self.opt_step_sbatch(sbatch_template, i)
          #print(new_structure)
          #for ensemble_i in range(len(metrics)):
          #  print(metrics[ensemble_i]['val_error'])
        else:
          with open(self.model_params_file, 'rb') as f:
            new_structure = Structure.from_dict(json.load(f)['structure'])
        
        sim_calculated = False
        sim_calls = 0

        while not sim_calculated:
          try:
            sim_calls += 1
            self.dataset.update(new_structure)
            sim_calculated = True
          except ASOSimulationException:
            if sim_calls <= max_sim_calls:
              if sbatch_template is None:
                new_structure = self.opt_step(predict_target = predict_target, 
                  save_file = None, retrain = False)
              else:
                new_structure = self.opt_step_sbatch(sbatch_template, i, 
                  retrain = False)
            else:
              raise ASOSimulationException("Max sim calls exceeded")


        #if new_structure_predict:
          #with torch.inference_mode():
          #  self.new_structure_predictions.append(self.model.predict(
          #    new_structure, 
          #    mask = self.dataset.simfunc.mask).cpu().numpy())

        if print_mismatches:
          print(self.dataset.mismatches[-1])

        gc.collect()
        torch.cuda.empty_cache()
        
        if save_progress_dir is not None:
          if self.verbosity == 0 or self.verbosity == 0.5:
            self.save(os.path.join(save_progress_dir, str(self.index
              ) + "_" + str(i) + ".json"))
            self.last_prog_file = os.path.join(save_progress_dir, 
              str(self.index) + "_" + str(i) + ".json")
            prev_progress_file = os.path.join(save_progress_dir, str(self.index
              ) + "_" + str(i - 1) + ".json")
          else:
            self.save(os.path.join(save_progress_dir, str(self.index
              ) + "_" + str(i) + ".pkl"))
            self.last_prog_file = os.path.join(save_progress_dir, 
              str(self.index) + "_" + str(i) + ".pkl")
            prev_progress_file = os.path.join(save_progress_dir, str(self.index
              ) + "_" + str(i - 1) + ".pkl")
          if os.path.exists(prev_progress_file):
            os.remove(prev_progress_file)
    except Exception as err:
      self.traceback = format_exc()
      self.error = err
      print(self.traceback)
      print(self.error)

  def read_opt_step_sbatch(self, file):
    with open(file, 'rb') as f:
      new_structure = Structure.from_dict(json.load(f)['structure'])
    self.model_params_file = file
    self.dataset.update(new_structure)

  def opt_step_sbatch(self, sbatch_template, stepi, retrain = True):
    #if self.model_params_file == f"gpu_job_{self.index}_{stepi}.json":
    #  retrain = False
    with open(sbatch_template, 'r') as file:
      sbatch_data = file.read()
    job_name = int(time.time()) % 604800
    sbatch_data = sbatch_data.replace('##PROG_FILE##', self.last_prog_file)
    sbatch_data = sbatch_data.replace('##MODEL_PARAMS_FILE##', 
      self.model_params_file)
    sbatch_data = sbatch_data.replace('##RETRAIN##', str(retrain))
    sbatch_data = sbatch_data.replace('##JOB_NAME##', str(job_name))
    new_job_file = f'gpu_job_{self.index}.sbatch'
    with open(new_job_file, 'w') as file:
      file.write(sbatch_data)

    try:
      subprocess.check_output(["sbatch", f"{new_job_file}"])
    except subprocess.CalledProcessError as e:
      print(e.output)
    
    finished = False
    while not finished:
      time.sleep(30)
      if os.path.isfile("DONE"):
        finished = True
        os.remove("DONE")

    gpu_job_file = f"gpu_job_{self.index}_{stepi}.json"
    with open(gpu_job_file, 'rb') as f:
      new_structure = Structure.from_dict(json.load(f)['structure'])
    
    self.model_params_file = gpu_job_file
    prev_gpu_file = f"gpu_job_{self.index}_{stepi-1}.json"
    if os.path.exists(prev_gpu_file):
      os.remove(prev_gpu_file)
    return new_structure

  def opt_step(self, predict_target = False, save_file = None, retrain = True):
    stepi = len(self.dataset.structures)

    if retrain:
      train_profile = self.configs[0]['aso_params']['model']['profiles'][
        np.searchsorted(-np.array(
          self.configs[0]['aso_params']['model']['switch_profiles']), 
          -(self.configs[0]['aso_params']['max_forward_calls'] - stepi))]
      
      for i in range(len(self.models)):
        model_err, metrics, model_params = self.models[i].train(
          self.dataset, i, **(train_profile))
        self.model_params[i] = model_params
        self.model_errs[i].append(model_err)
        self.model_metrics[i].append(metrics)

      #if not (self.target_structure is None) and predict_target:
      #  with torch.inference_mode():
      #    self.target_predictions.append(self.model.predict(
      #      self.target_structure, 
      #      mask = self.dataset.simfunc.mask).cpu().numpy())

    acq_profile = self.configs[0]['aso_params']['optimizer']['acq_profiles'][
        np.searchsorted(-np.array(
          self.configs[0]['aso_params']['optimizer']['switch_acq_profiles']), 
          -(self.configs[0]['aso_params']['max_forward_calls'] - stepi))]

    opt_profile = self.configs[0]['aso_params']['optimizer']['opt_profiles'][
        np.searchsorted(-np.array(
          self.configs[0]['aso_params']['optimizer']['switch_opt_profiles']), 
          -(self.configs[0]['aso_params']['max_forward_calls'] - stepi))]

    objective_cls = registry.get_objective_class(acq_profile['name'])
    objective = objective_cls(**(acq_profile['args']), )

    optimizer_cls = registry.get_optimizer_class(
      self.configs[0]['aso_params']['optimizer']['name'])

    new_structure, obj_values = optimizer_cls().run(self.models, 
      self.dataset, objective, self.sampler, 
      **(self.configs[0]['aso_params']['optimizer']['args']), **(opt_profile))
    self.opt_obj_values.append(obj_values)

    if not (save_file is None):
      split_save_file = save_file.split('.')
      save_file = split_save_file[0] + '_' + str(len(
        self.dataset.structures)) + '.' + split_save_file[1]
      model_params = []
      for i in range(len(self.model_params)):
        ind_model_params = []
        for j in range(len(self.model_params[i])):
          model_dict = {}
          state_dict = self.model_params[i][j]
          for param_tensor in state_dict:
            model_dict[param_tensor] = state_dict[param_tensor].detach().cpu(
              ).tolist()
          ind_model_params.append(model_dict)
        model_params.append(model_dict)
      res = {'index': self.index,
            'structure': new_structure.as_dict(),
            'model_params': model_params,
            'obj_values': [[] if x is None else x.tolist(
              ) for x in self.opt_obj_values],
      }
      with open(save_file, "w") as file: 
        json.dump(res, file)
        
    return new_structure

  def save(self, filename, additional_data = {}):
    if self.verbosity == 0:
      model_params = []
      if self.model_params is not None:
        for i in range(len(self.model_params)):
          model_dict = {}
          state_dict = self.model_params[i]
          if state_dict is not None:
            for param_tensor in state_dict:
              model_dict[param_tensor] = state_dict[param_tensor].detach().cpu(
                ).tolist()
          model_params.append(model_dict)

      res = {'index': self.index,
            'dataset': self.dataset.toJSONDict(
              save_structures = self.save_structures),
            'model_params': model_params,
            'obj_values': [[] if x is None else x.tolist(
              ) for x in self.opt_obj_values],
            'configs': self.configs,
      }
      with open(filename, "w") as file: 
        try:
          json.dump(res, file)
        except:
          print(res)
          raise NotImplementedError
    if self.verbosity == 0.5:
      res = {'index': self.index,
            'ys': [y.tolist() for y in self.dataset.ys],
            'target': self.dataset.target.tolist(),
            'mismatches': self.dataset.mismatches,
            'structures': [s.as_dict() for s in self.dataset.structures],
            'obj_values': [x.tolist() for x in self.opt_obj_values],
            'configs': self.configs,
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
