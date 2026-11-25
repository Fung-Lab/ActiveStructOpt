from activestructopt.common.registry import registry, setup_imports
from activestructopt.simulation.base import ASOSimulationException
from pymatgen.core.structure import Structure
from collections import OrderedDict
from traceback import format_exc
import numpy as np
import subprocess
import torch
import json
import time
import os
import gc
import copy

def load_structures(file):
  """The suggested structures saved by ActiveLearning.opt_step (a single
  'structure' in files written before multiple suggestions)."""
  with open(file, 'rb') as f:
    data = json.load(f)
  if 'structures' in data:
    return [Structure.from_dict(s) for s in data['structures']]
  return [Structure.from_dict(data['structure'])]

class ActiveLearning():
  def __init__(self, simfuncs, targets, initial_structure, index = -1, 
    config = None, target_structure = None, progress_file = None, 
    model_params_file = None, verbosity = 2, save_structures = True,
    save_progress_dir = None, save_initialization = False,
    override_config = False, sbatch = False):
    """sbatch: whether optimize will be run with an sbatch_template, so
    opt_step runs in the GPU job. When resuming from a progress_file, the
    sampler (and its constraints) are then only used in the GPU job, so they
    are not built here."""

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
      if progress_file.split(".")[-1] == 'json':
        with open(progress_file, 'rb') as f:
          progress_dict = json.load(f)
          self.configs = progress_dict['configs'] if not override_config else [
            simfunc.setup_config(config) for simfunc in simfuncs]

          self.opt_constraints = []
          for constraint_config in self.configs[0]['aso_params']['opt_constraints']:
            constraint_cls = registry.get_constraint_class(constraint_config['name'])
            self.opt_constraints.append(constraint_cls(**(constraint_config['args'])))

          # The resumed dataset does not sample, and in sbatch mode the
          # optimizer (the other user of the sampler) runs in the GPU job
          self.sampler_constraints = []
          self.sampler = None
          if not sbatch:
            for constraint_config in self.configs[0]['aso_params']['sampler_constraints']:
              constraint_cls = registry.get_constraint_class(constraint_config['name'])
              self.sampler_constraints.append(constraint_cls(**(constraint_config['args'])))
            sampler_cls = registry.get_sampler_class(
              self.configs[0]['aso_params']['sampler']['name'])
            self.sampler = sampler_cls(initial_structure, self.sampler_constraints,
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

      self.sampler_constraints = []
      for constraint_config in self.configs[0]['aso_params']['sampler_constraints']:
        constraint_cls = registry.get_constraint_class(constraint_config['name'])
        self.sampler_constraints.append(constraint_cls(**(constraint_config['args'])))

      self.opt_constraints = []
      for constraint_config in self.configs[0]['aso_params']['opt_constraints']:
        constraint_cls = registry.get_constraint_class(constraint_config['name'])
        self.opt_constraints.append(constraint_cls(**(constraint_config['args'])))

      sampler_cls = registry.get_sampler_class(
        self.configs[0]['aso_params']['sampler']['name'])
      self.sampler = sampler_cls(initial_structure, self.sampler_constraints,
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

    # Optional energy model, added to the optimizer objective scaled by the
    # optimizer's energy_scale
    self.energy_model = None
    energy_config = self.configs[0]['aso_params'].get('energy')
    if energy_config is not None:
      energy_cls = registry.get_energy_class(energy_config['name'])
      self.energy_model = energy_cls(**(energy_config.get('args', {})))

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
          # Up to max_sim_calls structures, simulated in order until one
          # succeeds
          if sbatch_template is None:
            if self.sampler is None:
              raise ValueError('No sampler: ActiveLearning was resumed with '
                'sbatch = True, so optimize needs an sbatch_template')
            new_structures = self.opt_step(predict_target = predict_target,
              save_file = None, n_structures = max_sim_calls)
          else:
            new_structures = self.opt_step_sbatch(sbatch_template, i,
              n_structures = max_sim_calls)
          #print(new_structure)
          #for ensemble_i in range(len(metrics)):
          #  print(metrics[ensemble_i]['val_error'])
        else:
          new_structures = load_structures(self.model_params_file)

        self.update_first_successful(new_structures)

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

  def update_first_successful(self, new_structures):
    """Simulates the structures in order, adding the first one whose
    simulation succeeds to the dataset."""
    for j, new_structure in enumerate(new_structures):
      try:
        self.dataset.update(new_structure)
        return
      except ASOSimulationException:
        print(f'Simulation of suggested structure {j + 1} of '
          f'{len(new_structures)} failed')
    raise ASOSimulationException("Max sim calls exceeded")

  def read_opt_step_sbatch(self, file):
    new_structures = load_structures(file)
    self.model_params_file = file
    self.update_first_successful(new_structures)

  def opt_step_sbatch(self, sbatch_template, stepi, n_structures = 1):
    # The GPU job always retrains because it starts without trained models
    with open(sbatch_template, 'r') as file:
      sbatch_data = file.read()
    job_name = int(time.time()) % 604800
    sbatch_data = sbatch_data.replace('##INDEX##', str(self.index))
    sbatch_data = sbatch_data.replace('##PROG_FILE##', self.last_prog_file)
    sbatch_data = sbatch_data.replace('##MODEL_PARAMS_FILE##',
      self.model_params_file)
    sbatch_data = sbatch_data.replace('##JOB_NAME##', str(job_name))
    sbatch_data = sbatch_data.replace('##N_STRUCTURES##', str(n_structures))
    new_job_file = f'gpu_job_{self.index}.sbatch'
    with open(new_job_file, 'w') as file:
      file.write(sbatch_data)

    try:
      subprocess.check_output(["sbatch", f"{new_job_file}"])
    except subprocess.CalledProcessError as e:
      print(e.output)
    
    done_file = f"DONE_{self.index}"
    finished = False
    while not finished:
      time.sleep(30)
      if os.path.isfile(done_file):
        finished = True
        os.remove(done_file)

    gpu_job_file = f"gpu_job_{self.index}_{stepi}.json"
    new_structures = load_structures(gpu_job_file)

    self.model_params_file = gpu_job_file
    prev_gpu_file = f"gpu_job_{self.index}_{stepi-1}.json"
    if os.path.exists(prev_gpu_file):
      os.remove(prev_gpu_file)
    return new_structures

  def opt_step(self, predict_target = False, save_file = None, retrain = True,
    n_structures = 1):
    """Returns a list of up to n_structures suggested structures, best
    first."""
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

    new_structures, obj_values = optimizer_cls().run(self.models,
      self.dataset, objective, self.sampler, self.opt_constraints,
      N = n_structures, energy_model = self.energy_model,
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
            'structures': [s.as_dict() for s in new_structures],
            'model_params': model_params,
            'obj_values': [[] if x is None else x.tolist(
              ) for x in self.opt_obj_values],
      }
      with open(save_file, "w") as file: 
        json.dump(res, file)

    return new_structures

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
    elif self.verbosity == 1:
      res = {'index': self.index,
            'dataset': self.dataset.toJSONDict(),
            'model_params': self.model_params, # this probably doesn't work as of now
            'error': self.error,
            'traceback': self.traceback}
      with open(filename, "w") as file:
        json.dump(res, file)
