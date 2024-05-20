from activestructopt.optimization.basinhopping.basinhopping import basinhop
from activestructopt.gnn.ensemble import Ensemble
from activestructopt.dataset.dataset import make_data_splits, update_datasets, get_perturbed_structure
from activestructopt.optimization.shared.constraints import lj_reject
import numpy as np
import gc
import torch
import pickle
import sys

def active_learning(
    optfunc, 
    args, 
    target,
    config, 
    initial_structure, 
    max_forward_calls = 100,
    N = 30, 
    k = 5, 
    perturbrmin = 0.0, 
    perturbrmax = 1.0, 
    perturblmax = 0.2, 
    perturbθmax = 10,
    split = 1/3, 
    device = 'cuda',
    bh_starts = 128,
    bh_iters_per_start = 100,
    bh_lr = 0.01,
    bh_step_size = 0.1,
    bh_σ = 0.0025,
    print_mses = True,
    save_progress_dir = None,
    λ = 1.0,
    seed = 0,
    finetune_epochs = 500,
    lr_reduction = 1.0,
    ):
  structures, ys, datasets, kfolds, test_indices, test_data, test_targets = make_data_splits(
    initial_structure,
    optfunc,
    args,
    config['dataset'],
    N = N,
    k = k,
    perturbrmin = perturbrmin,
    perturbrmax = perturbrmax,
    split = split,
    device = device,
    seed = seed,
  )
  config['dataset']['preprocess_params']['output_dim'] = len(ys[0])
  lr1, lr2 = config['optim']['lr'], config['optim']['lr'] / lr_reduction
  mses = [np.mean((y - target) ** 2) for y in ys]
  if print_mses:
    print(mses)
  active_steps = max_forward_calls - N
  ensemble = Ensemble(k, config)
  for i in range(active_steps):
    starting_structures = [initial_structure.copy() for _ in range(bh_starts)]
    for j in range(np.minimum(len(structures), bh_starts)):
      starting_structures[j] = structures[j].copy()
    if len(structures) < bh_starts:
      for j in range(len(structures), bh_starts):
        starting_structures[j] = get_perturbed_structure(initial_structure, 
          perturbrmin = perturbrmin, perturbrmax = perturbrmax, 
          perturblmax = perturblmax, perturbθmax = perturbθmax)

    ensemble.train(datasets, iterations = config['optim'][
      'max_epochs'] if i == 0 else finetune_epochs, lr = lr1 if i == 0 else lr2)
    ensemble.set_scalar_calibration(test_data, test_targets)
    new_structure = basinhop(ensemble, starting_structures, target, 
      config['dataset'], nhops = bh_starts, niters = bh_iters_per_start, 
      λ = 0.0 if i == (active_steps - 1) else λ, lr = bh_lr, 
      step_size = bh_step_size, rmcσ = bh_σ)
    structures.append(new_structure)
    datasets, y = update_datasets(
      datasets,
      new_structure,
      config['dataset'],
      optfunc,
      args,
      device,
    )
    ys.append(y)
    new_mse = np.mean((y - target) ** 2)
    mses.append(new_mse)
    if print_mses:
      print(new_mse)
    gc.collect()
    torch.cuda.empty_cache()
    if save_progress_dir is not None:
      res = {'index': sys.argv[1],
            'iter': i,
            'structures': structures,
            'ys': ys,
            'mses': mses}

      with open(save_progress_dir + "/" + str(sys.argv[1]) + "_" + str(i) + ".pkl", "wb") as file:
          pickle.dump(res, file)

  return structures, ys, mses, (
      datasets, kfolds, test_indices, test_data, test_targets, ensemble)
