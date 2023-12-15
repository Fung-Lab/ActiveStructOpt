from activestructopt.gnn.dataloader import prepare_data
from activestructopt.optimization.shared.constraints import lj_reject
import numpy as np

def make_data_splits(initial_structure, optfunc, args, config, 
                      perturbrmin = 0.1, perturbrmax = 1.0, 
                      perturblmin = -0.2, perturblmax = 0.2,
                      perturbθmin = -5, perturbθmax = 5,
                      N = 100, split = 0.85, k = 5, device = 'cuda'):
  structures = [initial_structure.copy() for _ in range(N)]
  for i in range(1, N):
    rejected = True
    while rejected:
      new_structure = initial_structure.copy()
      new_structure.perturb(np.random.uniform(perturbrmin, perturbrmax))
      new_structure.lattice = new_structure.lattice.from_parameters(
        max(0.0, new_structure.lattice.a + np.random.uniform(
          perturblmin, perturblmax)),
        max(0.0, new_structure.lattice.b + np.random.uniform(
          perturblmin, perturblmax)),
        max(0.0, new_structure.lattice.c + np.random.uniform(
          perturblmin, perturblmax)), 
        min(180.0, max(0.0, new_structure.lattice.alpha + np.random.uniform(
          perturbθmin, perturbθmax))), 
        min(180.0, max(0.0, new_structure.lattice.beta + np.random.uniform(
          perturbθmin, perturbθmax))), 
        min(180.0, max(0.0, new_structure.lattice.gamma + np.random.uniform(
          perturbθmin, perturbθmax)))
      )
      rejected = lj_reject(new_structure)
    structures[i] = new_structure.copy()
  ys = [optfunc(structures[i], **(args)) for i in range(N)]
  data = [prepare_data(structures[i], config, y = ys[i]).to(
    device) for i in range(N)]
      
  structure_indices = np.random.permutation(np.arange(1, N))
  trainval_indices = structure_indices[:int(np.floor(split * N) - 1)]
  trainval_indices = np.append(trainval_indices, [0])
  kfolds = np.array_split(trainval_indices, k)
  test_indices = structure_indices[int(np.floor(split * N) - 1):]
  test_data = [data[i] for i in test_indices]
  test_targets = [ys[i] for i in test_indices]
  train_indices = [np.concatenate(
    [kfolds[j] for j in range(k) if j != i]) for i in range(k)]
  
  datasets = [([data[j] for j in train_indices[i]], 
    [data[j] for j in kfolds[i]]) for i in range(k)]
  
  return structures, ys, datasets, kfolds, test_indices, test_data, test_targets

def update_datasets(datasets, new_structure, config, optfunc, args, device):
  y = optfunc(new_structure, **(args))
  new_data = prepare_data(new_structure, config, y = y).to(device)
  fold = len(datasets) - 1
  for i in range(len(datasets) - 1):
    if len(datasets[i][1]) < len(datasets[i + 1][1]):
      fold = i
      break
  datasets[fold][1].append(new_data)
  for i in range(len(datasets)):
    if fold != i:
      datasets[i][0].append(new_data)
  return datasets, y
