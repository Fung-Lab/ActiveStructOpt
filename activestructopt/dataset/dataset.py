from activestructopt.gnn.dataloader import prepare_data
from activestructopt.optimization.shared.constraints import lj_reject
import torch
import numpy as np

def make_data_splits(initial_structure, optfunc, args, config,
                      perturbrmin = 0.1, perturbrmax = 1.0, 
                      N = 100, split = 0.85, k = 5, device = 'cuda'):
  structures = [initial_structure.copy() for _ in range(N)]
  for i in range(1, N):
    rejected = True
    while rejected:
      new_structure = initial_structure.copy()
      new_structure.perturb(np.random.uniform(perturbrmin, perturbrmax))
      rejected = lj_reject(new_structure)
    structures[i] = new_structure.copy()
  ys = torch.tensor(np.array([optfunc(structures[i], **(args)
    ) for i in range(N)]), device = device)      
  structure_indices = np.random.permutation(np.arange(1, N))
  trainval_indices = structure_indices[:int(np.floor(split * N) - 1)]
  trainval_indices = np.append(trainval_indices, [0])
  test_indices = structure_indices[int(np.floor(split * N) - 1):]

  trainval = [prepare_data(structures[i], config, y = ys[i]).to(device) for i in trainval_indices]
  trainval_targets = ys[trainval_indices]
  test = [prepare_data(structures[i], config, y = ys[i]).to(device) for i in test_indices]
  test_targets = ys[test_indices]

  kfolds = np.array_split(np.arange(1, len(trainval_indices)), k)
  for i in range(k):
    kfolds[i] = kfolds[i].tolist()

  return (structures, ys, kfolds, test_indices, 
    trainval, trainval_targets, test, test_targets)

def update_datasets(kfolds, trainval, trainval_targets, new_structure, 
  config, optfunc, args, device):
  
  new_y = torch.unsqueeze(torch.tensor(optfunc(new_structure, **(args)), 
    device = device), 0)
  new_data = prepare_data(new_structure, config, y = new_y).to(device)

  kfolds[np.argmin([len(fold) for fold in kfolds])].append(len(trainval))
  trainval.append(new_data)
  trainval_targets = torch.cat([trainval_targets, new_y], 0)

  return kfolds, trainval, trainval_targets, new_y
