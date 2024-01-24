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
  ys = torch.tensor([optfunc(structures[i], **(args)) for i in range(N)], 
    device = device)
  pos = torch.tensor([structures[i].lattice.get_cartesian_coords(
    structures[i].frac_coords) for i in range(N)], device = device)      
  structure_indices = np.random.permutation(np.arange(1, N))
  trainval_indices = structure_indices[:int(np.floor(split * N) - 1)]
  trainval_indices = np.append(trainval_indices, [0])
  kfolds = np.array_split(trainval_indices, k)
  test_indices = structure_indices[int(np.floor(split * N) - 1):]
  train_indices = [np.concatenate(
      [kfolds[j] for j in range(k) if j != i]) for i in range(k)]

  train = torch.stack([pos[train_indices[i]] for i in range(k)])
  train_targets = torch.stack([ys[train_indices[i]] for i in range(k)])
  val = torch.stack([pos[train_indices[i]] for i in range(k)])
  val_targets = torch.stack([ys[train_indices[i]] for i in range(k)])
  test = [prepare_data(structures[i], config, y = ys[i]).to(device) for i in test_indices]
  test_targets = ys[test_indices].detach().cpu().numpy()

  return (structures, ys, kfolds, test_indices, 
    train, train_targets, val, val_targets, test, test_targets)

def update_datasets(train, train_targets, val, val_targets, new_structure, 
  optfunc, args, device):
  
  new_y = torch.tensor(optfunc(new_structure, **(args)), device = device)
  new_pos = torch.tensor(new_structure.lattice.get_cartesian_coords(
    new_structure.frac_coords), device = device)
  updated = False

  nfolds = val.size()[0]
  nstructs = val.size()[1]
  out_dim = val.size[2]

  for i in range(1, nfolds):
    if torch.max(val[i][nstructs - 1][:]) == 0:
      val[i][nstructs - 1][:] = new_pos
      val_targets[i][nstructs - 1][:] = new_y
      for j in range(nfolds):
        if i != j:
          train[j][nstructs - 1][:] = new_pos
          train_targets[j][nstructs - 1][:] = new_y
      updated = True
  
  if not updated:
    # https://discuss.pytorch.org/t/increasing-size-of-tensor-along-a-non-singleton-dimension/13676/4
    train = torch.cat([train, torch.zeros(nfolds, 1, out_dim)], 1)
    train_targets = torch.cat([train_targets, torch.zeros(nfolds, 1, out_dim)], 1)
    val = torch.cat([val, torch.zeros(nfolds, 1, out_dim)], 1)
    val_targets = torch.cat([val_targets, torch.zeros(nfolds, 1, out_dim)], 1)
    val[0][nstructs - 1][:] = new_pos
    val_targets[0][nstructs - 1][:] = new_y
    for j in range(1, nfolds):
      train[j][nstructs - 1][:] = new_pos
      train_targets[j][nstructs - 1][:] = new_y

  return train, train_targets, val, val_targets, new_y
