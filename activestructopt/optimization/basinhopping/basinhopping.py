import torch
import numpy as np
from activestructopt.gnn.dataloader import prepare_data, reprocess_data
from activestructopt.optimization.shared.constraints import lj_rmins, lj_repulsion
from matdeeplearn.preprocessor.helpers import (
    calculate_edges_master,
)
from pymatgen.core import Lattice

def run_adam(ensemble, target, starting_structures, config, ljrmins,
                    niters = 100, λ = 1.0, lr = 0.01, device = 'cpu'):
  nstarts = len(starting_structures)
  natoms = len(starting_structures[0])
  best_ucb = torch.tensor([float('inf')], device = device)
  best_x = torch.zeros(3 * natoms, device = device)
  best_cell = torch.zeros((3, 3), device = device) 
  target = torch.tensor(target, device = device)
  data = [prepare_data(s, config, pos_grad = True, device = device) for s in starting_structures]
  for i in range(nstarts):
    data[i].pos = torch.tensor(starting_structures[i].lattice.get_cartesian_coords(
        starting_structures[i].frac_coords), device = device, dtype = torch.float)
    data[i].cell = torch.tensor(starting_structures[i].lattice.matrix, 
                                device = device, dtype = torch.float).unsqueeze(0)
  optimizer = torch.optim.Adam([d.pos for d in data] + [d.cell for d in data], lr=lr)
  for i in range(niters):
    optimizer.zero_grad(set_to_none=True)
    for j in range(nstarts):
      data[j].pos.requires_grad_()
      reprocess_data(data[j], config, device)
    predictions = ensemble.predict(data, prepared = True)
    ucbs = torch.zeros(nstarts)
    ucb_total = torch.tensor([0.0], device = device)
    for j in range(nstarts):
      yhat = torch.mean((predictions[1][j] ** 2) + ((target - predictions[0][j]) ** 2))
      s = torch.sqrt(2 * torch.sum((predictions[1][j] ** 4) + 2 * (predictions[1][j] ** 2) * (
        (target - predictions[0][j]) ** 2))) / (len(target))
      ucb = yhat - λ * s + lj_repulsion(data[j], ljrmins)
      ucb_total += ucb
      ucbs[j] = ucb.detach()
    ucb_total.backward()
    if i != niters - 1:
      optimizer.step()
    if (torch.min(ucbs) < best_ucb).item():
      best_ucb = torch.min(ucbs).detach()
      best_x = data[torch.argmin(ucbs).item()].pos.detach().flatten()
      best_cell = data[torch.argmin(ucbs).item()].cell[0].detach()
    predictions = predictions.detach()
    data[j].pos = data[j].pos.detach()
    ucb = ucb.detach()
    yhat = yhat.detach()
    s = s.detach()
    del predictions, ucbs, ucb, yhat, s
    
  to_return = best_x.detach().cpu().numpy(), best_cell.detach().cpu().numpy()
  del best_ucb, best_x, best_cell, target, data
  return to_return

def basinhop(ensemble, starting_structures, target, config,
                  nhops = 10, niters = 100, λ = 1.0, lr = 0.01, 
                  step_size = 0.1, rmcσ = 0.0025):
  device = ensemble.device
  ljrmins = torch.tensor(lj_rmins, device = device)

  new_x, new_cell = run_adam(ensemble, target, starting_structures, config, ljrmins, 
    niters = niters, λ = λ, lr = lr, device = device)
  
  new_structure = starting_structures[0].copy()
  new_structure.lattice = Lattice(new_cell)
  for i in range(len(new_structure)):
    new_structure[i].coords = new_x[(3 * i):(3 * (i + 1))]
  return new_structure
