import torch
import numpy as np
from activestructopt.gnn.dataloader import prepare_data
from activestructopt.optimization.shared.constraints import lj_rmins, lj_repulsion
from matdeeplearn.preprocessor.helpers import (
    calculate_edges_master,
)

def run_adam(ensemble, target, starting_structures, config, ljrmins,
                    niters = 100, λ = 1.0, lr = 0.01, device = 'cpu'):
  nstarts = len(starting_structures)
  natoms = len(starting_structures[0])
  best_ucb = torch.tensor([float('inf')], device = device)
  best_x = torch.zeros(3 * natoms, device = device)
  target = torch.tensor(target, device = device)
  data = [prepare_data(s, config, pos_grad = True).to(device) for s in starting_structures]
  for i in range(nstarts):
    data[i].pos = torch.tensor(starting_structures[i].lattice.get_cartesian_coords(
        starting_structures[i].frac_coords), device = device, dtype = torch.float)
  optimizer = torch.optim.Adam([d.pos for d in data], lr=lr)
  for i in range(niters):
    optimizer.zero_grad(set_to_none=True)
    for j in range(nstarts):
      data[j].pos.requires_grad_()
    predictions = ensemble.predict(data, prepared = True)
    ucbs = torch.zeros(nstarts)
    for j in range(nstarts):
      yhat = torch.mean((predictions[1][j] ** 2) + ((target - predictions[0][j]) ** 2))
      s = torch.sqrt(2 * torch.sum((predictions[1][j] ** 4) + 2 * (predictions[1][j] ** 2) * (
        (target - predictions[0][j]) ** 2))) / (len(target))

      edge_gen_out = calculate_edges_master(
        config['preprocess_params']['edge_calc_method'],
        config['preprocess_params']['cutoff_radius'],
        config['preprocess_params']['n_neighbors'],
        config['preprocess_params']['num_offsets'],
        ["_"],
        data[j].cell,
        data[j].pos,
        data[j].z,
        device = device,
      )
      data[j].edge_index = edge_gen_out["edge_index"].to(device)
      data[j].edge_weight = edge_gen_out["edge_weights"].to(device)

      ucb = yhat - λ * s + lj_repulsion(data[j], ljrmins)
      ucb.backward(retain_graph = True)

      predictions[0][j] = torch.zeros(predictions[0][j].size())
      predictions[1][j] = torch.zeros(predictions[1][j].size())
      data[j].pos = data[j].pos.detach()
      ucbs[j] = ucb.detach()
      yhat, s = yhat.detach(), s.detach()
      del ucb, yhat, s, edge_gen_out
    if i != niters - 1:
      optimizer.step()
    if (torch.min(ucbs) < best_ucb).item():
      best_ucb = torch.min(ucbs).detach()
      best_x = data[torch.argmin(ucbs).item()].pos.detach().flatten()
    predictions = predictions.detach()
    del predictions, ucbs
    
  to_return = best_x.detach().cpu().numpy() 
  del best_ucb, best_x, target, data
  return to_return

def basinhop(ensemble, starting_structures, target, config,
                  nhops = 10, niters = 100, λ = 1.0, lr = 0.01, 
                  step_size = 0.1, rmcσ = 0.0025):
  device = ensemble.device
  ljrmins = torch.tensor(lj_rmins, device = device)

  new_x = run_adam(ensemble, target, starting_structures, config, ljrmins, 
    niters = niters, λ = λ, lr = lr, device = device)
  
  new_structure = starting_structures[0].copy()
  for i in range(len(new_structure)):
    new_structure[i].coords = new_x[(3 * i):(3 * (i + 1))]
  return new_structure
