import torch
import numpy as np
from activestructopt.gnn.dataloader import prepare_data
from activestructopt.optimization.shared.constraints import lj_rmins, lj_repulsion, lj_reject
from pymatgen.core.lattice import Lattice

def get_cell(x):
  zero = (0 * x[0][0]).reshape(1)
  # adapted to pytorch from https://github.com/materialsproject/pymatgen/blob/v2023.10.4/pymatgen/core/lattice.py#L341
  return torch.reshape(torch.cat((x[0][0].reshape(1), zero, x[0][1].reshape(1), 
    x[0][2].reshape(1), x[1][0].reshape(1), x[1][1].reshape(1), 
    zero, zero, x[1][2].reshape(1))), (1, 3, 3))

def run_adam(ensemble, target, x0, starting_structure, config, ljrmins,
                    niters = 100, λ = 1.0, lr = 0.01, device = 'cpu'):
  ucbs = torch.zeros(niters, device = device)
  xs = torch.zeros((niters, 3 * x0.size()[0]), device = device)
  target = torch.tensor(target, device = device)
  data = prepare_data(starting_structure, config, pos_grad = True).to(device)
  x = x0
  optimizer = torch.optim.Adam([x], lr=lr)
  for i in range(niters):
    optimizer.zero_grad(set_to_none=True)
    x.requires_grad_()
    data.cell = get_cell(x)
    data.pos = x[2:]
    prediction = ensemble.ensemble[0].trainer.model._forward(data)
    for j in range(1, ensemble.k):
      prediction = torch.cat((prediction,
                ensemble.ensemble[j].trainer.model._forward(data)), dim = 0)
    mean = torch.mean(prediction, dim = 0)
    # last term to remove Bessel correction and match numpy behavior
    # https://github.com/pytorch/pytorch/issues/1082
    std = ensemble.scalar * torch.std(prediction, dim = 0) * np.sqrt(
      (ensemble.k - 1) / ensemble.k)
    yhat = torch.mean((std ** 2) + ((target - mean) ** 2))
    s = torch.sqrt(2 * torch.sum((std ** 4) + 2 * (std ** 2) * (
      (target - mean) ** 2))) / (len(target))
    ucb = yhat - λ * s + lj_repulsion(data, ljrmins)
    if i != niters - 1:
      ucb.backward()
      optimizer.step()
    xs[i] = x.detach().flatten()
    ucbs[i] = ucb.detach().item()
    yhat, s, mean, std, prediction, ucb = yhat.detach(), s.detach(
      ), mean.detach(), std.detach(), prediction.detach(), ucb.detach()
    del yhat, s, mean, std, prediction, ucb
    
  to_return = ucbs.detach().cpu().numpy(), xs.detach().cpu().numpy()
  del ucbs, xs, target, data
  return to_return

def basinhop(ensemble, starting_structures, target, config,
                  nhops = 10, niters = 100, λ = 1.0, lr = 0.01, 
                  step_size = 0.1, rmcσ = 0.0025, lstep_size = 0.05, 
                  θstep_size = 1.0):
  device = ensemble.device
  ucbs = np.zeros((nhops, niters))
  xs = np.zeros((nhops, niters, 6 + 3 * len(starting_structures[0])))
  ljrmins = torch.tensor(lj_rmins, device = device)

  for i in range(nhops):
    lat0 = torch.tensor([[starting_structures[i].lattice.matrix[0][0], 
      starting_structures[i].lattice.matrix[0][2],
      starting_structures[i].lattice.matrix[1][0]],
      [starting_structures[i].lattice.matrix[1][1],
      starting_structures[i].lattice.matrix[1][2],
      starting_structures[i].lattice.matrix[2][2]],
      ], device = device, dtype = torch.float)
    x0 = torch.tensor(starting_structures[i].lattice.get_cartesian_coords(
      starting_structures[i].frac_coords), device = device, dtype = torch.float)
    x0 = torch.cat((lat0, x0), 0)

    new_ucbs, new_xs = run_adam(ensemble, target, x0, starting_structures[i], 
      config, ljrmins, niters = niters, λ = λ, lr = lr, device = device)
    
    ucbs[i] = new_ucbs
    xs[i] = new_xs

  hop, iteration = np.unravel_index(np.argmin(ucbs), ucbs.shape)
  new_structure = starting_structures[0].copy()

  new_structure.lattice = Lattice([
    [xs[hop][iteration][0], 0, xs[hop][iteration][1]], 
    [xs[hop][iteration][2], xs[hop][iteration][3], xs[hop][iteration][4]], 
    [0, 0, xs[hop][iteration][5]]])
  for i in range(len(new_structure)):
    new_structure[i].coords = xs[hop][iteration][(3 * i):(3 * (i + 1))]
  return new_structure
