import torch
from activestructopt.gnn.dataloader import prepare_data, reprocess_data
from activestructopt.optimization.shared.constraints import lj_rmins, lj_repulsion

def run_adam(ensemble, target, starting_structures, config, ljrmins,
                    niters = 100, λ = 1.0, lr = 0.01, device = 'cpu'):
  nstarts = len(starting_structures)
  natoms = len(starting_structures[0])
  best_ucb = torch.tensor([float('inf')], device = device)
  best_x = torch.zeros(3 * natoms, device = device)
  target = torch.tensor(target, device = device)
  data = [prepare_data(s, config, pos_grad = True, device = device, 
    preprocess = False) for s in starting_structures]
  for i in range(nstarts): # process node features
    reprocess_data(data[i], config, device, edges = False)
                      
  optimizer = torch.optim.Adam([d.pos for d in data], lr=lr)
  
  large_structure = False

  for i in range(niters):
    optimizer.zero_grad(set_to_none=True)
    for j in range(nstarts):
      data[j].pos.requires_grad_()
      reprocess_data(data[j], config, device, nodes = False)

    if not large_structure:
      try:
        predictions = ensemble.predict(data, prepared = True)
        ucbs = torch.zeros(nstarts)
        ucb_total = torch.tensor([0.0], device = device)
        for j in range(nstarts):
          yhat = torch.mean((predictions[1][j] ** 2) + (
            (target - predictions[0][j]) ** 2))
          s = torch.sqrt(2 * torch.sum((predictions[1][j] ** 4) + 2 * (
            predictions[1][j] ** 2) * ((target - predictions[0][j]) ** 2))) / (
            len(target))
          ucb = yhat - λ * s + lj_repulsion(data[j], ljrmins)
          ucb_total = ucb_total + ucb
          ucbs[j] = ucb.detach()
        ucb_total.backward()
        del predictions, ucb, yhat, s
      except torch.cuda.OutOfMemoryError:
        large_structure = True

    if large_structure:
      ucbs = torch.zeros(nstarts)
      for j in range(nstarts):
        predictions = ensemble.predict([data[j]], prepared = True)
        yhat = torch.mean((predictions[1][0] ** 2) + (
          (target - predictions[0][0]) ** 2))
        s = torch.sqrt(2 * torch.sum((predictions[1][0] ** 4) + 2 * (
          predictions[1][0] ** 2) * ((target - predictions[0][0]) ** 2))) / (
          len(target))
        ucb = yhat - λ * s + lj_repulsion(data[j], ljrmins)
        ucbs[j] = ucb.detach()
        ucb.backward()
        del predictions, yhat, s, ucb
    
    if i != niters - 1:
      optimizer.step()
    if (torch.min(ucbs) < best_ucb).item():
      best_ucb = torch.min(ucbs).detach()
      best_x = data[torch.argmin(ucbs).item()].pos.detach().flatten()
    del ucbs
    
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
