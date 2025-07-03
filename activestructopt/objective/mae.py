import torch
from activestructopt.objective.base import BaseObjective
from activestructopt.common.registry import registry

@registry.register_objective("MAE")
class MAE(BaseObjective):
  def __init__(self, weights = None, **kwargs) -> None:
    self.weights = weights

  def get(self, predictions: list[torch.Tensor], targets, device = 'cpu', N = 1, M = 1):
    if self.weights is None:
      weights = torch.ones(M, device = device)
    else:
      weights = torch.tensor(self.weights, device = device)

    maes = torch.zeros((M, N), device = device)
    mae_total = torch.tensor([0.0], device = device)
    for i in range(N):
      for j in range(M):
        mae = weights[j] * torch.mean(torch.abs(targets[j] - 
          predictions[j][0][i]))
        mae_total = mae_total + mae
        maes[j][i] = mae.detach()
        del mae

    return maes, mae_total
