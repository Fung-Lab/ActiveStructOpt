import torch
from activestructopt.objective.base import BaseObjective
from activestructopt.common.registry import registry

@registry.register_objective("MSEUncertainty")
class MSEUncertainty(BaseObjective):
  def __init__(self, λ = 0.1, weights = None, **kwargs) -> None:
    self.λ = λ
    self.weights = weights

  def get(self, predictions: list[torch.Tensor], targets, device = 'cpu', N = 1, M = 1):
    if self.weights is None:
      weights = torch.ones(M, device = device)
    else:
      weights = torch.tensor(self.weights, device = device)
    
    mses = torch.zeros((M, N), device = device)
    mse_total = torch.tensor([0.0], device = device)
    for i in range(N):
      for j in range(M):
        mse =  weights[j] * torch.maximum(torch.mean(torch.pow(targets[j] - predictions[j][0][i], 2)) - 
          self.λ * torch.mean(predictions[j][1][i]), torch.tensor(0.)) 
        mse_total = mse_total + mse
        mses[j][i] = mse.detach()
        del mse
    return mses, mse_total
