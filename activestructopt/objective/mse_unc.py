import torch
from activestructopt.objective.base import BaseObjective
from activestructopt.common.registry import registry

@registry.register_objective("MSEUncertainty")
class MSEUncertainty(BaseObjective):
  def __init__(self, λ = 0.1, **kwargs) -> None:
    self.λ = λ

  def get(self, predictions: torch.Tensor, target, device = 'cpu', N = 1, ):
    mses = torch.zeros(N, device = device)
    mse_total = torch.tensor([0.0], device = device)
    for i in range(N):
      mse = torch.maximum(torch.mean(torch.pow(target - predictions[0][i], 2)) - 
        self.λ * torch.mean(predictions[1][i]), torch.tensor(0.)) 
      mse_total = mse_total + mse
      mses[i] = mse.detach()
      del mse
    return mses, mse_total
