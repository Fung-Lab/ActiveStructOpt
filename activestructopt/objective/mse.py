import torch
from activestructopt.objective.base import BaseObjective
from activestructopt.common.registry import registry

@registry.register_objective("MSE")
class MSE(BaseObjective):
  def __init__(self, **kwargs) -> None:
    pass

  def get(self, predictions: list[torch.Tensor], target, device = 'cpu', N = 1):
    mses = torch.zeros(N, device = device)
    mse_total = torch.tensor([0.0], device = device)
    for i in range(N):
      mse = torch.mean((target - predictions[0][i]) ** 2)
      mse_total = mse_total + mse
      mses[i] = mse.detach()
      del mse
    return mses, mse_total
