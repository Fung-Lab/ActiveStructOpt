import torch
from activestructopt.objective.base import BaseObjective
from activestructopt.common.registry import registry

@registry.register_objective("UCB")
class UCB(BaseObjective):
  def __init__(self, λ = 1.0, **kwargs) -> None:
    self.λ = λ

  def get(self, predictions: list[torch.Tensor], target, device = 'cpu', N = 1):
    ucbs = torch.zeros(N, device = device)
    ucb_total = torch.tensor([0.0], device = device)
    for i in range(N):
      yhat = torch.mean((predictions[1][i] ** 2) + (
        (target - predictions[0][i]) ** 2))
      s = torch.sqrt(2 * torch.sum((predictions[1][i] ** 4) + 2 * (
        predictions[1][i] ** 2) * ((
        target - predictions[0][i]) ** 2))) / (len(target))
      ucb = torch.maximum(yhat - self.λ * s, torch.tensor(0.))
      ucb_total = ucb_total + ucb
      ucbs[i] = ucb.detach()
      del yhat, s, ucb
    return ucbs, ucb_total
