from activestructopt.model.base import BaseModel
from activestructopt.common.registry import registry
from activestructopt.dataset.base import BaseDataset

@registry.register_model("GaussianProcess")
class GaussianProcess(BaseModel):
  def __init__(self, config, **kwargs):
    pass

  def train(self, dataset: BaseDataset, **kwargs):
    try:
      from botorch.models import SingleTaskGP
      from botorch.models.transforms import Normalize, Standardize
      from botorch.acquisition import LogExpectedImprovement
      from botorch.fit import fit_gpytorch_mll
      from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError as e:
      raise ImportError('Error loading botorch') from e
    try:
      import torch
    except ImportError as e:
      raise ImportError('Error loading torch') from e
    
    gp = SingleTaskGP(
      train_X = dataset.X,
      train_Y = dataset.Y,
      input_transform=Normalize(d=dataset.num_atoms*3),
      outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    self.acqf = LogExpectedImprovement(model=gp, best_f = dataset.Y.max())

    return None, None, torch.empty(0)

  def predict(self, structure, **kwargs):
    try:
      import torch
    except ImportError as e:
      raise ImportError('Error loading torch') from e
    return torch.empty(0)
