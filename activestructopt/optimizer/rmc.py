from activestructopt.model.base import BaseModel
from activestructopt.dataset.base import BaseDataset
from activestructopt.objective.base import BaseObjective
from activestructopt.optimizer.base import BaseOptimizer
from activestructopt.sampler.base import BaseSampler
from activestructopt.common.registry import registry
from activestructopt.sampler.single_atom_perturbation import SingleAtomPerturbation
from pymatgen.core.structure import IStructure
from activestructopt.common.dataloader import prepare_data_pmg
import torch

@registry.register_optimizer("RMC")
class RMC(BaseOptimizer):
  def __init__(self) -> None:
    pass

  def run(self, model: BaseModel, dataset: BaseDataset, 
    objective: BaseObjective, sampler: BaseSampler, 
    steps = 10000, σ = 0.0025, latticeprob = 0.5, 
    σr = 0.1, σl = 0.1, σθ = 0.1, 
    save_obj_values = False, **kwargs) -> IStructure:

    device = model.device
    
    structure = dataset.structures[0].copy()
    prev_structure = structure.copy()
    target = torch.tensor(dataset.target, device = device)

    best_obj = torch.tensor([float('inf')], device = device)
    best_structure = None

    rmc_sampler = SingleAtomPerturbation(structure, perturbrmin = σr, 
      perturbrmax = σr, perturblmax = σl, perturbθmax = σθ, 
      lattice_prob = latticeprob)

    #obj_vals = None
    #if save_obj_values:
    #  obj_vals = torch.zeros((iters_per_start, starts), device = 'cpu')

    prev_obj = torch.tensor([float('inf')], device = device)

    for _ in range(steps):
      data = [prepare_data_pmg(structure, dataset.config, pos_grad = False, 
        device = device, preprocess = True, cell_grad = False
        )]
        
      with torch.inference_mode():
        predictions = model.predict(data, prepared = True, 
          mask = dataset.simfunc.mask)

      _, obj_total = objective.get(predictions, target, device = device, N = 1)

      #if save_obj_values:
      #  obj_vals[i, starti:(stopi + 1)] = objs.cpu()

      Δobjs = obj_total - prev_obj
      better = Δobjs <= 0
      hastings = torch.log(torch.rand(1, device = device)) < Δobjs / (
        -2 * σ ** 2)
      accept = torch.logical_or(better, hastings)
      if (obj_total < best_obj).item():
        best_obj = obj_total
        best_structure = structure.copy()
      if (accept).item():
        prev_structure = structure.copy()
        prev_obj = obj_total
    
      rmc_sampler.initial_structure = prev_structure.copy()
      structure = rmc_sampler.sample()

    return best_structure, None
