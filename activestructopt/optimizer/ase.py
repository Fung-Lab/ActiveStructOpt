from activestructopt.common.dataloader import prepare_data_ase
from activestructopt.common.constraints import lj_rmins, lj_repulsion
from activestructopt.model.base import BaseModel
from activestructopt.dataset.base import BaseDataset
from activestructopt.objective.base import BaseObjective
from activestructopt.optimizer.base import BaseOptimizer
from activestructopt.sampler.base import BaseSampler
from activestructopt.common.registry import registry
from pymatgen.core.structure import IStructure
from pymatgen.io.ase import AseAtomsAdaptor
import ase
import ase.optimize
import torch
import numpy as np

from ase.calculators.calculator import Calculator, all_changes
from ase.filters import FrechetCellFilter

class ASOCalc(Calculator):
  def __init__(self, model, dataset, objective, target, device, 
    constraint_scale, ljrmins, **kwargs):
    self.implemented_properties = ['energy', 'energies', 'forces', 'stress']
    Calculator.__init__(self, **kwargs)
    self.model = model
    self.dataset = dataset
    self.device = device
    self.objective = objective
    self.target = target
    self.constraint_scale = constraint_scale
    self.ljrmins = ljrmins

  def calculate(self, atoms = None, properties = ['energy'],
                system_changes = all_changes):
    Calculator.calculate(self, atoms, properties, system_changes)

    data = [prepare_data_ase(self.atoms, self.dataset.config, pos_grad = True, 
      cell_grad = True, device = self.device, preprocess = True)]

    predictions = self.model.predict(data, prepared = True, 
      mask = self.dataset.simfunc.mask)

    _, obj_total = self.objective.get(predictions, self.target, 
      device = self.device, N = 1)
    
    obj_total = obj_total + self.constraint_scale * lj_repulsion(data[0], 
      self.ljrmins)

    grads = torch.autograd.grad(obj_total, [data[0].pos, data[0].cell]) 

    self.results['energy'] = obj_total
    self.results['energies'] = np.ones(len(atoms)) * (obj_total / len(atoms))
    self.results['forces'] = grads[0].squeeze(0).detach().cpu().numpy()
    self.results['stress'] = grads[1].squeeze(0).detach().cpu().numpy()

@registry.register_optimizer("ASE")
class ASE(BaseOptimizer):
  def __init__(self) -> None:
    pass

  def run(self, model: BaseModel, dataset: BaseDataset, 
    objective: BaseObjective, sampler: BaseSampler, 
    starts = 128, iters_per_start = 100, optimizer = "FIRE",
    optimizer_args = {}, optimize_atoms = True, 
    optimize_lattice = False, constraint_scale = 1.0,
    **kwargs) -> IStructure:
    
    starting_structures = [dataset.structures[j].copy(
      ) if j < dataset.N else sampler.sample(
      ) for j in range(starts)]
  
    device = model.device
    nstarts = len(starting_structures)
    ljrmins = torch.tensor(lj_rmins, device = device)
    best_obj = np.inf
    best_struct = None
    target = torch.tensor(dataset.target, device = device)

    adaptor = AseAtomsAdaptor()
    
    for j in range(nstarts):
      ase_crystal = adaptor.get_atoms(starting_structures[j])
      ase_crystal.calc = ASOCalc(model, dataset, objective, 
        target, device, constraint_scale, ljrmins)
      if optimize_lattice:
        ase_crystal = FrechetCellFilter(ase_crystal)
      dyn = getattr(ase.optimize, optimizer)(ase_crystal)
      dyn.run(steps = iters_per_start, **optimizer_args)
      if ase_crystal.calc.results['energy'] < best_obj:
        best_struct = adaptor.get_structure(ase_crystal)
    
    return best_struct, None
