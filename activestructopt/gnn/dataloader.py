import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
import torch
from torch_geometric.data import Data
from matdeeplearn.preprocessor.helpers import (
    generate_edge_features,
    generate_node_features,
    calculate_edges_master,
)

def reduced_one_hot(Z):
  return torch.transpose(Z == torch.transpose(torch.unique(Z).repeat((Z.size()[0], 1)), 0, 1), 0, 1).float()

def prepare_data(
    structure, 
    config,
    y = None,
    pos_grad = False,
    device = None,
):
    if device == None:
      device = config['dataset_device']
    
    # based on https://github.com/Fung-Lab/MatDeepLearn_dev/blob/main/matdeeplearn/preprocessor/processor.py
    data = Data()
    adaptor = AseAtomsAdaptor()
    ase_crystal = adaptor.get_atoms(structure)
    data.batch = torch.zeros(len(structure), device = device, dtype = torch.long)
    data.n_atoms = torch.tensor([len(structure)], device = device, 
      dtype = torch.long)
    data.cell = torch.tensor([ase_crystal.get_cell().tolist()], 
      device = device, dtype = torch.float)
    data.z = torch.tensor(ase_crystal.get_atomic_numbers().tolist(), 
      device = device, dtype = torch.long)
    data.u = torch.Tensor(np.zeros((3))[np.newaxis, ...])
    data.pos = torch.tensor(ase_crystal.get_positions().tolist(), 
      device = device, dtype = torch.float)
    if pos_grad:
      data.pos.requires_grad_()

    reprocess_data(data, config, device)

    if y is not None:
      data.y = torch.tensor(np.array([y]))

    return data

def reprocess_data(data, config, device):
    r = config['preprocess_params']['cutoff_radius']
    n_neighbors = config['preprocess_params']['n_neighbors']

    if config['preprocess_params']['preprocess_edges']:
      edge_gen_out = calculate_edges_master(
        config['preprocess_params']['edge_calc_method'],
        r,
        n_neighbors,
        config['preprocess_params']['num_offsets'],
        ["_"],
        data.cell,
        data.pos,
        data.z,
        device = device
      ) 
                                              
      data.edge_index = edge_gen_out["edge_index"]
      data.edge_vec = edge_gen_out["edge_vec"]
      data.edge_weight = edge_gen_out["edge_weights"]
      data.cell_offsets = edge_gen_out["cell_offsets"]
      data.neighbors = edge_gen_out["neighbors"]            
    
      if(data.edge_vec.dim() > 2):
        data.edge_vec = data.edge_vec[data.edge_index[0], data.edge_index[1]] 

    if config['preprocess_params']['preprocess_edge_features']:
      data.edge_descriptor = {}
      data.edge_descriptor["distance"] = data.edge_weight
      data.distances = data.edge_weight

    if config['preprocess_params']['preprocess_node_features']:
      generate_node_features(data, n_neighbors, device=device, 
        node_rep_func = reduced_one_hot)
        
    if config['preprocess_params']['preprocess_edge_features']:
      generate_edge_features(data, config['preprocess_params']['edge_dim'], 
        r, device=device)
      if config['preprocess_params']['preprocess_edges']:
        delattr(data, "edge_descriptor")
