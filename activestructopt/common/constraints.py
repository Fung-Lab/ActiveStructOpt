import torch
import periodictable
from pathlib import Path
import numpy as np

lj_rmins = np.genfromtxt(str(Path(__file__).parent / "lj_rmins.csv"), 
  delimiter=",")
el_symbols = np.array([periodictable.elements[i].symbol for i in range(95)])

def get_z(site):
  return np.argmax(el_symbols == site.species.elements[0].symbol)

def lj_repulsion(data, ljrmins, scale = 40):
  rmins = ljrmins[(data.z[data.edge_index[0]] - 1), 
    (data.z[data.edge_index[1]] - 1)]
  repulsions = torch.where(rmins > data.edge_weight, 
    torch.pow(rmins / data.edge_weight, 12), 1.0)
  return (torch.mean(repulsions) - 1) / scale

def lj_repulsion_pymatgen(structure, scale = 40, buffer = 0.85):
  repulsions = []
  for i in range(len(structure)):
    for j in range(i, len(structure)):
      rmin = lj_rmins[get_z(structure.sites[i]) - 1, get_z(
        structure.sites[j]) - 1] * buffer
      r = np.min([structure.lattice.a, structure.lattice.b, 
        structure.lattice.c]) if i == j else structure.sites[i].distance(
        structure.sites[j])
      repulsions.append(max(0, (rmin / r) ** 12 - 1))
  return np.mean(repulsions) / scale


def lj_reject(structure, buffer = 0.85):
  for i in range(len(structure)):
    for j in range(i + 1, len(structure)):
      if structure.sites[i].distance(structure.sites[j]) < (lj_rmins[get_z(
        structure.sites[i]) - 1][get_z(structure.sites[j]) - 1] * buffer):
        return True
  return False
