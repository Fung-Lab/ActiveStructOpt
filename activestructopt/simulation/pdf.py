from activestructopt.simulation.base import BaseSimulation
from activestructopt.common.registry import registry
from scipy.stats import norm
import numpy as np
from pymatgen.optimization.neighbors import find_points_in_spheres
from functools import reduce
import os
import subprocess
import shutil

@registry.register_simulation("PDF")
class PDF(BaseSimulation):
  def __init__(self, initial_structure, bisoequiv = 2.0, qmax = 18, qmin = 1, 
    rmax = 10, rmin = 1, python = None, 
    folder = None, **kwargs) -> None:
    self.bisoequiv = bisoequiv
    self.qmax = qmax
    self.qmin = qmin
    self.rmax = rmax
    self.rmin = rmin
    self.python = python
    self.mask = [True for _ in initial_structure.species]
    self.natoms = len(initial_structure)
    self.outdim = int(np.round(100 * (self.rmax - self.rmin)))
    self.parent_folder = folder

  def setup_config(self, config):
    config['dataset']['preprocess_params']['prediction_level'] = 'node'
    config['dataset']['preprocess_params']['output_dim'] = self.outdim
    return config

  def get(self, struct, group = False, separator = None):
    self.lattice_string = f'Lattice(a={struct.lattice.a}, b={struct.lattice.b}, c={struct.lattice.c}, alpha={struct.lattice.alpha}, beta={struct.lattice.beta}, gamma={struct.lattice.gamma})'
    atom_string = [f"""Atom(atype = '{s.species.elements[0].symbol}', xyz = [{s.frac_coords[0]}, {s.frac_coords[1]}, {s.frac_coords[2]}], occupancy = 1.0, lattice = lat),""" for s in struct.sites]
    self.atoms_string = f'[{reduce(lambda x, y: x + y, atom_string)}]'
    pass

  def resolve(self):
    subfolders = [int(x) for x in os.listdir(self.parent_folder)]
    new_folder = os.path.join(self.parent_folder, str(np.max(
      subfolders) + 1 if len(subfolders) > 0 else 0))
    os.mkdir(new_folder)

    file_content = f"""from diffpy.structure.lattice import Lattice
from diffpy.structure.atom import Atom
from diffpy.structure.structure import Structure
from diffpy.srreal.pdfcalculator import PDFCalculator
import numpy as np

def main():
    lat = {self.lattice_string}

    atoms = {self.atoms_string}

    bisoequiv = {self.bisoequiv}
    qmax = {self.qmax}
    qmin = {self.qmin}
    rmax = {self.rmax}
    rmin = {self.rmin}
    outfile = '{os.path.join(new_folder, 'pdf_out.txt')}'

    total_struct = Structure(atoms = atoms, lattice = lat)
    total_struct.Bisoequiv = bisoequiv

    pdfc0 = PDFCalculator(qmax = qmax, qmin = qmin, rmax = rmax, rmin = rmin)

    _, g0_total = pdfc0(total_struct)

    ind_contribs = []

    for j in range(len(atoms)):
        missing_atom_struct = Structure(atoms = [atoms[i] for i in np.where(np.arange(len(atoms)) != j)[0]], lattice = lat)
        missing_atom_struct.Bisoequiv = bisoequiv

        _, g0_ind = pdfc0(missing_atom_struct)
        ind_contribs.append(g0_total - g0_ind)

    # correct for broadening
    new_ind_contribs = len(atoms) * (np.array(ind_contribs) + np.repeat(
        ((g0_total - np.sum(np.array(ind_contribs), axis = 0)) / len(atoms))[np.newaxis, :], len(atoms), axis = 0))
    np.savetxt(outfile, new_ind_contribs)

if __name__ == "__main__":
    main()
"""
    with open(os.path.join(new_folder, 'run_pdf.py'), "w") as text_file:
      text_file.write(file_content)
    
    response = subprocess.check_output([self.python, os.path.join(new_folder, 'run_pdf.py')])
    pdf = np.loadtxt(os.path.join(new_folder, 'pdf_out.txt'))
    shutil.rmtree(new_folder)

    return pdf

  def garbage_collect(self, is_better):
    return

  def get_mismatch(self, to_compare, target):
    return np.mean((np.mean(to_compare, axis = 0) - target) ** 2)
