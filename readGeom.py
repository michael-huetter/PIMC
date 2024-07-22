"""""
Read the molecular geometry given in the xyz format in geom.in
"""""

import numpy as np

def read_xyz(file_path):
    """Read the geom.in file."""

    with open(file_path, 'r') as file:
        lines = file.readlines()
        atoms = []
        coordinates = []

        for line in lines[2:]:  
            parts = line.split()
            atoms.append(parts[0])
            coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return atoms, np.array(coordinates)

def getGeom():
    atoms, coord = read_xyz("geom.in")
    return atoms, coord