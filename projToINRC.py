"""""
Naivly project the xyz coordinates to internal coordinates. 
"""""

import numpy as np
from numba import jit
from numba.typed import List


def parse_zmatrix(file_path):
    """
    Read the Z-matrix from a file.
    """

    zmatrix = []
    with open(file_path, "r") as file:
        for line in file:
            tokens = line.split()
            if tokens:
                zmatrix.append(tokens)
    return zmatrix

def preprocess_zmatrix(zmatrix):
    """
    Convert zmatrix to a format that can be processed by Numba.
    """

    processed = []
    for row in zmatrix:
        processed_row = []
        for item in row:
            try:
                processed_row.append(float(item))
            except ValueError:
                processed_row.append(-1)  # Example handling for non-numeric values
        processed.append(np.array(processed_row, dtype=np.float64))
    return List(processed)

#@jit(nopython=True)
def calculate_distance(atom1, atom2):
    """
    Calculate the distance between two atoms.
    """

    return np.linalg.norm(atom1 - atom2)

#@jit(nopython=True)
def calculate_angle(atom1, atom2, atom3):
   """
   Calculate the angle formed by three atoms.
   """

   vec1 = atom1 - atom2
   vec2 = atom3 - atom2
   cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
   angle = np.arccos(cos_angle)
   return np.degrees(angle)

#@jit(nopython=True)
def calculate_dihedral(atom1, atom2, atom3, atom4):
   """
   Calculate the dihedral angle formed by four atoms.
   """

   # Normal vectors to planes formed by atoms 1-2-3 and atoms 2-3-4
   n1 = np.cross(atom1 - atom2, atom3 - atom2)
   n2 = np.cross(atom2 - atom3, atom4 - atom3)
   # Normalize
   n1 /= np.linalg.norm(n1)
   n2 /= np.linalg.norm(n2)
   # Cosine and sine of dihedral angle
   cos_dihedral = np.dot(n1, n2)
   sin_dihedral = np.dot(np.cross(n1, n2), atom3 - atom2) / np.linalg.norm(atom3 - atom2)
   # Dihedral angle in degrees
   dihedral = np.arctan2(sin_dihedral, cos_dihedral)
   return np.degrees(dihedral)

#@jit(nopython=True)
def proj(R, zmatrix):
    
    INRC_R = []
    INRC_A = []
    INRC_D = []

    for i, entry in enumerate(zmatrix):
        if i == 0:
            continue  
        elif i == 1:
            # Calculate bond length
            bond = calculate_distance(R[0], R[1])
            INRC_R.append(bond)
        elif i == 2:
            # Calculate bond length and angle
            bond = calculate_distance(R[0], R[2])
            angle = calculate_angle(R[2], R[0], R[1])
            INRC_R.append(bond)
            INRC_A.append(angle)
        else:
            # Calculate bond, angle, and dihedral
            bond = calculate_distance(R[0], R[i])
            angle = calculate_angle(R[i], R[0], R[1])
            dihedral = calculate_dihedral(R[i], R[0], R[1], R[2])
            INRC_R.append(bond)
            INRC_A.append(angle)
            INRC_D.append(dihedral)

    return INRC_R, INRC_A, INRC_D


# Main execution
def proj_main(R):
    raw_zmatrix = parse_zmatrix("geomNRC.in")
    processed_zmatrix = preprocess_zmatrix(raw_zmatrix)
    return proj(R, processed_zmatrix) # proj[0]: bonds; proj[1]: angle; proj[2]: dihedral

