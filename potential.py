"""
Define your potential here. getV functino is calles from the main MCMC loop!!!
"""

import numpy as np
from numba import njit
import configparser
from pathlib import Path
import joblib

from projToINRC import proj_main


config = configparser.ConfigParser()
here = Path(__file__).resolve().parent
config.read(here / "input.in")
use_jit = config.getboolean("PIMC", "use_jit") 

k_str = config.get('maxent', 'k')
k = np.array([float(x.strip()) for x in k_str.split(",")], dtype=np.float64)

def cJIT(func):

    if use_jit:
        return njit()(func)
    else:
        return func


############Some example potentials for testing#######################
@cJIT
def _HO(R):
    total = 0.0
    for i in range(R.shape[0]):
        total += 0.5 * k[i] * np.dot(R[i], R[i])
    return total

############Called from main code#######################

@cJIT
def getV(R: np.array, eState: int) -> float:
    """
    Called from main code to get the potential energy at a given geometry R. 
    """

    return _HO(R)


    
@cJIT   
def getGradV(R: np.array, eState: int) -> np.array:
    """
    Only needed if virial estimator is used.
    """

    return np.array([R[0][0], R[0][1], R[0][2]])

@cJIT
def getDiabV(R: np.array) -> tuple:
    """
    Only needed in the diabatic limit. So if non_adiabatic_coupling or PoE is used.
    """

    pass