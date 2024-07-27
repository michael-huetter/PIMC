"""
Define your potential here. getV functino is calles from the main MCMC loop!!!
"""

import numpy as np
from numba import njit
import configparser

config = configparser.ConfigParser()
config.read('input.in')
use_jit = config.getboolean("PIMC", "use_jit") 

def conditional_jit(func):

    if use_jit:
        return njit()(func)
    else:
        return func


############Some example potentials for testing#######################

@conditional_jit
def _HO(R):

    return 0.5 * (R[0][0]**2 + R[0][1]**2 + R[0][2]**2)

############Called from main code#######################
    
@conditional_jit
def getV(R: np.array, eState: np.array) -> float:
    """
    Called from main code to get the potential energy at a given geometry R. 
    """

    if eState[0] == 0:
        return _HO(R)
    else:
        return _HO(R) + 2
    
@conditional_jit   
def getGradV(R: np.array, eState: np.array) -> np.array:
    """
    Only needed if virial estimator is used.
    """

    return np.array([R[0][0], R[0][1], R[0][2]])

@conditional_jit
def getDiabV(R: np.array, eState: np.array) -> tuple:
    """
    Only needed in the diabatic limit. So if non_adiabatic_coupling or PoE is used.
    """

    return _HO(R), 0, 0, _HO(R)+2