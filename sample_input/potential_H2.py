"""
Define your potential here. getV functino is calles from the main MCMC loop!!!
"""

import numpy as np
from numba import njit
import configparser
import joblib

from projToINRC import proj_main


config = configparser.ConfigParser()
config.read('input.in')
use_jit = config.getboolean("PIMC", "use_jit") 

def cJIT(func):

    if use_jit:
        return njit()(func)
    else:
        return func


############Morse potential of H2@FCI/aug-cc-pvDZ#######################

@cJIT
def H2_morse(R):
   
    r = np.linalg.norm(R[0,:]) 
    alpha = 2.02881752 / 1.8897268777744
    r_e = 0.76345602 * 1.8897268777744
    A = 0.17089793

    return A * (1 - np.exp(-alpha * (r - r_e)))**2

@cJIT
def H2_morse_grad(R):

    # check if gradient is correct, test it with numerical one, then try with two particle case

    R = R[0,:]#-R[1,:]
    r = np.linalg.norm(R) 
    alpha = 2.02881752  / 1.8897268777744
    r_e = 0.76345602 * 1.8897268777744
    A = 0.17089793

    return 2 * A * alpha * (1 - np.exp(-alpha * (r - r_e))) * np.exp(-alpha * (r - r_e)) * (R/r)

############Called from main code#######################
    
@cJIT
def getV(R: np.array, eState: int) -> float:
    """
    Called from main code to get the potential energy at a given geometry R. 
    """

    return H2_morse(R)

    
@cJIT   
def getGradV(R: np.array, eState: int) -> np.array:
    """
    Only needed if virial estimator is used.
    """

    return H2_morse_grad(R)

@cJIT
def getDiabV(R: np.array) -> tuple:
    """
    Only needed in the diabatic limit. So if non_adiabatic_coupling or PoE is used.
    """

    pass