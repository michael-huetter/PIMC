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
def C2_morse_0(R):
   
    r = np.linalg.norm(R[0,:]) 
    alpha = 2.86378259 / 1.8897268777744
    r_e = 1.23899468 * 1.8897268777744
    A = 0.17943589
    De = 75.80462248

    return A * (1 - np.exp(-alpha * (r - r_e)))**2 - De

@cJIT
def C2_morse_1(R):
   
    r = np.linalg.norm(R[0,:]) 
    alpha = 2.05596914 / 1.8897268777744
    r_e = 1.31764468 * 1.8897268777744
    A = 0.25420046
    De = 75.80171396

    return A * (1 - np.exp(-alpha * (r - r_e)))**2 - De

@cJIT
def H2_morse_grad_0(R):

    r = np.linalg.norm(R[0,:]) 
    alpha = 2.86378259 / 1.8897268777744
    r_e = 1.23899468 * 1.8897268777744
    A = 0.17943589

    return 2 * A * alpha * (1 - np.exp(-alpha * (r - r_e))) * np.exp(-alpha * (r - r_e)) * (R[0,:]/r)

@cJIT
def H2_morse_grad_1(R):

    r = np.linalg.norm(R[0,:]) 
    alpha = 2.05596914 / 1.8897268777744
    r_e = 1.31764468 * 1.8897268777744
    A = 0.25420046

    return 2 * A * alpha * (1 - np.exp(-alpha * (r - r_e))) * np.exp(-alpha * (r - r_e)) * (R[0,:]/r)

############Called from main code#######################
    
@cJIT
def getV(R: np.array, eState: int) -> float:
    """
    Called from main code to get the potential energy at a given geometry R. 
    """

    if eState == 0:
        return C2_morse_0(R)
    else:
        return C2_morse_1(R)
    
@cJIT   
def getGradV(R: np.array, eState: int) -> np.array:
    """
    Only needed if virial estimator is used.
    """

    if eState == 0:
        return H2_morse_grad_0(R)
    else:
        return H2_morse_grad_1(R)

@cJIT
def getDiabV(R: np.array) -> tuple:
    """
    Only needed in the diabatic limit. So if non_adiabatic_coupling or PoE is used.
    """

    pass