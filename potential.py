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


############Called from main code#######################
    
@conditional_jit
def getV(R: np.array, eState: int) -> float:
    """
    Called from main code to get the potential energy at a given geometry R. 
    """

    pass
    
@conditional_jit   
def getGradV(R: np.array, eState: int) -> np.array:
    """
    Only needed if virial estimator is used.
    """

    pass

@conditional_jit
def getDiabV(R: np.array) -> tuple:
    """
    Only needed in the diabatic limit. So if non_adiabatic_coupling or PoE is used.
    """

    pass