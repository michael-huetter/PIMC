"""
Define your potential in this script.
"""

import numpy as np
from numba import njit
import configparser

config = configparser.ConfigParser()
config.read('input.in')
use_jit = config.getboolean("PIMC", "use_jit") 

def cJIT(func):

    if use_jit:
        return njit()(func)
    else:
        return func


############Called from main code#######################
    
@cJIT
def getV(R: np.array, eState: int) -> float:
    """
    Called from main code to get the potential energy at a given geometry R. 
    """

    pass
    
@cJIT   
def getGradV(R: np.array, eState: int) -> np.array:
    """
    Only needed if virial estimator is used.
    """

    pass

@cJIT
def getDiabV(R: np.array) -> tuple:
    """
    Only needed in the diabatic limit. So if non_adiabatic_coupling or PoE is used.
    """

    pass