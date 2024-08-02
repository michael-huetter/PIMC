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

############Called from main code#######################
    
@cJIT
def getV(R: np.array, eState: int) -> float:

    return 0.5 * R[0,0]**2

@cJIT   
def getGradV(R: np.array, eState: int) -> np.array:
    
    return np.array([R[0,0]])  

@cJIT
def getDiabV(R: np.array) -> tuple:
    
    pass