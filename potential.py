"""
Define your potential here. V is calles from the main MCMC loop
"""

import numpy as np
from numba import jit, njit
from projToINRC import proj_main
import joblib
import matplotlib.pyplot as plt
import configparser
config = configparser.ConfigParser()
config.read('input.in')
use_jit = str(config["settings"]["use_jit"]) 

def conditional_jit(func):

    if use_jit == "True":
        return njit()(func)
    else:
        return func

############Some example potentials for testing#######################

@conditional_jit
def HO(R):

    V1 = .5 * (R[0]**2+R[1]**2+R[2]**2)
    V2 = .5 * (R[0]**2+R[1]**2+R[2]**2) + 2

    return V1, V2

@conditional_jit
def morsePot(r):
    """
    Parameters:
    - r: Distance between the atoms.
    - D_e: Depth of the potential well.
    - alpha: Width of the potential well.
    - r_e: Equilibrium bond distance.
    """

    # Define parameters as needed (here H2@FCI/aug-cc-pVDZ)
    D_e = 1.16637054
    alpha = 2.02881752 # 1/amstrong
    r_e = 0.76345602 # amstrong
    A = 0.17089793  # hartree

    return A * (1 - np.exp(-alpha * (r - r_e)))**2 - D_e

@conditional_jit
def GradMorsePot(r):
  """
  Analytic grad of the Morse Potential. 
  """
  alpha = 2.02881752
  r_e = 0.76345602
  A = 0.17089793

  return 2*alpha*A * ( np.exp(-alpha * (r - r_e)) - np.exp(-2*alpha * (r - r_e)) )



@conditional_jit
def getVHO(R, eState):

    V1, V2 = HO(R)

    if eState[0] == 0:
        return V1
    else: 
        return V2
    
@conditional_jit
def getVgaussProcess(R):

    scaler_loaded = joblib.load('scaler.pkl')
    gp_loaded = joblib.load('VgaussProcess.pkl')

    X_new_scaled = scaler_loaded.transform(R)
    y_pred, sigma = gp_loaded.predict(X_new_scaled, return_std=True)

    return y_pred, sigma
    
@conditional_jit
def getV(R, eState):

    R = R[0]
    V1, V2 = HO(R)

    if eState[0] == 0:
        return V1
    else: 
        return V2

@conditional_jit   
def getGradV(R, eState):

    R = R[0]
    gradV = np.array([R[0], R[1], R[2]])

    return gradV    

@conditional_jit
def getDiabV(R, eState):

    pass