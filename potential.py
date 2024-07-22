"""
Define your potential here. getV functino is calles from the main MCMC loop!!!
"""

import numpy as np
from numba import jit, njit
from projToINRC import proj_main
import joblib
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
def V_H2(R):
    """
    Morse potential for H2 calculated at the FCI/aug-cc-pVDZ level of theory. In atomic units.
    """

    r =  np.sqrt(R[0][0]**2 + R[0][1]**2 + R[0][2]**2)
    D_e = 1.16637054
    alpha = 3.8339091936
    r_e = 1.4427226821
    A = 0.17089793
    
    return A * (1 - np.exp(-alpha * (r - r_e)))**2 - D_e



############Called from main code#######################
    
@conditional_jit
def getV(R, eState):
    """
    Called from main code to get the potential energy at a given geometry R. 
    """

    return V_H2(R)
    
@conditional_jit   
def getGradV(R, eState):
    """
    Only needed if virial estimator is used.
    """

    r = np.sqrt(R[0][0]**2 + R[0][1]**2 + R[0][2]**2)
    alpha = 3.8339091936
    r_e = 1.4427226821
    A = 0.17089793
    
    return 2*A*alpha*np.exp(-alpha * (r - r_e))*( 1 - np.exp(-alpha * (r - r_e)) ) * np.array([R[0][0]/r, R[0][1]/r, R[0][2]/r])

@conditional_jit
def getDiabV(R, eState):
    """
    Only needed in the diabatic limit.
    """

    pass

@conditional_jit
def getVgaussProcess(R):
    """
    If a gaussian process regression model is used to fit the PES.
    """

    scaler_loaded = joblib.load('scaler.pkl')
    gp_loaded = joblib.load('VgaussProcess.pkl')

    X_new_scaled = scaler_loaded.transform(R)
    y_pred, sigma = gp_loaded.predict(X_new_scaled, return_std=True)

    return y_pred, sigma