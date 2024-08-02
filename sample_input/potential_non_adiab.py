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

    R = R[0,:]
    r = np.linalg.norm(R)
    k11 = 4*10**-5
    k22 = 3.2*10**-5
    r11 = 7
    r22 = 10.5
    e11 = 0
    e22 = 2.2782*10**-5
    c = 5*10**-5
    a = 0.4
    r12 = 8.75

    V11 = .5 * k11 * (r-r11)**2 + e11
    V22 = .5 * k22 * (r-r22)**2 + e22
    V12 = c*np.exp(-a*(r-r12)**2)

    H = np.array([[V11, V12], [V12, V22]])
    w, v = np.linalg.eigh(H)
    V_ad_1, V_ad_2 = w

    if eState == 0:
        return V_ad_1
    else:
        return V_ad_2   

@cJIT   
def getGradV(R: np.array, eState: int) -> np.array:

    pass

@cJIT
def getDiabV(R: np.array) -> tuple:
    
    R = R[0,:]
    r = np.linalg.norm(R)
    k11 = 4*10**-5
    k22 = 3.2*10**-5
    r11 = 7
    r22 = 10.5
    e11 = 0
    e22 = 2.2782*10**-5
    c = 5*10**-5
    a = 0.4
    r12 = 8.75

    V11 = .5 * k11 * (r-r11)**2 + e11
    V22  = .5 * k22 * (r-r22)**2 + e22
    V12 = c*np.exp(-a*(r-r12)**2)

    return V11, V12, V12, V22