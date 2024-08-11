"""
Define your potential here. getV functino is calles from the main MCMC loop!!!
"""

import numpy as np
from numba import njit
import configparser
import joblib
from numpy import array

#from projToINRC import proj_main


config = configparser.ConfigParser()
config.read('input.in')
use_jit = config.getboolean("PIMC", "use_jit") 

def cJIT(func):

    if use_jit:
        return njit()(func)
    else:
        return func


############Morse potential of H2@FCI/aug-cc-pvDZ#######################

morse_param = array([array([1.59647928e+00, 1.37807813e+00, 3.17416753e+00, 1.73448395e-05]), 
                     array([1.59647928e+00, 1.37807813e+00, 3.17416753e+00, 1.73448395e-05]), 
                     array([1.59606121, 1.32230438, 3.37607623, 0.01963534]), 
                     array([1.59606121, 1.32230438, 3.37607623, 0.01963534]), 
                     array([1.59510584, 1.31763599, 3.3832987 , 0.04040005]), 
                     array([1.59510196, 1.31764224, 3.38330274, 0.04040255]),
                     array([1.59441336, 1.31236258, 3.39545783, 0.06414528]), 
                     array([1.59441336, 1.31236258, 3.39545783, 0.06414528]), 
                     array([1.59387018, 1.30295491, 3.4226143 , 0.09166429]), 
                     array([1.5941657 , 1.29811783, 3.45211566, 0.09297255]), 
                     array([1.58370652, 1.44800292, 2.84450431, 0.11494284]), 
                     array([1.58370652, 1.44800292, 2.84450431, 0.11494284]), 
                     array([1.58510702, 1.45382487, 2.82747049, 0.13092164]), 
                     array([1.58510179, 1.45389291, 2.82729067, 0.13092207]), 
                     array([1.58616645, 1.46285599, 2.796528  , 0.14429438]), 
                     array([1.58616645, 1.46285599, 2.796528  , 0.14429438]), 
                     array([1.58726212, 1.47126277, 2.7727506 , 0.15262919]), 
                     array([1.58320566, 1.3895963 , 3.0140491 , 0.15826923]), 
                     array([1.58320566, 1.3895963 , 3.0140491 , 0.15826923]), 
                     array([1.58553622, 1.47046469, 2.79737204, 0.15836436]), 
                     array([1.64363422, 1.42523332, 2.58619529, 0.48966874]), 
                     array([1.64363423, 1.42523255, 2.58619865, 0.48966871]), 
                     array([1.64269555, 1.42205932, 2.61216357, 0.49383906]), 
                     array([1.64269555, 1.42205932, 2.61216357, 0.49383906]), 
                     array([1.64239852, 1.421249  , 2.61948103, 0.49520889])])

@cJIT
def morse(R, eState):
   
    r = R[0,0] 
    r_e, alpha, D, A = morse_param[eState]

    return D * (1 - np.exp(-alpha * (r - r_e)))**2 + A


@cJIT
def morse_grad_0(R, eState):

    r = R[0,0] 
    r_e, alpha, D, A = morse_param[eState]

    return np.array([2 * D * alpha * (1 - np.exp(-alpha * (r - r_e))) * np.exp(-alpha * (r - r_e))])


############Called from main code#######################
    
@cJIT
def getV(R: np.array, eState: int) -> float:
    """
    Called from main code to get the potential energy at a given geometry R. 
    """

    return morse(R, eState)
    
@cJIT   
def getGradV(R: np.array, eState: int) -> np.array:
    """
    Only needed if virial estimator is used.
    """

    return morse_grad_0(R, eState)

@cJIT
def getDiabV(R: np.array) -> tuple:
    """
    Only needed in the diabatic limit. So if non_adiabatic_coupling or PoE is used.
    """

    pass