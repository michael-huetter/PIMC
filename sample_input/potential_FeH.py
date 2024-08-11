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

morse_param = array([array([1.59647928e+00, 1.37807813e+00, 1.17561760e-01, 6.42480222e-07]), 
                     array([1.59647928e+00, 1.37807813e+00, 1.17561760e-01, 6.42480222e-07]), 
                     array([1.59606121e+00, 1.32230436e+00, 1.25039863e-01, 7.27234959e-04]), 
                     array([1.59606121e+00, 1.32230436e+00, 1.25039863e-01, 7.27234959e-04]), 
                     array([1.59510584e+00, 1.31763597e+00, 1.25307363e-01, 1.49629798e-03]), 
                     array([1.59510196e+00, 1.31764223e+00, 1.25307511e-01, 1.49639062e-03]), 
                     array([1.59441336, 1.31236257, 0.1257577 , 0.00237575]), 
                     array([1.59441336, 1.31236257, 0.1257577 , 0.00237575]), 
                     array([1.59387019, 1.30295489, 0.1267635 , 0.00339497]), 
                     array([1.5941657 , 1.29811781, 0.12785614, 0.00344343]), 
                     array([1.58370652, 1.44800293, 0.10535201, 0.00425714]), 
                     array([1.58370652, 1.44800293, 0.10535201, 0.00425714]), 
                     array([1.58510702, 1.45382487, 0.10472113, 0.00484895]), 
                     array([1.58510179, 1.45389291, 0.10471447, 0.00484897]), 
                     array([1.58616645, 1.46285599, 0.10357511, 0.00534424]), 
                     array([1.58616645, 1.46285599, 0.10357511, 0.00534424]), 
                     array([1.58726212, 1.47126278, 0.10269447, 0.00565293]), 
                     array([1.58320566, 1.3895963 , 0.11163145, 0.00586182]), 
                     array([1.58320566, 1.3895963 , 0.11163145, 0.00586182]), 
                     array([1.58553622, 1.47046469, 0.10360637, 0.00586535]), 
                     array([1.64363422, 1.42523332, 0.09578501, 0.01813588]), 
                     array([1.64363423, 1.42523255, 0.09578514, 0.01813588]), 
                     array([1.64269555, 1.42205932, 0.0967468 , 0.01829034]), 
                     array([1.64269555, 1.42205932, 0.0967468 , 0.01829034]), 
                     array([1.64239852, 1.421249  , 0.09701782, 0.01834107])])

@cJIT
def morse(R, eState):
   
    r = R[0,0] 
    r_e, alpha, D, A = morse_param[eState]

    r_e *= 1.8897268777744
    alpha /= 1.8897268777744

    return D * (1 - np.exp(-alpha * (r - r_e)))**2 + A


@cJIT
def morse_grad_0(R, eState):

    r = R[0,0] 
    r_e, alpha, D, A = morse_param[eState]

    r_e *= 1.8897268777744
    alpha /= 1.8897268777744

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