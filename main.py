import os
from concurrent.futures import ProcessPoolExecutor
import ctypes
import functools
import logging 

import numpy as np
from numba import njit, typed
import configparser
from tqdm import tqdm

from helpers import save_to_csv, wOut, remove_all_files_in_folder
from potential import getV, getGradV, getDiabV
from stats import rcc
from readGeom import getGeom

"""
Read input parameters-----------------------------------------------------------------
"""

config = configparser.ConfigParser()
config.read('input.in')

numParticles = config.getint("system", "numParticles")  
lam = str(config["system"]["lam"]); lam_list = lam.split(','); lam = [float(item) for item in lam_list]  

numMCSteps = config.getint("PIMC", "numMCSteps")
staging = config.getboolean("PIMC", "staging") 
m = config.getint("PIMC", "m") # stage length
numTimeSlices =  config.getint("PIMC", "numTimeSlices")
delta = config.getfloat("PIMC", "delta") # CoM displacement
delta_bead = config.getfloat("PIMC", "delta_bead") 
use_jit = config.getboolean("PIMC", "use_jit") 
echange = config.getboolean("PIMC", "echange") 
non_adiabatic_coupling = config.getboolean("PIMC", "non_adiabatic_coupling") 
PoE = config.getboolean("PIMC", "PoE")
rand_seed = config.getint("PIMC", "rand_seed")
kinVirial = config.getboolean("PIMC", "kin_virial") 

corrSkip = config.getint("convergence", "corrSkip")
thermSkip = config.getint("convergence", "thermSkip")

log_flag = config.getboolean("debug", "logging")

try:
    # Load shared library
    lib = ctypes.CDLL('./lib_PoE.so')
    lib.initialize_random_seed()

    lib.performPoE.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, 
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
    ]
except:
    wOut("Warning: Could not load shared library for PoE.")

# Define jit decorator
def cJIT(func):

    if use_jit:
        return njit()(func)
    else:
        return func
    
# Configure logging and define logging decorator
logging.basicConfig(
    filename='debug.log',  
    filemode='w',  
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
def log(func):

    if log_flag:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logging.info(f"Running {func.__name__} with args: {args} and kwargs: {kwargs}")
            try:
                result = func(*args, **kwargs)
                logging.info(f"{func.__name__} completed successfully with result: {result}")
                return result
            except Exception as e:
                logging.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                raise  
        return wrapper
    else:
        return func

"""
Functions to compute Potential Energy/Action-----------------------------------------------------------------
"""

@cJIT
def getEig(V: np.array) -> np.array:
    
    eigValues, eigVectors = np.linalg.eig(V)
    S = eigVectors

    return S

@cJIT
def get_phi(beads: np.array, eState: np.array, numTimeSlices: int, n: int) -> float:
    """
    Currnetly only implementet for diatomics       
    """

    S = []
    for j in range(numTimeSlices):
        R = beads[j,0:]
        V_result = getDiabV(R, eState) 
        if isinstance(V_result, float):
            V_tot = np.full((n * n,), V_result)
        else:
            V_tot = np.array(V_result)
        V = V_tot.reshape((n, n))
        S_i = getEig(V)
        S.append(S_i)
    
    phi = 1

    for i in range(len(S)):

        current_vector = S[i][:, eState[i]]
        next_vector = S[(i+1) % len(S)][:, eState[(i+1) % len(S)]]   # korrekt soo?

        dot = np.dot(current_vector, next_vector)

        phi *= dot

    return phi

@cJIT
def potEnergy(beads: np.array, numTimeSlices: int, eState: np.array) -> float:

    PE = 0.0
    for j in range(numTimeSlices):

        R = beads[j,0:]
        V_result = getV(R, eState)
        PE = PE + V_result

    return PE/(numTimeSlices) 

@cJIT
def potAction(beads: np.array, tau: float, numTimeSlices: int, n: int, eState: np.array) -> float:

    PE = 0.0
    for j in range(numTimeSlices):

        R = beads[j,0:]
        V_result = getV(R, eState)
        PE = PE + V_result

    if non_adiabatic_coupling:
        phi = get_phi(beads, eState, numTimeSlices, n)
    else:
        phi = 1
    
    return PE*tau - np.log(np.abs(phi))/tau


"""
Kinetic Energy-----------------------------------------------------------------
"""

@cJIT
def kinetic_estimator(beads: np.array, tau: float, lam: float, numTimeSlices: int, numParticles: int) -> float:
    """
    Thermodynamic estimator for the kinetic energy. 
    """

    tot = 0.0
    for tslice in range(numTimeSlices):
        tslicep1 = (tslice + 1) % numTimeSlices
        for ptcl in range(numParticles):
            norm = 1.0/(4.0*lam[ptcl]*tau*tau)
            delR = beads[tslicep1,ptcl] - beads[tslice,ptcl]
            tot = tot - norm*np.dot(delR, delR)
        
    return (3/2) * numParticles/tau + tot/numTimeSlices

@cJIT
def kinetic_action(beads: np.array, tau: float, lam: float, numTimeSlices: int, numParticles: int) -> float:
    """
    Kinetic action. 
    """

    tot = 0.0
    for tslice in range(numTimeSlices):
        tslicep1 = (tslice + 1) % numTimeSlices
        for ptcl in range(numParticles):
            norm = 1.0/(4.0*lam[ptcl]*tau*tau)
            delR = beads[tslicep1,ptcl] - beads[tslice,ptcl]
            tot = tot + norm*np.dot(delR, delR)
        
    return tot*tau 

@cJIT
def virial_estimator(beads: np.array, tau: float, numTimeSlices: int, numParticles: int, eState: np.array) -> float:
    """
    Virial estimator for the kinetic energy.
    """

    tot = 0.0
    for tslice in range(numTimeSlices):
        for ptcl in range(numParticles):
            Rc = (1/numTimeSlices) * np.sum(beads[:, ptcl, :], axis=0)
            delR = beads[tslice,ptcl] - Rc
            dVdR = getGradV(beads[tslice,:], eState)
            tot = tot + np.dot(delR, dVdR)

    tot = tot * 1/(2*numTimeSlices)

    return (3*numParticles)/(2*tau*numTimeSlices) + tot

"""
Some further used estimators  -----------------------------------------------------------------
"""

@cJIT
def bond_length(beads: np.array, numTimeSlices: int) -> float:
    """
    Compute bond length for diatomic molecules.
    """
    
    x = 0
    for j in range(numTimeSlices):
        x +=  np.sqrt( beads[j][0][0]**2 + beads[j][0][1]**2 + beads[j][0][2]**2) 

    return x/numTimeSlices

@cJIT
def bead_pos(beads: np.array, numTimeSlices: int) -> tuple[float, float, float]:

    x, y, z = 0, 0, 0
    for j in range(numTimeSlices):
        x +=  beads[j][0][0]
        y +=  beads[j][0][1]
        z +=  beads[j][0][2]

    return x/numTimeSlices, y/numTimeSlices, z/numTimeSlices

    
"""
Implementation of diffrent MC steps  -----------------------------------------------------------------
"""

@log
@cJIT
def center_of_mass_move(beads: np.array, ptcl: int, tau: float, delta: float, numTimeSlices: int, n: int, eState: np.array) -> tuple[np.ndarray, bool]:

    shift = delta*(2.0*np.random.random(3) - 1.0)
    
    beads_new = np.copy(beads)

    beads_new[:,ptcl] += shift
    
    oldAction = potAction(beads, tau, numTimeSlices, n, eState)
    newAction = potAction(beads_new, tau, numTimeSlices, n, eState)
    
    if np.random.random() < np.exp(-(newAction - oldAction)):
        return beads_new, True
    else:
        return beads, False

@log
@cJIT
def bead_move(beads: np.array, ptcl: int, tau: float, delta: float, numTimeSlices: int, lam: float, numParticles: int, n: int, eState: np.array) -> tuple[np.ndarray, bool]:
    """
    Individual bead move. Used if staging is turned off.
    """

    shift = delta*(2.0*np.random.random(3) - 1.0)

    beads_new = np.copy(beads)
    
    rand_tslice = np.random.randint(0, numTimeSlices)
    beads_new[rand_tslice,ptcl] += shift

    oldAction = potAction(beads, tau, numTimeSlices, n, eState) + kinetic_action(beads, tau, lam, numTimeSlices, numParticles)
    newAction = potAction(beads_new, tau, numTimeSlices, n, eState) + kinetic_action(beads_new, tau, lam, numTimeSlices, numParticles)

    if np.random.random() < np.exp(-(newAction - oldAction)):
        return beads_new, True
    else:
        return beads, False

@log  
@cJIT
def staging_move(beads: np.array, ptcl: int, tau: float, lam: float, numTimeSlices: int, m: int, n: int, eState: np.array) -> tuple[np.ndarray, bool]:

    beads_new = np.copy(beads)
    
    alpha_start = np.random.randint(0,numTimeSlices)
    alpha_end = (alpha_start + m) % numTimeSlices    
    
    for a in range(1,m):
        tslice = (alpha_start + a) % numTimeSlices
        tslicem1 = (tslice - 1) % numTimeSlices
        tau1 = (m-a)*tau
        avex = (tau1*beads[tslicem1,ptcl] + tau*beads[alpha_end,ptcl]) / (tau + tau1)
        sigma2 = 2.0*lam[ptcl] / (1.0 / tau + 1.0 / tau1)
        beads_new[tslice,ptcl] = avex + np.sqrt(sigma2)*np.random.randn()

    oldAction = potAction(beads, tau, numTimeSlices, n, eState)
    newAction = potAction(beads_new, tau, numTimeSlices, n, eState)

    if np.random.random() < np.exp(-(newAction - oldAction)):
        return beads_new, True
    else:
        return beads, False

@log    
def global_e_change(beads: np.array, tau: float, numTimeSlices: int, n: int, eState: np.array) -> tuple[np.ndarray, bool]:
    """
    Change the electronic state of the system globally. Adiabatic limit. 
    """

    old_eState = np.copy(eState)
    new_eState = np.copy(eState)
    
    possible_states = np.setdiff1d(np.arange(n), old_eState[0])
    rand_e_state = np.random.choice(possible_states)
    new_eState = np.full_like(eState, rand_e_state)

    oldAction = potAction(beads, tau, numTimeSlices, n, old_eState)
    newAction = potAction(beads, tau, numTimeSlices, n, new_eState)

    if np.random.random() < np.exp(-(newAction - oldAction)):
        return new_eState, True
    else:
        return old_eState, False

@log
def local_e_change(beads: np.array, tau: float, numTimeSlices: int, n: int, eState: np.array) -> tuple[np.ndarray, bool]:
    """
    Individual bead moves. Diabatic limit.
    """

    old_eState = eState
    new_eState = eState
  
    to_move = np.random.randint(0, numTimeSlices)
    possible_states = list(range(n)) 
    possible_states = [i for i in possible_states if i != old_eState[to_move]]
    rand_e_state = np.random.choice(possible_states)
    new_eState[to_move] = rand_e_state
    
    oldAction = potAction(beads, tau, numTimeSlices, n, old_eState)
    newAction = potAction(beads, tau, numTimeSlices, n, new_eState)

    if np.random.random() < np.exp(-(newAction - oldAction)):
        return new_eState, True
    else:
        return old_eState, False

@log
def PoE_move(eState: np.array, xi: int, xi_change_interval: int, numTimeSlices: int, n: int, k: int, xi_possible: np.array, beads: np.array, tau: float) -> tuple[np.array, int]:

    xi_old = xi
    eState_old = np.copy(eState)
    
    if k % xi_change_interval == 0:
        xi = np.random.choice(xi_possible)
    
    xi_possible = xi_possible.astype(np.int32)
    xi_current = ctypes.c_int(0)

    xi_possible_ptr = xi_possible.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    eState_ptr = eState.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    while np.array_equal(eState, eState_old): 
        lib.performPoE(k, xi, xi_change_interval, xi_possible_ptr, xi_possible.size, numTimeSlices, n, eState_ptr, ctypes.byref(xi_current))  

    newAction = potAction(beads, tau, numTimeSlices, n, eState)
    oldAction = potAction(beads, tau, numTimeSlices, n, eState_old)

    if np.random.random() < np.exp(-(newAction - oldAction)):
        return eState, xi_current.value
    else:
        return eState_old, xi_old


"""
Main MC loop  -----------------------------------------------------------------
"""


def MCMC(numSteps, beads, tau, lam, delta, m, numTimeSlices, numParticles, n, echange, eState, eCL, eCG):
    
    EnergyTrace = []
    PositionTrace = []
    eStateTrace = []
    xiTrace = []
    numAccept = {"CoM":0, "Staging":0, "Bead": 0, "eChange": 0}

    if use_jit:
        lam = typed.List(lam)
    
    # only needed if PoE is turned on
    xi_possible = np.arange(0, numTimeSlices + 1, 2) if numTimeSlices % 2 == 0 else np.arange(0, numTimeSlices, 2)
    wOut(f"xi_possible: {xi_possible}")
    xi = 0
    xi_change_interval = eCL

    for k in tqdm(range(numSteps), ascii=" >=", desc="MC steps"):
        
        # sample potential and kinetic action with CoM and staging/individual bead moves
        for ptcl in np.random.randint(0,numParticles,numParticles):
                beads, accept = center_of_mass_move(beads, ptcl, tau, delta, numTimeSlices, n, eState)
                numAccept["CoM"] += accept
        if staging:
            for ptcl in np.random.randint(0,numParticles,numParticles): 
                beads, accept = staging_move(beads, ptcl, tau, lam, numTimeSlices, m, n, eState)
                numAccept["Staging"] += accept
        if not staging:
            for ptcl in np.random.randint(0,numParticles,numParticles):
                beads, accept = bead_move(beads, ptcl, tau, delta_bead, numTimeSlices, lam, numParticles, n,eState)
                numAccept["CoM"] += accept
        
        # sample multiple electronic states
        if echange and not PoE:
            if k % eCG == 0:
                eState, accept = global_e_change(beads, tau, numTimeSlices, n, eState)
                numAccept["eChange"] += accept
            if non_adiabatic_coupling:
                if k % eCL == 0:
                    eState = local_e_change(beads, tau, numTimeSlices, n, eState)

        if PoE:
            eState, xi = PoE_move(eState, xi, xi_change_interval, numTimeSlices, n, k, xi_possible, beads, tau)
            #print(f"({k})", "xi: ", xi, " <-> eState: ", eState)
            

        # keep track of observables
        if k % corrSkip == 0 and k > thermSkip:
            potE = potEnergy(beads, numTimeSlices, eState)
            if kinVirial:
                kinEvirial = virial_estimator(beads, tau, numTimeSlices, numParticles, eState)
                kinEthermo = kinetic_estimator(beads, tau, lam, numTimeSlices, numParticles)
                EnergyTrace.append([potE, kinEthermo, kinEvirial])
            else:
                kinEthermo = kinetic_estimator(beads, tau, lam, numTimeSlices, numParticles)
                EnergyTrace.append([potE, kinEthermo])
            eStateTrace.append(np.copy(eState))
            xiTrace.append([xi])
            if numParticles == 1:
                PositionTrace.append(bead_pos(beads, numTimeSlices))
            else:
                PositionTrace.append(beads)

    return np.array(PositionTrace), np.array(EnergyTrace), numAccept, np.array(eStateTrace), np.array(xiTrace)

"""
Run multiple T loops in parallel  -----------------------------------------------------------------
"""

def main(T, n, echange, eCL, eCG):
    
    tau = 1.0/(T*numTimeSlices)

    # beads[bead][particl][coordinate]
    beads = np.zeros([numTimeSlices,numParticles, 3])

    if numParticles == 1:
        # If only one particle randomly distribute them
        for tslice in range(numTimeSlices):
            for ptcl in range(numParticles):
                beads[tslice,ptcl] = 0.5*(-1.0 + 2.0*np.random.random(3)) # [-0.5, 0.5)
    elif numParticles > 1:
        # If more than one particle is use distribute them according to the input geometry in geom.in
        atoms, coord = getGeom()
        wOut(f"({T}) Input geometry:\n {coord}")
        for i in range(numParticles):
            for j in range(numTimeSlices):
                beads[j, i] = coord[i] + np.random.rand(3)*0.1

    # initialize e-states (cold start, may also be tried differantly) 
    eState = np.zeros(numTimeSlices, dtype=np.int32)    

    Position, Energy, numAccept, eState, xiTrace = MCMC(numMCSteps, beads, tau, lam, delta, m, numTimeSlices, numParticles, n, echange, eState, eCL, eCG)
    
    return Energy, Position, eState, numAccept, xiTrace

def worker(args):

    i, n, echange, eCL, eCG = args
    Energy, Position, eState, numAccept, xiTrace = main(i, n, echange, eCL, eCG)

    save_to_csv(Energy[:,0], f'{i}_PotEnergyTrace.csv')

    if kinVirial:
        save_to_csv(Energy[:,2], f'{i}_KinEnergyTrace.csv')
        save_to_csv(Energy[:,1], f'{i}_KinThermoEnergyTrace.csv')
    else:
        save_to_csv(Energy[:,1], f'{i}_KinEnergyTrace.csv')

    save_to_csv(Position, f'{i}_PositionTrace.csv')
    save_to_csv(eState, f'{i}_eStatTrace.csv')
    save_to_csv(xiTrace, f'{i}_xiTrace.csv')

    for key in numAccept:
        numAccept[key] /= numMCSteps*numParticles

    wOut(f"({i}) numAccept: {numAccept}")
   
def parallel_main(T, n, echange, eCL, eCG):

    with ProcessPoolExecutor() as executor:
        executor.map(worker, [(i, n, echange, eCL, eCG) for i in T])

"""
Initialize  -----------------------------------------------------------------
"""

if __name__ == "__main__": 
    
    remove_all_files_in_folder("output")

    if os.path.exists("output.out"):
        os.remove("output.out")

    # Read in temperature loops
    T = str(config["system"]["T"]) 
    T_list = T.split(',')
    T = [float(item) for item in T_list]

    # Write some input parameters to output file
    wOut(f"PIMC V1.1")
    wOut(f"Avalible CPUs: {os.cpu_count()}")
    wOut(f"Used CPUs: {len(T)}")
    wOut(f"JIT compiler: {use_jit}")
    wOut(f"Virial estimator: {kinVirial}")
    wOut(f"T: {T}")
    wOut(f"lam: {lam}")
    wOut(f"rand_seed: {rand_seed}")
    wOut(f"numParticles: {numParticles}")
    wOut(f"numTimeSlices: {numTimeSlices}")
    wOut(f"Stage length: {m}")
    wOut(f"Delta (CoM): {delta}")
    
    if not staging:
        wOut(f"Delta (Bead): {delta_bead}")
        wOut(f"Warning: Staging is turned off")

    n = config.getint("system", "n") # dimension of the potetnial matrix (nxn) 
    
    match non_adiabatic_coupling, PoE:
        case True, False:
            eCL = config.getint("convergence", "eCL") # local e change
            eCG = config.getint("convergence", "eCG") # global e change
            wOut(f"Non-adiabatic coupling True: Diabatic limit (ecL: {eCL}, eCG: {eCG})")
        case False, True:
            wOut("Warning: Non-adiabatic couplings should be turned on for PoE")
            eCL = config.getint("convergence", "eCL") # local e change
            eCG = config.getint("convergence", "eCG") # global e change
            wOut(f"Non-adiabatic coupling True: Diabatic limit (ecL: {eCL}, eCG: {eCG})")
        case False, False:
            eCL = np.inf
            eCG =config.getint("convergence", "eCG") 
            wOut(f"Only global e-changes: Adiabatic limit (ecL: {eCL}, eCG: {eCG})")
        case True, True:
            eCL = config.getint("convergence", "eCL") 
            eCG = np.inf
            wOut(f"PoE True: Diabatic limit (ecL: {eCL}, eCG: {eCG})")
    
    # sanitiy check of some input parameters
    if numParticles > 1 and (non_adiabatic_coupling or PoE):
        wOut(f"Error: Non-adiabatic coupling and PoE are only implemented for one particle atm")
        exit() 
    if thermSkip < 1000:
        wOut(f"Warning: thermSkip is very low")
    if corrSkip < 10:
        wOut(f"Warning: corrSkip is very low")
    if echange and n<2:
        wOut(f"Error: echange is turned on but n < 2")
        exit(f"Error: echange is turned on but n < 2")
    
    # run PICM simulations
    np.random.seed(rand_seed)
    parallel_main(T, n, echange, eCL, eCG)
    # some basic statistics on the output is written to output.out
    rcc()