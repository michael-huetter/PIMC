import numpy as np
from numba import jit, njit, typed
from concurrent.futures import ProcessPoolExecutor
import os
from helpers import save_to_csv, write_debug_log, remove_all_files_in_folder
import configparser
from readGeom import getGeom
from stats import rcc
from potential import getV, getGradV, getDiabV
from tqdm import tqdm



"""
Read input parameters -----------------------------------------------------------------
"""

config = configparser.ConfigParser()
config.read('input.in')
echange = str(config["settings"]["echange"]) 
non_adiabatic_coupling = str(config["settings"]["non_adiabatic_coupling"]) 
lam = str(config["settings"]["lam"]); lam_list = lam.split(','); lam = [float(item) for item in lam_list]
numParticles = int(config["settings"]["numParticles"])    
numMCSteps = int(config["settings"]["numMCSteps"])
corrSkip = int(config["settings"]["corrSkip"])
thermSkip = int(config["settings"]["thermSkip"])
rand_seed = int(config["settings"]["rand_seed"])
staging = str(config["settings"]["staging"]) 
delta = float(config["settings"]["delta"]) # CoM displacement
m = int(config["settings"]["m"]) # stage length
numTimeSlices =  int(config["settings"]["numTimeSlices"])
delta_bead = float(config["settings"]["delta_bead"]) 
use_jit = str(config["settings"]["use_jit"]) 
kinVirial = str(config["settings"]["kin_virial"]) 
PoE = str(config["settings"]["PoE"])

def conditional_jit(func):

    if use_jit == "True":
        return njit()(func)
    else:
        return func

"""
Potential Energy-----------------------------------------------------------------
"""

def expm(M):
    """
    Matrix exponential (M needs to be diagonal!!!)
    """
    
    eM = np.exp(np.diag(M))
    
    return np.diag(eM)

def diag(V, tau):
    """
    Diagonalize the diabatic-matrix
    V = diabatic matrix (should be symmetric!!!)
    """
    
    eigValues, eigVectors = np.linalg.eig(V)
    G = np.diag(eigValues)
    S = eigVectors

    return S.T @ expm(-G*tau) @ S

def getEig(V):
    """
    Diagonalize the diabatic-matrix
    V = diabatic matrix (should be symmetric!!!)
    """
    
    eigValues, eigVectors = np.linalg.eig(V)
    S = eigVectors

    return S

def get_phi(beads, eState, numTimeSlices, n):

    S = []
    for j in range(numTimeSlices):
        
        R = beads[j,0:]
        V_result = getDiabV(R)         # <-----
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
        next_vector = S[(i+1) % len(S)][eState[(i+1) % len(S)]]
        dot = np.dot(current_vector, next_vector)
        phi *= dot

    return phi

@conditional_jit
def potEnergy(beads, tau, numTimeSlices, n, eState):
    """
    Include non-adiabatic effects trough a thermally averaged mean-field potential.
    """

    PE = 0.0
    for j in range(numTimeSlices):

        R = beads[j,0:]
        V_result = getV(R, eState)
        PE = PE + V_result

    return PE/(numTimeSlices) 

@conditional_jit
def potAction(beads, j_min, j_max, tau, numTimeSlices, n, eState):

    PE = 0.0
    for j in range(numTimeSlices):

        R = beads[j,0:]
        V_result = getV(R, eState)
        PE = PE + V_result

    if non_adiabatic_coupling == "True":
        phi = get_phi(beads, eState, numTimeSlices, n)
    else:
        phi = 1
    
    return PE*tau - np.log(np.abs(phi))/tau


"""
Kinetic Energy-----------------------------------------------------------------
"""

@conditional_jit
def kinetic_estimator(beads, tau, lam, numTimeSlices, numParticles):
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

@conditional_jit
def kinetic_action(beads, tau, lam, numTimeSlices, numParticles):
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

@conditional_jit
def virial_estimator(beads, tau, lam, numTimeSlices, numParticles, eState):
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
Further estimators  -----------------------------------------------------------------
"""
@conditional_jit
def beadPos(beads, numTimeSlices):
    """
    Estimator for the binding lengt.
    """
    
    x = 0
    for j in range(numTimeSlices):
        x +=  np.sqrt( beads[j][0][0]**2 + beads[j][0][1]**2 + beads[j][0][2]**2) 
    return x/numTimeSlices
    
"""
MC steps  -----------------------------------------------------------------
"""

@conditional_jit
def center_of_mass_move(beads, ptcl, tau, delta, numTimeSlices, n, eState):
    """
    Center of mass update. (displacing an entire particle worldline)
    delta: controle the acceptance ratio of a center of mass move
    """

    shift = delta*(2.0*np.random.random(3) - 1.0)

    oldbeads = np.copy(beads[:,ptcl])
    oldAction = potAction(beads, 1, numTimeSlices, tau, numTimeSlices, n, eState)

    for tslice in range(numTimeSlices):
        beads[tslice,ptcl] = oldbeads[tslice] + shift

    
    newAction = potAction(beads, 1, numTimeSlices, tau, numTimeSlices, n, eState)
    
    if np.random.random() < np.exp(-(newAction - oldAction)):
        return True
    else:
        beads[:,ptcl] = np.copy(oldbeads)
        return False

@conditional_jit
def bead_move(beads, ptcl, tau, delta, numTimeSlices, lam, numParticles,n, eState):
    """
    Individual bead move
    """

    shift = delta*(2.0*np.random.random(3) - 1.0)

    oldbeads = np.copy(beads[:,ptcl])
    oldAction = potAction(beads, 1, numTimeSlices, tau, numTimeSlices, n, eState) + kinetic_action(beads, tau, lam, numTimeSlices, numParticles)
    
    to_move = np.random.randint(0, numTimeSlices)
    beads[to_move,ptcl] = oldbeads[to_move] + shift
  
    newAction = potAction(beads, 1, numTimeSlices, tau, numTimeSlices, n, eState) + kinetic_action(beads, tau, lam, numTimeSlices, numParticles)

    if np.random.random() < np.exp(-(newAction - oldAction)):
        return True
    else:
        beads[:,ptcl] = np.copy(oldbeads)
        return False
    
@conditional_jit
def staging_move(beads, ptcl, tau, lam, numTimeSlices, m, n, eState):
    """
    Staging move, which exactly samples the free particle propagator between two positions using the Levy construction.
    m: controle acceptance ratio of staging moves. m<P!!!
    """

    alpha_start = np.random.randint(0,numTimeSlices)
    alpha_end = (alpha_start + m) % numTimeSlices

    oldbeads = np.zeros((m-1, 3))
    for a in range(1,m):
        tslice = (alpha_start + a) % numTimeSlices
        oldbeads[a-1] = beads[tslice,ptcl]    
    oldAction = potAction(beads, 1, numTimeSlices, tau, numTimeSlices, n, eState) # is 1 correct?
    
    for a in range(1,m):
        tslice = (alpha_start + a) % numTimeSlices
        tslicem1 = (tslice - 1) % numTimeSlices
        tau1 = (m-a)*tau
        avex = (tau1*beads[tslicem1,ptcl] + tau*beads[alpha_end,ptcl]) / (tau + tau1)
        sigma2 = 2.0*lam[ptcl] / (1.0 / tau + 1.0 / tau1)
        beads[tslice,ptcl] = avex + np.sqrt(sigma2)*np.random.randn()

    newAction = potAction(beads, 1, numTimeSlices, tau, numTimeSlices, n, eState)

    if np.random.random() < np.exp(-(newAction - oldAction)):
        return True
    else:
        for a in range(1,m):
            tslice = (alpha_start + a) % numTimeSlices
            beads[tslice,ptcl] = oldbeads[a-1]
        return False
    
def e_change(beads, ptcl, tau, numTimeSlices, n, eState):
    """
    Include multiple PES trough changing the e-state of all beads simultanuously -> adiabatic!!!
    """

    old_eState = eState
    new_eState = eState
    oldAction = potAction(beads, 1, numTimeSlices, tau, numTimeSlices, n, old_eState)
  
    possible_states = np.setdiff1d(np.arange(n), old_eState[0])
    rand_e_state = np.random.choice(possible_states)
    new_eState = np.full_like(eState, rand_e_state)

    newAction = potAction(beads, 1, numTimeSlices, tau, numTimeSlices, n, new_eState)

    if np.random.random() < np.exp(-(newAction - oldAction)):
        return new_eState, True
    else:
        return old_eState, False

def nonAdiab(beads, ptcl, tau, numTimeSlices, n, eState):
    """
    Include non-adiabatic effects trough moving the beads individually between PES
    """

    old_eState = eState
    new_eState = eState
    oldAction = potAction(beads, 1, numTimeSlices, tau, numTimeSlices, n, old_eState)
  
    to_move = np.random.randint(0, numTimeSlices)
    possible_states = list(range(n)) 
    # remove old_estate
    possible_states = [i for i in possible_states if i != old_eState[to_move]]
    rand_e_state = np.random.choice(possible_states)
    new_eState[to_move] = rand_e_state
    
    newAction = potAction(beads, 1, numTimeSlices, tau, numTimeSlices, n, new_eState)

    if np.random.random() < np.exp(-(newAction - oldAction)):
        return new_eState
    else:
        return old_eState

    
def PoE(beads, ptcl, tau, numTimeSlices, n, eState):
    """
    Propagation of excitation move.
    """
    
    pass

def MCMC(numSteps, beads, tau, lam, delta, m, numTimeSlices, numParticles, n, echange, eState, k_e, k_c):
    """
    Perform the path integral Monte Carlo simulation.
    """
    
    EnergyTrace = []
    PositionTrace = []
    eStateTrace = []
    numAccept = {"CoM":0, "Staging":0, "Bead": 0, "eChange": 0}

    if use_jit == "True":
        lam = typed.List(lam)


    #for k in range(numSteps): 
    for k in tqdm(range(numSteps)):

        # try a center-of-mass move
        if k % 1 == 0:
            for i in np.random.randint(0,numParticles,numParticles):
                    numAccept["CoM"] += center_of_mass_move(beads, i, tau, delta, numTimeSlices, n, eState)
        
        if staging == "True":
            # try a staging move
            for i in np.random.randint(0,numParticles,numParticles): 
                numAccept["Staging"] += staging_move(beads, i, tau, lam, numTimeSlices, m, n, eState)
        else:
            # try a bead move
            for i in np.random.randint(0,numParticles,numParticles):
                numAccept["Bead"] += bead_move(beads, i, tau, delta_bead, numTimeSlices, lam, numParticles, n,eState)
        
        if echange == "True":
            if k % k_c == 0:
                for i in np.random.randint(0,numParticles,numParticles):
                    eState, accept = e_change(beads, i, tau, numTimeSlices, n, eState)
                    numAccept["eChange"] += accept
            if non_adiabatic_coupling == "True":
                if k % k_e == 0:
                    for i in np.random.randint(0,numParticles,numParticles):
                        eState = nonAdiab(beads, i, tau, numTimeSlices, n, eState)
        
        # keep track of the energy and position
        potE = potEnergy(beads, tau, numTimeSlices, n, eState)
        
        

        if k % corrSkip == 0 and k > thermSkip:
            if kinVirial == "True":
                kinEvirial = virial_estimator(beads, tau, lam, numTimeSlices, numParticles, eState)
                kinEthermo = kinetic_estimator(beads, tau, lam, numTimeSlices, numParticles)
                EnergyTrace.append([potE, kinEthermo, kinEvirial])
            else:
                kinEthermo = kinetic_estimator(beads, tau, lam, numTimeSlices, numParticles)
                EnergyTrace.append([potE, kinEthermo])
            eStateTrace.append(eState)
            if numParticles == 1:
                PositionTrace.append(beadPos(beads, numTimeSlices))
            else:
                PositionTrace.append(beads)
      
    return np.array(PositionTrace), np.array(EnergyTrace), numAccept, np.array(eStateTrace)



"""
Initialize  -----------------------------------------------------------------
"""

def main(T, n, echange, k_e, k_c):
    
    tau = 1.0/(T*numTimeSlices)

    # initialize main data structure: beads[bead][particl][coordinate]
    beads = np.zeros([numTimeSlices,numParticles, 3])

    if numParticles == 1:
        # If only one particle randomly distribute them
        for tslice in range(numTimeSlices):
            for ptcl in range(numParticles):
                beads[tslice,ptcl] = 0.5*(-1.0 + 2.0*np.random.random(3)) # [-0.5, 0.5)
    elif numParticles > 1:
        # If more than one particle is use distribute them according to the input geometry in geom.in
        atoms, coord = getGeom()
        write_debug_log(f"({T}) Input geometry:\n {coord}")
        for i in range(numParticles):
            for j in range(numTimeSlices):
                beads[j, i] = coord[i] + np.random.rand(3)*0.1

    # initialize e-states (cold start, may also be tried differantly) 
    eState = np.zeros(numTimeSlices, dtype=int)    

    # run main MCMC loop (were the magic hapens)
    Position, Energy, numAccept, eState = MCMC(numMCSteps, beads, tau, lam, delta, m, numTimeSlices, numParticles, n, echange, eState, k_e, k_c)
    
    return Energy, Position, eState, numAccept

def worker(args):

    i, n, echange, k_e, k_c = args
    Energy, Position, eState, numAccept = main(i, n, echange, k_e, k_c)

    # Save Energy and Position data to CSV files
    save_to_csv(Energy[:,0], f'{i}_PotEnergyTrace.csv')
    if kinVirial == "True":
        save_to_csv(Energy[:,2], f'{i}_KinEnergyTrace.csv')
    else:
        save_to_csv(Energy[:,1], f'{i}_KinEnergyTrace.csv')
    save_to_csv(Energy, f'{i}_EnergyTrace.csv')
    save_to_csv(Position, f'{i}_PositionTrace.csv')
    save_to_csv(eState, f'{i}_eStatTrace.csv')
    for key in numAccept:
        numAccept[key] /= numMCSteps*numParticles
    write_debug_log(f"({i}) numAccept: {numAccept}")
   

def parallel_main(T, n, echange, k_e, k_c):

    with ProcessPoolExecutor() as executor:
        executor.map(worker, [(i, n, echange, k_e, k_c) for i in T])

if __name__ == "__main__": 

    # Overwrite old outputs
    remove_all_files_in_folder("output")
    try:
        os.remove("debug.log")
    except:
        pass

    # Read temperature loops
    T = str(config["settings"]["T"]) 
    T_list = T.split(',')
    T = [float(item) for item in T_list]

    # Write input parameters to debug file
    write_debug_log(f"PIMC V1.1")
    write_debug_log(f"Avalible CPUs: {os.cpu_count()}")
    write_debug_log(f"Used CPUs: {len(T)}")
    write_debug_log(f"JIT compiler: {use_jit}")
    write_debug_log(f"Virial estimator: {kinVirial}")
    write_debug_log(f"T: {T}")
    write_debug_log(f"lam: {lam}")
    write_debug_log(f"rand_seed: {rand_seed}")
    write_debug_log(f"numParticles: {numParticles}")
    write_debug_log(f"numTimeSlices: {numTimeSlices}")
    write_debug_log(f"Stage length: {m}")
    write_debug_log(f"Delta (CoM): {delta}")

    if staging == "False":
        write_debug_log(f"Delta (Bead): {delta_bead}")
        write_debug_log(f"Warning: Staging is turned off")

    n = int(config["settings"]["n"]) # dimension of the potetnial matrix (nxn) 
    # Read echange time lags
    if non_adiabatic_coupling == "True":
        k_e = int(config["settings"]["eCL"]) # local e change
        k_c = int(config["settings"]["eCG"]) # global e change
        if numParticles != 1:
            write_debug_log(f"Error: Non-adiabatic couplings can currantly only be used with one particle")
            exit()
    else:
        k_e = float("inf")
        k_c = int(config["settings"]["eCG"])

    if echange == "True":
        write_debug_log(f"eCG: {k_c}")
        write_debug_log(f"eCL: {k_e}")
        if n == 1:
            write_debug_log(f"Warning: echange is True, while n is set to 1")
    
    # sanitiy check of some input parameters
    if numParticles < 1:
        write_debug_log(f"Error: Looks like you tried to bring {numParticles} particles. We need at least 1 to get this party started.")
        exit()
    
    #if len(getV(np.array([0.0]), np.array([0]))):
    #    write_debug_log(f"Error: getV does not return a float")
    #    exit()

    # run PICM simulations
    parallel_main(T, n, echange, k_e, k_c)
    # resample and check convergance
    rcc()
    
