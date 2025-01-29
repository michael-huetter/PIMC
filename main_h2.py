"""
░▒▓███████▓▒░░▒▓█▓▒░▒▓██████████████▓▒░ ░▒▓██████▓▒░  
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓███████▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░ 
-----------------------------------------------------
PIMC - Path Integral Monte Carlo Simulation
-----------------------------------------------------
Includes:
- Multiple electronic states
- Non-adiabatic couplings in the diabatic representation
- Staging
- Virial and Thermodynamic estimators
- Arbitrary defined potential energy surfaces
Usage:
- Compile PIMC locally with: pip install .
- Specify the simulation parameters
- Specify the potential energy surfaces
- Run the simulation
see below for an example.
Note:
- The main simulation class is MCMC which is a wrapper for the C++ code.
- Non-adiabatic couplings are currently only implemented for 1D PES.
- The primitive approximation to the path integral is used.
-----------------------------------------------------
"""
import PIMC
from helpers import wOut, initialize_output_file
import os
import multiprocessing 
import numpy as np
import matplotlib.pyplot as plt
import time

# -.-. .... --- --- ... .    .-- .. ... . .-.. -.-- 
# Input parameter examples, H2 potential
beads = 20
particles = 1
dim = 3
mass = [918.076336713]
num_steps = 3_000_000
step_size_com = 0.1
step_size_sbm = 0.05
echange = False
eCL = 1
eCG = 1
therm_skip = 1000
corr_skip = 20
staging = False # Not working atm :/ whyyyyyyyyyyyyyyyyy?
stage_length = 10
T = [0.00095]
num_CPU = 1
virial = True
n_estates = 1
# -.-. .... --- --- ... .    .-- .. ... . .-.. -.-- 

def run_sim(T):
    wOut(f"Running simulation for T = {T}")
    sim = PIMC.MCMC(beads, particles, dim, T, mass, num_steps, step_size_com, step_size_sbm, echange, eCL, eCG, therm_skip, corr_skip, staging, stage_length, virial, n_estates)
    #sim.print_parameters()
    sim.set_initial_positions(np.zeros((beads, particles, dim)) + 1.0)
    sim.run()
    wOut(f"Simulation for T = {T} finished. Acceptance rate: {sim.get_acceptance_rates()}. Mean energy: {np.mean(sim.get_energy_trace())}")
    pos = sim.get_position_trace()
    e = sim.get_energy_trace()
    e = np.array(e)
    pos = np.array(pos)
    #sim.print_parameters()
    pos = pos*0.529177
    e = e*27.211
    print(np.mean(e))
    print(np.mean(pos))
    plt.hist(pos, bins=100)
    plt.show()

if __name__ == "__main__":

    start_time = time.time()

    if os.cpu_count() < num_CPU:
        Warning("Number of used CPUs is larger than available CPUs.")
    initialize_output_file()
    wOut(f"PIMC V1.2")

    pool = multiprocessing.Pool(processes=num_CPU)
    pool.map(run_sim, T)

    wOut(f"Total runtime: {time.time()-start_time} s")