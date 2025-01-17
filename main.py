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

# -.-. .... --- --- ... .    .-- .. ... . .-.. -.-- 
beads = 20
particles = 1
dim = 3
mass = [1.0]
num_steps = 500_000
step_size_com = 1.0
step_size_sbm = 0.1
echange = True
eCL = 1
eCG = 1
therm_skip = 1000
corr_skip = 20
staging = True
stage_length = 18
T = np.linspace(0.3, 5.0, 4)
num_CPU = 4
# -.-. .... --- --- ... .    .-- .. ... . .-.. -.-- 

def run_sim(T):
    wOut(f"Running simulation for T = {T}")
    sim = PIMC.MCMC(beads, particles, dim, T, mass, num_steps, step_size_com, step_size_sbm, echange, eCL, eCG, therm_skip, corr_skip, staging, stage_length)
    #sim.print_parameters()
    sim.run()
    wOut(f"Simulation for T = {T} finished. Acceptance rate: {sim.get_acceptance_rates()}. Mean energy: {np.mean(sim.get_energy_trace())}")
    #pos = sim.get_position_trace()
    return sim.get_energy_trace()

if __name__ == "__main__":

    if os.cpu_count() < num_CPU:
        Warning("Number of used CPUs is larger than available CPUs.")
    initialize_output_file()
    wOut(f"PIMC V1.2")

    pool = multiprocessing.Pool(processes=num_CPU)
    E_trace = pool.map(run_sim, T)

    E = np.zeros(len(T))
    for i in range(len(T)):
        E[i] = np.mean(E_trace[i])

    plt.plot(T, E, "o", label="PIMC")
    T = np.linspace(0.1, 5.0, 100)
    plt.plot(T ,(3/2)/np.tanh(0.5/T), label="Analytical")
    plt.plot(T, (3/2)/np.tanh(0.5/T) + 1 / (np.exp(1/T) + 1))
    plt.xlabel("Temperature")
    plt.ylabel("Energy")
    plt.legend()
    plt.show()

