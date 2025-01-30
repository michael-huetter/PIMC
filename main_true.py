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
Known issues:
- Staging is not working
- Non-adiabatic couplings are not implemented yet
- Virial estimator only available if potential is defined and compiled in the C++ code
- Hardcoded absolute path to the setup.py file
-----------------------------------------------------
Usage:
- Compile PIMC locally with: pip install .
- Initialize the simulation parameters: input = input_parameters()
- Costume analytic potential can be defined from python with: input_parameters.set_potential(input, my_potential)
  or costume analytic potential can be defined in C++ and compiled (Potential.hpp) -> much faster
- Initial positions can be set with: input_parameters.initial_positions(input, [[0.0, 0.0, 0.0], ...])
- Initialize the simulation: sim = input.initialize_simulation()
- Run the simulation: sim.run()
Note:
- Natural units are used: hbar = 1, m = 1, e = 1
-----------------------------------------------------
"""
from helpers.files import wOut, initialize_output_file
from helpers.initialize import input_parameters
from helpers.units import c
import numpy as np
import matplotlib.pyplot as plt

input = input_parameters(particles=1, mass=[c.PROTON_MASS_IN_AU/2], T=300*c.K_TO_HARTREE, beads=20, n_estates=1, num_steps=2_000_000, step_size_com=0.1, step_size_sbm=0.02)
input_parameters.initial_positions(input, [[1.0, 0.0, 0.0]])
sim = input.initialize_simulation()
sim.run()

E_trace = sim.get_energy_trace()
E_trace = np.array(E_trace)*c.HARTREE_TO_EV
pos_trace = sim.get_position_trace()
pos_trace = np.array(pos_trace)*c.BOHR_TO_AMSTRONG

print(f"Acceptance rate: {sim.get_acceptance_rates()}")
print(f"Mean energy: {np.mean(E_trace)} eV")
print(f"Mean position: {np.mean(pos_trace)} A")

plt.hist(np.array(pos_trace), bins=100)
plt.show()

