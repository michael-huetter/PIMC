"""
Slow, slower, ab-initio PIMC. Just to play a bit around:)
"""

import numpy as np
import PIMC
from pyscf import dft, gto
import matplotlib.pyplot as plt

def PySCF(pos, dim):
    x, y, z = pos
    x *= 0.529177
    y *= 0.529177
    z *= 0.529177
    r = np.sqrt(x**2 + y**2 + z**2)
    mol_h2 = gto.M(atom = f"H 0 0 0; H 0 0 {r}", basis = "sto-3g", verbose=0)
    rks_h2 = dft.RKS(mol_h2)
    rks_h2.xc = "b3lyp"
    pot_mat = np.zeros((dim, dim))
    pot_mat[0, 0] = rks_h2.kernel()
    return pot_mat

# ---- Parameters ----
beads = 1
particles = 1
dim = 3
mass = [918.076336713]
num_steps = 1_000
step_size_com = 0.1
step_size_sbm = 0.05
echange = False
eCL = 1
eCG = 1
therm_skip = 100
corr_skip = 20
staging = False # Not working atm :/ whyyyyyyyyyyyyyyyyy?
stage_length = 10
T = 0.00095
num_CPU = 1
virial = False
n_estates = 1
# ---- Parameters ----

if __name__ == "__main__":

    PIMC.set_potential(PySCF)
    sim = PIMC.MCMC(beads, particles, dim, T, mass, num_steps, step_size_com, step_size_sbm, echange, eCL, eCG, therm_skip, corr_skip, staging, stage_length, virial, n_estates)
    sim.set_initial_positions(np.zeros((beads, particles, dim)) + 1.0)
    sim.run()
    pos = sim.get_position_trace()
    pos = np.array(pos)
    pos = pos*0.529177
    print(np.mean(pos))
    plt.hist(pos, bins=100)
    plt.show()
