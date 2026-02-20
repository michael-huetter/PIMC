import numpy as np
import configparser
from pathlib import Path
from matplotlib import pyplot as plt
from numba import njit


# --- user inputs ---
k = 1  # Spring const of HO

config = configparser.ConfigParser()
here = Path(__file__).resolve().parent 
config.read(here.parent / "input.in")
T_list = [float(x) for x in config["system"]["T"].split(",")]
lam_list = [float(x.strip()) for x in config["system"]["lam"].split(",")]
lam0 = lam_list[0]  # single particle case
numTimeSlices = config.getint("PIMC", "numTimeSlices")
numParticles = config.getint("system", "numParticles")

# Analytic solution for 3D HO
def C_analytic_3D(tau, beta, omega, k):
    tau = np.abs(tau)
    x = beta * omega

    denom1 = np.expm1(x)         
    denom2 = -np.expm1(-x)       

    term1 = np.exp(omega * tau) / denom1
    term2 = np.exp(-omega * tau) / denom2

    pref = omega / (2.0 * k)
    C1D = pref * (term1 + term2)

    return 3 * C1D

# Correlation function
@njit
def imaginary_time_corr(beads: np.ndarray, numTimeSlices: int, numParticles: int) -> np.ndarray:
    """
    C[n] = (1/N) * (1/P) * sum_{ptcl} sum_{j} (r_{j+n} * r_j)
    """
    max_n = numTimeSlices #// 2  # Since C(n) = C(P - n)
    num_beads = beads.shape[0]
    C = np.zeros((num_beads, max_n + 1), dtype=np.float64)

    for i in range(num_beads):
        for nsep in range(max_n + 1):
            tot = 0.0
            for ptcl in range(numParticles):
                for j in range(numTimeSlices):
                    jp = (j + nsep) % numTimeSlices
                    r = np.dot(beads[i, jp, ptcl], beads[i, j, ptcl])
                    tot += r
            C[i, nsep] = tot / (numParticles * numTimeSlices)

    return C

# Imaginary time correlation function calculated from CTrace.csv and PositionTrace.npy and compared to analytic function
for T in T_list:
    # From CTrace.csv ---------------------------------------------------
    filename = f"output/{T}_CTrace.csv"
    CTrace = np.loadtxt(filename, delimiter=",")

    # average over MC samples
    C_mean = np.mean(CTrace, axis=0)
    C_err  = np.std(CTrace, axis=0, ddof=1) / np.sqrt(CTrace.shape[0])
    # --------------------------------------------------------------------

    # From PositionTrace.npy ---------------------------------------------------
    filename = f"output/{T}_PositionTrace.npy"
    beads = np.load(filename)

    CTrace_pos = imaginary_time_corr(beads, numTimeSlices, numParticles)
    C_mean_pos = np.mean(CTrace_pos, axis=0)
    # --------------------------------------------------------------------------

    # Comparison to analytic solution
    n = np.arange(0, len(C_mean))
    n_pos = np.arange(0, len(C_mean_pos))
    beta = 1 / T
    tau = n * beta / numTimeSlices
    omega = np.sqrt(2 * np.array(lam0) * k)
    C_th = C_analytic_3D(tau, beta, omega, k)

    # Plots for comparison
    n_plot = np.linspace(0, numTimeSlices, 500)
    tau_plot = n_plot * beta / numTimeSlices
    C_th_plot = C_analytic_3D(tau_plot, beta, omega, k)

    plt.figure()
    plt.scatter(n, C_mean, label='CTrace', marker='o', alpha=1, s=40, edgecolor=None)
    plt.scatter(n_pos, C_mean_pos, label='PositionTrace', marker='x', c='red', alpha=1, s=20, edgecolor=None)
    plt.plot(n_plot, C_th_plot, label='Analytic solution')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel('Step size in imaginary time n')
    plt.ylabel('Imaginary time correlation function')
    plt.title(f'Correlation Function vs Step Size at Temp: {T}')
    plt.legend()

    # Check if average of first half matches average of second half of each array
    C_half = CTrace.shape[0] // 2
    C_mean_1 = np.mean(CTrace[:C_half])
    C_mean_2 = np.mean(CTrace[C_half:])
    print(f'\nTemp T = {T}')
    print('C Mean:')
    print(f'First half average: {C_mean_1}')
    print(f'Second half average: {C_mean_2}')

    pos_half = CTrace_pos.shape[0] // 2
    pos_mean_1 = np.mean(CTrace_pos[:pos_half])
    pos_mean_2 = np.mean(CTrace_pos[pos_half:])
    print('\nPos Mean:')
    print(f'First half average: {pos_mean_1}')
    print(f'Second half average: {pos_mean_2}')

plt.show()

    