import numpy as np
import configparser
from pathlib import Path
from matplotlib import pyplot as plt
from numba import njit


# Correlation function
@njit
def imaginary_time_corr(beads: np.ndarray, numTimeSlices: int, numParticles: int) -> np.ndarray:
    """
    C[n] = (1/N) * (1/P) * sum_{ptcl} sum_{j} (r_{j+n} * r_j)
    """
    max_n = numTimeSlices // 2  # Since C(n) = C(P - n)
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

# Blocking funcitons to reduce correlation in cov matrix
def make_blocks(samples: np.ndarray, block_size: int) -> np.ndarray:
    n_samples = samples.shape[0]
    n_blocks = n_samples // block_size

    if n_blocks < 2:
        raise ValueError(
            f"Need at least 2 blocks, got n_samples={n_samples}, block_size={block_size}"
        )

    trimmed = samples[:n_blocks * block_size]
    reshaped = trimmed.reshape(n_blocks, block_size, samples.shape[1])
    blocks = reshaped.mean(axis=1)

    return blocks

def blocked_stats(samples: np.ndarray, block_size: int):
    blocks = make_blocks(samples, block_size)
    mean = np.mean(blocks, axis=0)
    err = np.std(blocks, axis=0, ddof=1) / np.sqrt(blocks.shape[0])
    cov = np.cov(blocks, rowvar=False, ddof=1)

    return mean, err, cov, blocks
    
def compute_correlation_data():

    config = configparser.ConfigParser()
    here = Path(__file__).resolve().parent 
    config.read(here.parent / "input.in")
    output_dir = here.parent / "output"
    T_list = [float(x) for x in config["system"]["T"].split(",")]
    lam_str = config.get('system', 'lam')
    lam = np.array([float(x.strip()) for x in lam_str.split(",")], dtype=np.float64)
    numTimeSlices = config.getint("PIMC", "numTimeSlices")
    numParticles = config.getint("system", "numParticles")
    k_str = config.get('maxent', 'k')
    k = np.array([float(x.strip()) for x in k_str.split(",")], dtype=np.float64)
    block_size = config.getint('maxent', 'block_size')

    results = {}

    # Imaginary time correlation function calculated from CTrace.csv and PositionTrace.npy and compared to analytic function
    for T in T_list:
        # From CTrace.csv ---------------------------------------------------
        filename = output_dir / f"{T}_CTrace.csv"
        CTrace = np.loadtxt(filename, delimiter=",")

        # average over MC samples
        C_mean = np.mean(CTrace, axis=0)
        C_err  = np.std(CTrace, axis=0, ddof=1) / np.sqrt(CTrace.shape[0])
        # --------------------------------------------------------------------

        # From PositionTrace.npy ---------------------------------------------------
        filename = output_dir / f"{T}_PositionTrace.npy"
        beads = np.load(filename)

        CTrace_pos = imaginary_time_corr(beads, numTimeSlices, numParticles)
        C_mean_pos, C_err_pos, C_cov_pos, C_blocks_pos = blocked_stats(CTrace_pos, block_size)
        # --------------------------------------------------------------------------

        # Parameters
        n = np.arange(0, len(C_mean))
        n_pos = np.arange(0, len(C_mean_pos))
        beta = 1 / T
        tau = n * beta / numTimeSlices
        omega = np.sqrt(2 * lam * k)

        results[T] = {
            "CTrace_pos": CTrace_pos,
            "C_mean_pos": C_mean_pos,
            "C_err_pos": C_err_pos,
            "C_cov_pos": C_cov_pos,
            "C_blocks_pos": C_blocks_pos,

            "CTrace": CTrace,
            "C_mean": C_mean,
            "C_err": C_err,

            "tau": tau,
            "beta": beta,
            "omega": omega,
            "n": n,
            "n_pos": n_pos,
            'k': k
            }

    return results, T_list, numTimeSlices

if __name__ == "__main__":
    results, T_list, numTimeSlices = compute_correlation_data()

    for T in T_list:
        tau = results[T]["tau"]
        C_mean = results[T]["C_mean"]
        C_mean_pos = results[T]["C_mean_pos"]
        C_err = results[T]["C_err"]
        CTrace = results[T]["CTrace"]
        CTrace_pos = results[T]["CTrace_pos"]
        beta = results[T]["beta"]
        omega = results[T]["omega"]
        n = results[T]["n"]
        n_pos = results[T]["n_pos"]
        k = results[T]['k']

        for i in range(len(k)):
            # Plots for comparison with analytic solution
            n_plot = np.linspace(0, numTimeSlices // 2, 500)
            tau_plot = n_plot * beta / numTimeSlices
            C_th_plot = C_analytic_3D(tau_plot, beta, omega[i], k[i])

            plt.figure()
            plt.scatter(n, C_mean, label='CTrace', marker='o', alpha=1, s=40, edgecolor=None)
            plt.scatter(n_pos, C_mean_pos, label='PositionTrace', marker='x', c='red', alpha=1, s=20, edgecolor=None)
            plt.plot(n_plot, C_th_plot, label='Analytic solution')
            plt.xscale('linear')
            plt.yscale('linear')
            plt.xlabel('Step size in imaginary time n')
            plt.ylabel('Imaginary time correlation function')
            plt.title(f'Correlation Function vs Step Size at Temp = {T}, k = {k[i]}, omega = {omega[i]}')
            plt.legend()

    plt.show()

    