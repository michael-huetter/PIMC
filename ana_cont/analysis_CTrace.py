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


# Connected correlation function
@njit
def connected_imaginary_time_corr(beads, numTimeSlices, numParticles):
    n_samples, _, _, dim = beads.shape
    max_n = numTimeSlices // 2
    norm = numParticles * numTimeSlices
    C = np.zeros((n_samples, max_n + 1), dtype=np.float64)

    for i in range(n_samples):
        # per-sample centroid
        r_bar = np.zeros(dim)
        for ptcl in range(numParticles):
            for j in range(numTimeSlices):
                for d in range(dim):
                    r_bar[d] += beads[i, j, ptcl, d]
        for d in range(dim):
            r_bar[d] /= norm

        for nsep in range(max_n + 1):
            tot = 0.0
            for ptcl in range(numParticles):
                for j in range(numTimeSlices):
                    jp = (j + nsep) % numTimeSlices
                    for d in range(dim):
                        tot += (beads[i, jp, ptcl, d] - r_bar[d]) * \
                               (beads[i, j,  ptcl, d] - r_bar[d])
            C[i, nsep] = tot / norm
    return C


def remove_zero_mode(C_tau, beta, omega_min_expected):
    """Subtract centroid contribution if the gap justifies it"""
    plateau = C_tau[-1]
    # Check: at tau = beta/2 if the lowest physical mode already negligible
    decay_factor = np.exp(-beta * omega_min_expected / 2)
    if decay_factor < 0.01 and plateau > 0:
        return C_tau - plateau, plateau
    else:
        return C_tau, 0.0


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
    n_blocks = blocks.shape[0]
    mean = np.mean(blocks, axis=0)
    err = np.std(blocks, axis=0, ddof=1) / np.sqrt(n_blocks)
    cov = np.cov(blocks, rowvar=False, ddof=1) / n_blocks
    return mean, err, cov, blocks
    

# Computing correlation data
def compute_correlation_data():
    # Input
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

    # Imaginary time correlation function calculated from and PositionTrace.npy and compared to analytic function
    for T in T_list:
        beta = 1 / T
        omega = np.sqrt(2 * lam * k)

        # From PositionTrace.npy ---------------------------------------------------
        filename = output_dir / f"{T}_PositionTrace.npy"
        beads = np.load(filename)

        CTrace_pos = imaginary_time_corr(beads, numTimeSlices, numParticles)
        C_mean_pos, C_err_pos, C_cov_pos, C_blocks_pos = blocked_stats(CTrace_pos, block_size)
        # C_mean_pos, _ = remove_zero_mode(C_mean_pos, beta, omega)
        # --------------------------------------------------------------------------
        
        # Connected correlation-----------------------------------------------------
        CTrace_conn = connected_imaginary_time_corr(beads, numTimeSlices, numParticles)
        C_mean_conn, C_err_conn, C_cov_conn, C_blocks_conn = blocked_stats(CTrace_conn, block_size)
        # -------------------------------------------------------------------------

        # Parameters
        n = np.arange(0, len(C_mean_pos))
        tau = n * beta / numTimeSlices

        results[T] = {
            "CTrace_pos": CTrace_pos,
            "C_mean_pos": C_mean_pos,
            "C_err_pos": C_err_pos,
            "C_cov_pos": C_cov_pos,
            "C_blocks_pos": C_blocks_pos,

            "CTrace_conn": CTrace_conn,
            "C_mean_conn": C_mean_conn,
            "C_err_conn": C_err_conn,
            "C_cov_conn": C_cov_conn,
            "C_blocks_conn": C_blocks_conn,

            "tau": tau,
            "beta": beta,
            "omega": omega,
            "n": n,
            'k': k
            }

    return results, T_list, numTimeSlices

if __name__ == "__main__":
    results, T_list, numTimeSlices = compute_correlation_data()

    for T in T_list:
        C_mean_conn = results[T]['C_mean_conn']
        C_mean_pos = results[T]["C_mean_pos"]
        beta = results[T]["beta"]
        omega = results[T]["omega"]
        n = results[T]["n"]
        k = results[T]['k']

        for i in range(len(k)):
            # Plots for comparison with analytic solution
            n_plot = np.linspace(0, numTimeSlices // 2, 500)
            tau_plot = n_plot * beta / numTimeSlices
            C_th_plot = C_analytic_3D(tau_plot, beta, omega[i], k[i])

            plt.figure()
            plt.scatter(n, C_mean_pos, label='PositionTrace blocked', marker='x', c='red', alpha=1, s=20, edgecolor=None)
            plt.plot(n_plot, C_th_plot, label='Analytic solution')
            plt.xscale('linear')
            plt.yscale('linear')
            plt.xlabel('Step size in imaginary time n')
            plt.ylabel('Imaginary time correlation function')
            plt.title(f'Correlation Function vs Step Size at Temp = {T}, k = {k[i]}, omega = {omega[i]}')
            plt.legend()
            plt.tight_layout()

            plt.figure()
            plt.scatter(n, C_mean_conn, label='CTrace connected', marker='o', alpha=1, s=40, edgecolor=None)
            plt.xscale('linear')
            plt.yscale('linear')
            plt.xlabel('Step size in imaginary time n')
            plt.ylabel('Imaginary time correlation function')
            plt.title(f'Connected Correlation Function vs Step Size at Temp = {T}, k = {k[i]}, omega = {omega[i]}')
            plt.legend()
            plt.tight_layout()

    plt.show()

