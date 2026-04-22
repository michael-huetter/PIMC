import shutil
from pathlib import Path
import configparser

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from analysis_CTrace import compute_correlation_data
import ana_cont.continuation as cont



# Output folders
output_dir_spec = Path("spectrum_plots")
if output_dir_spec.exists():
    shutil.rmtree(output_dir_spec)
output_dir_spec.mkdir(parents=True, exist_ok=True)


# Read input parameters
def load_maxent_config(config_path="input.in"):
    config = configparser.ConfigParser()
    here = Path(__file__).resolve().parent 
    config.read(here.parent / "input.in")

    maxent = {
        "n_w": config.getint("maxent", "n_w"),
        "method": config.get("maxent", "method"),
        "alpha_determination": config.get("maxent", "alpha_determination"),
        "optimizer": config.get("maxent", "optimizer"),
        "kernel_mode": config.get("maxent", "kernel_mode"),
        "wmax_abs": config.getfloat("maxent", "wmax_abs")
    }

    return maxent


def run_maxent_for_temperature(results, params, T):
    # From results
    beta = results[T]["beta"]
    tau = results[T]["tau"]              
    C_tau = results[T]["C_mean_pos"] # blocked mean
    cov = results[T]["C_cov_pos"] # blocked covariance
    omega = results[T]["omega"] # HO frequency scale
    print(f'\nOmega1 = {omega[0]}\nOmega2 = {omega[1]}\n')

    # From params
    n_w = params['n_w']
    method = params['method']
    alpha_determination = params['alpha_determination']
    optimizer = params['optimizer']
    kernel_mode = params['kernel_mode']
    wmax_abs = params['wmax_abs']            

    # Real freq grid
    w = np.linspace(0.0, wmax_abs, n_w)

    # Diferent models to try
    ################################################################
    # # Flat positive default model, normalized
    # model = np.ones_like(w)

    # # Gauss model
    model = np.zeros_like(w)
    for i in range(len(omega)):
        sigma_i = 0.5 * omega[i]
        model += np.exp(-(w - omega[i])**2 / (2.0 * sigma_i**2))

    # # Weakly decaying model
    # w_scale = max(w[-1] / 5, 1e-8)
    # model = np.exp(-w / w_scale)
    ################################################################

    # Model normalization
    model /= np.trapezoid(model, w)

    # Build continuation problem
    probl = cont.AnalyticContinuationProblem(
        im_axis=tau,
        re_axis=w,
        im_data=C_tau,
        kernel_mode=kernel_mode,
        beta=beta,
    )

    # Solve MaxEnt
    sol, sols_alpha = probl.solve(
        method=method,
        alpha_determination=alpha_determination,
        model=model,
        cov=cov,
        optimizer=optimizer,
    )

    return {
        "w": w,
        "solution": sol,
        "all_alpha_solutions": sols_alpha,
    }


# Main
if __name__ == "__main__":
    params = load_maxent_config()
    results, T_list, _ = compute_correlation_data()

    for T in T_list:
        out = run_maxent_for_temperature(results, params, T)
        w = out['w']
        sol = out["solution"]

        # Spectum amplitude and backtransform
        amplitude = sol.A_opt
        CTrace_back = sol.backtransform

        # Compare backtransfrom to CTrace
        omega = results[T]["omega"]
        tau = results[T]["tau"]              
        CTrace = results[T]["C_mean_pos"]

        CTrace_shape = CTrace - np.mean(CTrace) 
        C_back_shape = CTrace_back - np.mean(CTrace_back)


        # Plot spectum curve
        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(2, 2, height_ratios=[1, 1])
        ax0 = fig.add_subplot(gs[0, :])

        # Spectrum plot
        ax0.plot(w, amplitude, label='Frequency')
        for omega0 in omega:
            ax0.axvline(x=omega0, linestyle='--', linewidth=1.5, label=f'ω₀={omega0:.2f}')
        ax0.set_xlabel('Omega')
        ax0.set_ylabel('Amplitude')
        ax0.set_title(f'Spectrum at Temperature: {T}')
        ax0.legend()

        # Backtransform comparison
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.plot(tau, CTrace, "o", label="input C(tau)")
        ax1.plot(tau, CTrace_back, "-", label="backtransform")
        ax1.set_xlabel(r"$\tau$")
        ax1.set_ylabel(r"$C(\tau)$")
        ax1.set_title("Backtransform check")
        ax1.legend()

        # Backtransfrom shape comparison
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.plot(tau, CTrace_shape, "o", label="input - avg")
        ax2.plot(tau, C_back_shape, "-", label="backtransform - avg")
        ax2.set_xlabel(r"$\tau$")
        ax2.set_ylabel(r"$C(\tau)$")
        ax2.set_title("Shape check")
        ax2.legend()

        plt.tight_layout()
        filename = f"spectrum_plots/T_{T}.png"
        plt.savefig(filename, dpi=300)      

plt.show()