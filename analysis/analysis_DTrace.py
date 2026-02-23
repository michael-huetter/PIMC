import numpy as np
import configparser
from pathlib import Path


# --- user inputs ---
drop_n0 = True
k = 1  # Spring const of HO

config = configparser.ConfigParser()
here = Path(__file__).resolve().parent 
config.read(here.parent / "input.in")
T_list = [float(x) for x in config["system"]["T"].split(",")]

lam_list = [float(x.strip()) for x in config["system"]["lam"].split(",")]
lam0 = lam_list[0]  # single particle case
numTimeSlices = config.getint("PIMC", "numTimeSlices")

# Analytic solution for HO
def D_analytic_3D(tau, beta, omega, lam):
    m = 1.0 / (2.0 * lam)

    pref = 1.0 / (m * omega)

    term1 = 1.0 / np.tanh(beta * omega / 2.0)
    term2 = np.cosh(omega * (tau - beta/2.0)) / np.sinh(beta * omega / 2.0)

    D1D = pref * (term1 - term2)

    return 3.0 * D1D

# Calculate mean squared displacement and compare to analytic function
for T in T_list:
    filename = f"output/{T}_DTrace.csv"

    # --- load data ---
    DTrace = np.loadtxt(filename, delimiter=",")

    # --- average over MC samples ---
    D_mean = np.mean(DTrace, axis=0)
    D_err  = np.std(DTrace, axis=0, ddof=1) / np.sqrt(DTrace.shape[0])

    # --- drop n=0 if desired ---
    if drop_n0:
        D_mean = D_mean[1:]
        D_err  = D_err[1:]
        n = np.arange(1, len(D_mean) + 1)
    else:
        n = np.arange(0, len(D_mean))

    print(f'\nMean squared displacement for Temp {T}: ', D_mean)
    # print(f'Mean squared displacement Error for Temp {T}: ', D_err)

    # Comparison to analytic solution
    beta = 1 / T
    tau = n * beta / numTimeSlices
    omega = np.sqrt(2 * np.array(lam0) * k)
    D_th = D_analytic_3D(tau, beta, omega, lam0)
    print('\nAnalytic solution for mean squared displacement: ', D_th)
