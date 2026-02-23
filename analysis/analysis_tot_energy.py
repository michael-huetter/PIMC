import numpy as np
import configparser
from pathlib import Path
from matplotlib import pyplot as plt


# --- user inputs ---
k = 1  # Spring const of HO

config = configparser.ConfigParser()
here = Path(__file__).resolve().parent 
config.read(here.parent / "input.in")
T_list = [float(x) for x in config["system"]["T"].split(",")]
lam_list = [float(x.strip()) for x in config["system"]["lam"].split(",")]
lam0 = lam_list[0]  # single particle case
omega = np.sqrt(2 * lam0 * k)

# Analytic solution for thermal average energy in 3D HO
def energy_3d_ho(T, omega):
    return 3 * (omega/2 + omega / (np.exp(omega/T) - 1))

E_tot = []
E_an = []
# Totoal energy from KinEnergyTrace and PotEnergyTrace compared to analytical value
for T in T_list:
    pot_filename = f"output/{T}_PotEnergyTrace.csv"
    kin_filename = f"output/{T}_KinEnergyTrace.csv"
    pot = np.loadtxt(pot_filename, delimiter=",")
    kin = np.loadtxt(kin_filename, delimiter=",")

    pot_mean = np.mean(pot)
    kin_mean = np.mean(kin)

    E_tot.append(pot_mean + kin_mean)
    E_an.append(energy_3d_ho(T, omega))

# Plot of Energy vs Temp
plt.figure()
plt.scatter(T_list, E_tot, label='Total Energy', marker='o', alpha=1, s=40, edgecolor=None)
plt.scatter(T_list, E_an, label='Analytical Solution', marker='x', c='red', alpha=1, s=20, edgecolor=None)
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel('Temperature T')
plt.ylabel('Total Energy <E>')
plt.title(f'Total Energy vs Temperature')
plt.legend()
plt.show()



    



