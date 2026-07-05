import numpy as np
import configparser
from pathlib import Path
from matplotlib import pyplot as plt
import analysis_funktions as af
import pandas as pd


k = 1  # Spring const of HO
HMC = 1
Skip = 200

config = configparser.ConfigParser()
here = Path(__file__).resolve().parent
config.read(here.parent / "input.in")
T_list = [float(x) for x in config["system"]["T"].split(",")]
lam_list = [float(x.strip()) for x in config["system"]["lam"].split(",")]
lam0 = lam_list[0]  # single particle case
omega = np.sqrt(2 * lam0 * k)
N = config.getint("PIMC", "numMCSteps")


# Analytic solution for thermal average energy in 3D HO
def energy_3d_ho(T, omega):
    return 3 * (omega / 2 + omega / (np.exp(omega / T) - 1))


E_tot = []
E_tot_delta = []
# Totoal energy from KinEnergyTrace and PotEnergyTrace compared to analytical value
for T in T_list:
    E_kin_roh = pd.read_csv(f"../output_{HMC}/{T}_KinEnergyTrace.csv")
    E_pot_roh = pd.read_csv(f"../output_{HMC}/{T}_PotEnergyTrace.csv")
    E_kin_i = E_kin_roh.to_numpy()
    E_pot_i = E_pot_roh.to_numpy()
    Eges = E_kin_i[Skip:]+E_pot_i[Skip:]

    Eges_mean = np.mean(Eges)
    Eges_arc = af.autokorr(Eges, 8000)
    tau_int_Eges = af.integrat_atokorr(Eges_arc)
    delta_Eges = np.sqrt(2 * tau_int_Eges * np.var(Eges) / (N - Skip))
    E_tot.append(Eges_mean)
    E_tot_delta.append(delta_Eges)

T_an = np.linspace(0,11,100)
E_an = energy_3d_ho(T_an,omega)

print(energy_3d_ho(T_list,omega)/2)

# Plot of Energy vs Temp
plt.figure()
plt.plot(T_list, E_tot, "or", label=r"$\langle E_{ges} \rangle$ HMC Sample")
#plt.plot(T_an,3*T_an)
plt.errorbar(T_list,E_tot, yerr=E_tot_delta, linestyle ="None",label = "Fehlerbalken",color="orange")
plt.plot(T_an, E_an, "b", label="Analytische Lösung")
plt.xlabel("Temperatur")
plt.ylabel(r"$\langle$ Gesamt Energie $\rangle$")
plt.title("Gesamt Energie vs Temperatur")
plt.axis([0,10.7,0,32])
plt.grid(linestyle = "--")
plt.legend()
plt.show()






