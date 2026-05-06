import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import analysis_funktions as af
import configparser

config = configparser.ConfigParser()
config.read('../input.in')

T_s = config.get("system","T")
T = [float(x) for x in T_s.split(",")]
N = config.getint("PIMC", "numMCSteps")
HMC = 1
Skip = 500



Ekin_mean = []
Epot_mean = []
Pos_mean = []

delta_Ekin = []
delta_Epot = []
delta_Pos = []

for Ti in T:
    E_kin_roh = pd.read_csv(f"../output_{HMC}/{Ti}_KinEnergyTrace.csv")
    E_pot_roh = pd.read_csv(f"../output_{HMC}/{Ti}_PotEnergyTrace.csv")
    Pos_roh = pd.read_csv(f"../output_{HMC}/{Ti}_PositionObsTrace.csv")

    E_kin_i = E_kin_roh.to_numpy()
    E_pot_i = E_pot_roh.to_numpy()
    Pos_i = Pos_roh.to_numpy()

    Ekin_mean_i = np.mean(E_kin_i[Skip:])
    Epot_mean_i = np.mean(E_pot_i[Skip:])
    Pos_mean_i = np.mean(Pos_i[Skip:])

    kin_arc = af.autokorr(E_kin_i, 4000)
    pot_arc = af.autokorr(E_pot_i, 4000)
    pos_arc = af.autokorr(Pos_i, 4000)

    tau_int_pot = af.integrat_atokorr(pot_arc)
    tau_int_kin = af.integrat_atokorr(kin_arc)
    tau_int_pos = af.integrat_atokorr(pos_arc)

    delta_Ekin_i = np.sqrt(2 * tau_int_kin * np.var(E_kin_i[Skip:]) / (N - Skip))
    delta_Epot_i = np.sqrt(2 * tau_int_pot * np.var(E_pot_i[Skip:]) / (N - Skip))
    delta_Pos_i = np.sqrt(2 * tau_int_pos * np.var(Pos_i[Skip:]) / (N - Skip))

    Ekin_mean.append(Ekin_mean_i)
    Epot_mean.append(Epot_mean_i)
    Pos_mean.append(Pos_mean_i)
    delta_Ekin.append(delta_Ekin_i)
    delta_Epot.append(delta_Epot_i)
    delta_Pos.append(delta_Pos_i)

plt.plot(T,Ekin_mean,"or",label="Kinetische Energie")
plt.title(r"$E_{kin}$ und $E_{pot}$ über Temperatur")
plt.plot(T,Epot_mean,"ob",label="Potenzielle Energie")
plt.errorbar(T,Ekin_mean,yerr=delta_Ekin,linestyle ="None",label = "Fehlerbalken",color="orange")
plt.errorbar(T,Epot_mean,yerr=delta_Epot,linestyle="None",color="orange")
plt.xlabel("Temperatur")
plt.ylabel("Energie")
plt.legend()
plt.grid(linestyle = "--")
plt.show()