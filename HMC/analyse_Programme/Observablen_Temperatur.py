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
Skip = 2000



Ekin_mean = []
Epot_mean = []
Pos_mean = []

delta_Ekin = []
delta_Epot = []
delta_Pos = []

tau_int = []

for Ti in T:
    E_kin_roh = pd.read_csv(f"../output_{HMC}/{Ti}_KinEnergyTrace.csv")
    E_pot_roh = pd.read_csv(f"../output_{HMC}/{Ti}_PotEnergyTrace.csv")
    #Pos_roh = pd.read_csv(f"../output_{HMC}/{Ti}_PositionObsTrace.csv")

    E_kin_i = E_kin_roh.to_numpy()
    E_pot_i = E_pot_roh.to_numpy()
    #Pos_i = Pos_roh.to_numpy()

    Ekin_mean_i = np.mean(E_kin_i[Skip:])
    Epot_mean_i = np.mean(E_pot_i[Skip:])
    #Pos_mean_i = np.mean(Pos_i[Skip:])

    kin_arc = af.autokorr(E_kin_i[Skip:], 500)
    pot_arc = af.autokorr(E_pot_i[Skip:], 500)
    #pos_arc = af.autokorr(Pos_i[Skip:], 4000)

    tau_int_pot = af.integrat_atokorr(pot_arc)
    tau_int_kin = af.integrat_atokorr(kin_arc)
    #tau_int_pos = af.integrat_atokorr(pos_arc)

    delta_Ekin_i = np.sqrt(2 * tau_int_kin * np.var(E_kin_i[Skip:]) / (N - Skip))
    delta_Epot_i = np.sqrt(2 * tau_int_pot * np.var(E_pot_i[Skip:]) / (N - Skip))
    #delta_Pos_i = np.sqrt(2 * tau_int_pos * np.var(Pos_i[Skip:]) / (N - Skip))

    Ekin_mean.append(Ekin_mean_i)
    Epot_mean.append(Epot_mean_i)
    #Pos_mean.append(Pos_mean_i)
    delta_Ekin.append(delta_Ekin_i)
    delta_Epot.append(delta_Epot_i)
    #delta_Pos.append(delta_Pos_i)
    tau_int.append([tau_int_kin,tau_int_pot])
    print(Ti, ":" , float(tau_int_kin),float(tau_int_pot),float(Ekin_mean_i),float(delta_Ekin_i),float(Epot_mean_i),float(delta_Epot_i))



plt.title(r"$\langle E_{pot} \rangle$ und $\langle E_{kin} \rangle$ über Temperatur")
plt.plot(T,Ekin_mean,"or",label=r"$\langle E_{kin} \rangle$", alpha=0.8)
plt.plot(T,Epot_mean,"ob",label=r"$\langle E_{pot} \rangle$", alpha=0.6)
plt.errorbar(T,Ekin_mean,yerr=delta_Ekin,linestyle ="None",label = "Fehlerbalken",color="orange")
plt.errorbar(T,Epot_mean,yerr=delta_Epot,linestyle="None",color="orange")
plt.xlabel("Temperatur")
plt.ylabel("Energie")
plt.legend()
plt.grid(linestyle = "--")
plt.show()