import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import analysis_funktions as af


HMC = 0
T = 10.0



E_kin_roh = pd.read_csv(f"../output_{HMC}/{T}_KinEnergyTrace.csv")
E_pot_roh = pd.read_csv(f"../output_{HMC}/{T}_PotEnergyTrace.csv")
Pos_roh = pd.read_csv(f"../output_{HMC}/{T}_PositionObsTrace.csv")


Ekin = E_kin_roh.to_numpy()
Epot = E_pot_roh.to_numpy()
Pos = Pos_roh.to_numpy()


pot_arc = af.autokorr(Epot,4000)
kin_arc = af.autokorr(Ekin,4000)
Pos_arc = af.autokorr(Pos,4000)


tau_int_pot = af.integrat_atokorr(pot_arc)
tau_int_kin = af.integrat_atokorr(kin_arc)
tau_int_pos = af.integrat_atokorr(Pos_arc)


print(f"{tau_int_kin}")
print(f"{tau_int_pot}")
print(f"{tau_int_pos}")


fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].plot(kin_arc)
axs[0].set_title("Kinetische Energie Autokorrelation")
axs[0].grid(linestyle = "--")
axs[0].text(3000,0.9,f"$\\tau_{{int}}$ = {tau_int_kin:.2f}")

axs[1].plot(pot_arc)
axs[1].set_title("Potentielle Energie Autokorrelation")
axs[1].grid(linestyle = "--")
axs[1].text(3000,0.9,f"$\\tau_{{int}}$ = {tau_int_pot:.2f}")


axs[2].plot(Pos_arc)
axs[2].set_title("Positions Autokorrelation")
axs[2].grid(linestyle = "--")
axs[2].text(3000,0.9,f"$\\tau_{{int}}$ = {tau_int_pos:.2f}")
plt.tight_layout()
plt.show()