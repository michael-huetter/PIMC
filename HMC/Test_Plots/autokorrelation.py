import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import analysis_funktions as af


HMC = 1
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


print(f"{tau_int_pot}")
print(f"{tau_int_kin}")
print(f"{tau_int_pos}")

plt.plot(pot_arc)
plt.title("Potentielle Energie Autokorrelation")
plt.grid(linestyle="--")
plt.show()
plt.clf()

plt.plot(kin_arc)
plt.title("Kinetische Energie Autokorrelation")
plt.grid(linestyle="--")
plt.show()
plt.clf()

plt.plot(Pos_arc)
plt.title("Positions Autokorrelation")
plt.grid(linestyle="--")
plt.show()