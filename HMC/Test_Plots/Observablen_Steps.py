import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import analysis_funktions as af

HMC = 1
T = 2.0
Skip = 100
N = 10000

E_kin_roh = pd.read_csv(f"../output_{HMC}/{T}_KinEnergyTrace.csv")
E_pot_roh = pd.read_csv(f"../output_{HMC}/{T}_PotEnergyTrace.csv")

Ekin = E_kin_roh.to_numpy()
Epot = E_pot_roh.to_numpy()

Ekin_mean = np.mean(Ekin[Skip:])
Epot_mean = np.mean(Epot[Skip:])

pot_arc = af.autokorr(Epot,4000)
kin_arc = af.autokorr(Ekin,4000)

tau_int_pot = af.integrat_atokorr(pot_arc)
tau_int_kin = af.integrat_atokorr(kin_arc)

delta_Ekin = np.sqrt(2 * tau_int_kin * np.var(Ekin[Skip:]) / (N-Skip))
delta_Epot = np.sqrt(2 * tau_int_pot * np.var(Epot[Skip:]) / (N-Skip))


N = len(Ekin)
Steps = np.arange(0,N)


fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# Plot 1: Kinetische Energie
axs[0].plot(Steps, Ekin)
axs[0].plot([0, N], [Ekin_mean, Ekin_mean])
axs[0].set_title(f"Kinetische Energie ({T=})")
axs[0].legend(["Samplepoints", f"Mean = {Ekin_mean:.3f} $\\pm$ {delta_Ekin:.4f}"])
axs[0].set_xlabel("Steps")
axs[0].set_ylabel("Energy")

# Plot 2: Potenzielle Energie
axs[1].plot(Steps, Epot)
axs[1].plot([0, N], [Epot_mean, Epot_mean])
axs[1].set_title(f"Potenzielle Energie ({T=})")
axs[1].legend(["Samplepoints", f"Mean = {Epot_mean:.2f} $\\pm$ {delta_Epot:.2f}"])
axs[1].set_xlabel("Steps")
axs[1].set_ylabel("Energy")

plt.tight_layout()
plt.show()