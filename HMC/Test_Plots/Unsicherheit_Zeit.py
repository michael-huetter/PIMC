import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import analysis_funktions as af

monte_steps = 10000
HMC = 1
T = 0.5
start_step = 500
skip = 500
Zeit = 16 #gesammtzteit für eine Temeratur (gerade 85s bei HMC)
Zeit_step = Zeit/monte_steps

E_kin_roh = pd.read_csv(f"../output_{HMC}/{T}_KinEnergyTrace.csv")
E_pot_roh = pd.read_csv(f"../output_{HMC}/{T}_PotEnergyTrace.csv")
Pos_roh = pd.read_csv(f"../output_{HMC}/{T}_PositionObsTrace.csv")

Ekin = E_kin_roh.to_numpy()
Epot = E_pot_roh.to_numpy()
Pos = Pos_roh.to_numpy()

pot_arc = af.autokorr(Epot,1000)
kin_arc = af.autokorr(Ekin,1000)
Pos_arc = af.autokorr(Pos,1000)

tau_int_pot = af.integrat_atokorr(pot_arc)
tau_int_kin = af.integrat_atokorr(kin_arc)
tau_int_pos = af.integrat_atokorr(Pos_arc)


N = len(Ekin)-start_step

Ekin_array = np.empty((int(N/skip),int(N/skip)))
Epot_array = np.empty((int(N/skip),int(N/skip)))
Pos_array = np.empty((int(N/skip),int(N/skip)))

for i in range(int(N/skip)):
    Ni = int(N * i/skip)
    # Mittelwerte
    Ekin_array[0,i] = np.mean(Ekin[start_step:Ni+start_step])
    Epot_array[0,i] = np.mean(Epot[start_step:Ni+start_step])
    Pos_array[0,i] = np.mean(Pos[start_step:Ni+start_step])

    #Unsicherheit (atokorrelation beachten)
    Ekin_array[1,i] = np.sqrt(2 * tau_int_kin * np.var(Ekin[start_step:Ni+start_step]) / Ni)
    Epot_array[1,i] = np.sqrt(2 * tau_int_pot * np.var(Epot[start_step:Ni+start_step]) / Ni)
    Pos_array[1,i] = np.sqrt(2 * tau_int_pos * np.var(Pos[start_step:Ni+start_step]) / Ni)



def steps_Zeit(x):
    return x * Zeit_step

def Zeit_steps(x):
    return x / Zeit_step

steps = np.arange(int(N/skip)) * skip + start_step


fig, ax = plt.subplots(constrained_layout=True)

#E_max = max(np.array(Ekin_array[0][np.isfinite(Ekin_array[0])]))
#E_min = min(np.array(Ekin_array[0][np.isfinite(Ekin_array[0])]))
ax.plot(steps,Ekin_array[0],"or")
ax.errorbar(steps,Ekin_array[0],yerr=Ekin_array[1])
ax.yaxis.get_major_formatter().set_useOffset(False)
ax.grid(linestyle ="--")
#ax.set(xlim=(0, monte_steps), ylim=(E_min, E_max))
ax.set_title("Ekin-Mittelwert über Schritte")
secax = ax.secondary_xaxis('top', functions=(steps_Zeit,Zeit_steps))
secax.set_xlabel('Zeit (s)')

ax.set_xlabel('Steps')
plt.show()
ax.cla()

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(steps,Epot_array[0],"or")
ax.errorbar(steps,Epot_array[0],yerr=Epot_array[1])
ax.grid(linestyle ="--")
ax.set_title("Epot-Mittelwert über Schritte")
secax = ax.secondary_xaxis('top', functions=(steps_Zeit,Zeit_steps))
secax.set_xlabel('Zeit (s)')

ax.set_xlabel('Steps')
plt.show()
ax.cla()

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(steps,Pos_array[0],"or")
ax.errorbar(steps,Pos_array[0],yerr=Pos_array[1])
ax.grid(linestyle ="--")
ax.set_title("Pos-Mittelwert über Schritte")
secax = ax.secondary_xaxis('top', functions=(steps_Zeit,Zeit_steps))
secax.set_xlabel('Zeit (s)')

ax.set_xlabel('Steps')
plt.show()

