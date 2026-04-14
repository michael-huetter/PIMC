import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


HMC = 1
T = 2.0



def autokorr(observable):
    mean = np.mean(observable)
    x = observable - mean
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    var = np.dot(x,x)/n
    max_l = 1000

    acf = np.empty(max_l+1)

    for l in range(max_l+1):
        acf[l] = np.dot(x[:n-l],x[l:])/(n-l)

    acf /= var
    return acf

def integrat_atokorr(acf):
    tau_int = 0.5
    for t in range(1, len(acf)):
        if acf[t] <= 0:
            break
        tau_int += acf[t]
    return tau_int

E_kin_roh = pd.read_csv(f"../output_{HMC}/{T}_KinEnergyTrace.csv")
E_pot_roh = pd.read_csv(f"../output_{HMC}/{T}_PotEnergyTrace.csv")
Pos_roh = pd.read_csv(f"../output_{HMC}/{T}_PositionObsTrace.csv")


Ekin = E_kin_roh.to_numpy()
Epot = E_pot_roh.to_numpy()
Pos = Pos_roh.to_numpy()


pot_arc = autokorr(Epot)
kin_arc = autokorr(Ekin)
Pos_arc = autokorr(Pos)


tau_int_pot = integrat_atokorr(pot_arc)
tau_int_kin = integrat_atokorr(kin_arc)
tau_int_pos = integrat_atokorr(Pos_arc)


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