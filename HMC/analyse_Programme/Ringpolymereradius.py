import numpy as np
import matplotlib.pyplot as plt
import analysis_funktions as af
import configparser

config = configparser.ConfigParser()
config.read("../input.in")

T_s = config.get("system", "T")
T = [float(x) for x in T_s.split(",")]

HMC = 1
max_l = 500
skip = 200

Rg2_List = []
Err_List = []

def analytic_rg2_3d(T):
    # reduzierte Einheiten: hbar = kB = m = omega = 1
    return 3.0 * (0.5 / np.tanh(1.0 / (2.0 * T)) - T)

for Ti in T:
    # Erwartete Form: (MC_steps, P, numParticles, simulation_dim)
    pos = np.load(f"../output_{HMC}/{Ti}_PositionTrace.npy")
    pos = pos[skip:]

    # Schwerpunkt des Ringpolymers pro MC-Schritt
    centroid = np.mean(pos, axis=1, keepdims=True)

    # Abstand jedes Beads vom Schwerpunkt
    dR = pos - centroid

    # |R_k - R_c|^2
    dR2 = np.sum(dR**2, axis=-1)

    # Mittel über Beads und Teilchen pro MC-Schritt
    Rg2_trace = np.mean(dR2, axis=(1, 2))

    Rg2_mean = np.mean(Rg2_trace)

    this_max_l = min(max_l, len(Rg2_trace) - 1)
    acf = af.autokorr(Rg2_trace, this_max_l)
    tau_int = af.integrat_atokorr(acf)

    sigma = np.std(Rg2_trace, ddof=1)
    N = len(Rg2_trace)
    Rg2_err = sigma * np.sqrt(2.0 * tau_int / N)

    Rg2_List.append(Rg2_mean)
    Err_List.append(Rg2_err)

T_plot = np.linspace(0, 11, 500)
Rg2_analytic = analytic_rg2_3d(T_plot)

plt.figure()
plt.plot(T, Rg2_List,"ob",label = "HMC")
plt.errorbar(T, Rg2_List, yerr=Err_List, linestyle = "none", label="Fehlerbalken",color = "red")
plt.plot(T_plot, Rg2_analytic, label="Analytisch",color = "orange")

plt.xlabel("Temperatur T")
plt.ylabel(r"$R_g^2$")
plt.title("Ringpolymer-Radius / quantenmechanische Delokalisierung")
plt.grid(linestyle="--")
plt.axis([0,10.5,0,0.9])
plt.legend()
plt.show()
