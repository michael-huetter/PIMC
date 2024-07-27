# short script to benchmark the PIMC code against the exact solution for the 3D harmonic oscillator.

import os

import numpy as np
import matplotlib.pyplot as plt

from helpers import get_filenames

path_to_python_inptr = "/usr/local/bin/python3"

# Exact result for the 3D harmonic oscillator
def HO3DEnergyExact(T):
    """
    The exact HO energy.
    """

    return (3/2)/np.tanh(0.5/T)

def HO3DEnergyExactAdiab(T, dE):
    """
    The exact HO energy with two adiabats seperated by a energy of dE.
    """

    return (3/2)/np.tanh(0.5/T) + dE / (np.exp(dE/T) + 1)

def HOExactDist(q, T):

    dq = (0.5/np.tanh(0.5/T))**0.5
    return 1/((2*np.pi)**0.5*dq) * np.exp(-0.5*q**2/dq**2)


def read_output():

    T_pimc = []
    U = []
    K = []
    R = []
    e = []
     
    files = get_filenames("output")
    for file in files[1]:
            if file.split("_")[0] not in T_pimc:
                T_pimc.append(file.split("_")[0])

    for t in T_pimc:
        K.append(np.mean(np.loadtxt("output/"+t+"_KinEnergyTrace.csv")))
        U.append(np.mean(np.loadtxt("output/"+t+"_PotEnergyTrace.csv")))
        R.append(np.loadtxt("output/"+t+"_PositionTrace.csv", delimiter=","))
        e.append(np.loadtxt("output/"+t+"_eStatTrace.csv", delimiter=","))

    T_pimc = np.array(T_pimc, dtype=float)
    U = np.array(U)
    K = np.array(K)
    R = np.array(R)
    e = np.array(e)

    return T_pimc, U, K, R, e


###############Harmonic Oscillator with one adiabat.####################

print("Running PIMC for 3D Harmonic Oscillator with one adiabat.")
os.system("cp sample_input/potential_HO.py potential.py")
os.system("cp sample_input/input_HO.in input.in")
os.system(path_to_python_inptr + " main.py")

# plotting

T = np.linspace(0.1, 6, 1000)
x = np.linspace(-13, 13, 1000)

T_pimc, U, K, R, e = read_output()

fig, axs = plt.subplots(3)
fig.suptitle("3D Harmonic Oscillator")
axs[0].plot(T, HO3DEnergyExact(T), "b-",label="Exact")
axs[0].plot(T_pimc, U+K, "ro",label="PIMC")
axs[0].set_xlabel(r"$T$ in natural units")
axs[0].set_ylabel(r"$E$ in natural units")
axs[0].legend()
axs[1].plot(x, .5*x**2, "b-",label="Potential")
#axs[1].hist(R, bins=100, density=True, histtype='step', color='r',label="PIMC")

i = 0
for t in T_pimc:
    axs[1].hlines(t, -np.sqrt(2*t), np.sqrt(2*t), color='k', linestyle='--', label=f"T={t}")
    axs[2].hist(R[i,:,0], bins=100, histtype='step', color='r', density=True)
    axs[2].hist(R[i,:,1], bins=100, histtype='step', color='g', density=True)
    axs[2].hist(R[i,:,2], bins=100, histtype='step', color='b', density=True)
    axs[2].plot(x, np.array([HOExactDist(q, t) for q in x]), "k-", label="Exact")
    i += 1

axs[1].legend()
axs[2].set_xlabel(r"$x$ in natural units")
axs[2].legend()

plt.show()


###############Harmonic Oscillator with two adiabats.####################

dE = 2

print("Running PIMC for 3D Harmonic Oscillator with two adiabats.")
os.system("cp sample_input/potential_HO_two_adiabats.py potential.py")
os.system("cp sample_input/input_HO_two_adiabats.in input.in")
os.system(path_to_python_inptr + " main.py")

T_pimc, U, K, R, e = read_output()

fig, axs = plt.subplots(3)
fig.suptitle("3D Harmonic Oscillator with two adiabats")
axs[0].plot(T, HO3DEnergyExactAdiab(T, dE), "b-", label="Exact two adiabats")
axs[0].plot(T, HO3DEnergyExact(T), "g-",label="Exact one adiabat")
axs[0].plot(T_pimc, U+K, "ro",label="PIMC")
axs[0].set_xlabel(r"$T$ in natural units")
axs[0].set_ylabel(r"$E$ in natural units")
axs[0].legend()
axs[1].plot(x, .5*x**2, "b-",label="V1")
axs[1].plot(x, .5*x**2+2, "b-",label="V2")
axs[1].legend()

plt.show()

###############Harmonic Oscillator classical limit.####################

print("Running PIMC simulation in the classical limit of P=1.")
os.system("cp sample_input/potential_HO.py potential.py")
os.system("cp sample_input/input_HO_classical.in input.in")
os.system(path_to_python_inptr + " main.py")

T_pimc, U, K, R, e = read_output()

fig, axs = plt.subplots(3)
fig.suptitle("PIMC in the classical limit of P=1")
axs[0].plot(T, 3*T, "b-", label="Classical HO")
axs[0].plot(T, HO3DEnergyExact(T), "g-",label="Quantum HO")
axs[0].plot(T_pimc, U+K, "ro",label="PIMC")
axs[0].set_xlabel(r"$T$ in natural units")
axs[0].set_ylabel(r"$E$ in natural units")
axs[0].legend()
axs[1].plot(x, .5*x**2, "b-",label="Potential")
axs[1].legend()

plt.show()