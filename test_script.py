# short script to benchmark the PIMC code against the exact solution for the 3D Harmonic Oscillator.

import os
from time import time

import numpy as np
import matplotlib.pyplot as plt

from helpers import get_filenames

path_to_python_inptr = "/usr/local/bin/python3"

# Some colors for the terminal output (may not work on some Windows terminals without ANSI Escape Codes)
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"

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


###############General.####################

print(BLUE + "General information." + RESET)
cpu_c = os.cpu_count()
if cpu_c < 3:
    print(RED + f"Error: Not enough CPUs avalible to run this script. Only {cpu_c} avalible, and 3 needed." + RESET)
print(GREEN + f"Number of avalible CPUs: {cpu_c}" + RESET)

###############Harmonic Oscillator with one adiabat.####################

print(BLUE + "Running PIMC for 3D Harmonic Oscillator with one adiabat." + RESET)
os.system("cp sample_input/potential_HO.py potential.py")
os.system("cp sample_input/input_HO.in input.in")
t0 = time()
os.system(path_to_python_inptr + " main.py")
t1 = time()
print(GREEN + f"Runtime: {t1-t0:.2f} s" + RESET)

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

print(BLUE + "Running PIMC simulation for 3D Harmonic Oscillator with two adiabats." + RESET)
os.system("cp sample_input/potential_HO_two_adiabats.py potential.py")
os.system("cp sample_input/input_HO_two_adiabats.in input.in")
t0 = time()
os.system(path_to_python_inptr + " main.py")
t1 = time()
print(GREEN + f"Runtime: {t1-t0:.2f} s" + RESET)

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

groundstat_count = []
tot_count = []
for sub_array in e:
    count = 0
    tot = 0
    for arr in sub_array:
        if np.all(arr == 0):
            count += 1
        tot += 1
    tot_count.append(tot)
    groundstat_count.append(count)

groundstat_count = np.array(groundstat_count)
tot_count = np.array(tot_count)
excited_count = tot_count - groundstat_count
bottom = groundstat_count

axs[2].bar(T_pimc, groundstat_count, label='Ground state')
axs[2].bar(T_pimc, excited_count, bottom=bottom, label='Excited state')
axs[2].legend()
axs[2].set_xlabel(r"$T$ in natural units")

plt.show()
 

###############Harmonic Oscillator classical limit.####################

print(BLUE + "Running PIMC simulation simulation for the Harmonic Oscillator the classical limit of P=1." + RESET)
os.system("cp sample_input/potential_HO.py potential.py")
os.system("cp sample_input/input_HO_classical.in input.in")
t0 = time()
os.system(path_to_python_inptr + " main.py")
t1 = time()
print(GREEN + f"Runtime: {t1-t0:.2f} s" + RESET)

T_pimc, U, K, R, e = read_output()

fig, axs = plt.subplots(3)
fig.suptitle(r"Classical 3D HO (classical limit of $P$=1)")
axs[0].plot(T, 3*T, "b-", label="Classical HO")
axs[0].plot(T, HO3DEnergyExact(T), "g-",label="Quantum HO")
axs[0].plot(T_pimc, U+K, "ro",label="PIMC")
axs[0].set_xlabel(r"$T$ in natural units")
axs[0].set_ylabel(r"$E$ in natural units")
axs[0].legend()
axs[1].plot(x, .5*x**2, "b-",label="Potential")
axs[1].legend()

i = 0
for t in T_pimc:
    axs[2].hist(R[i,:,0], bins=100, histtype='step', color='r', density=True)
    axs[2].hist(R[i,:,1], bins=100, histtype='step', color='g', density=True)
    axs[2].hist(R[i,:,2], bins=100, histtype='step', color='b', density=True)
    i += 1

plt.show()


###############Testing PoE moves.####################

dE = 2

print(BLUE + "Running PIMC simulation for 3D Harmonic Oscillator with two adiabats using PoE moves." + RESET)
os.system("cp sample_input/potential_HO_two_adiabats.py potential.py")
os.system("cp sample_input/input_HO_two_adiabats_PoE.in input.in")
t0 = time()
os.system(path_to_python_inptr + " main.py")
t1 = time()
print(GREEN + f"Runtime: {t1-t0:.2f} s" + RESET)

if not os.path.exists("output/0.5_eStatTrace.csv"):
    print(RED + "Error: PoE moves failed. Make sure the shared library is proparly compiled and referanced in main.py" + RESET)
    exit()

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

groundstat_count = []
tot_count = []
for sub_array in e:
    count = 0
    tot = 0
    for arr in sub_array:
        if np.all(arr == 0):
            count += 1
        tot += 1
    tot_count.append(tot)
    groundstat_count.append(count)

groundstat_count = np.array(groundstat_count)
tot_count = np.array(tot_count)
excited_count = tot_count - groundstat_count
bottom = groundstat_count

axs[2].bar(T_pimc, groundstat_count, label='Ground state')
axs[2].bar(T_pimc, excited_count, bottom=bottom, label='Excited state')
axs[2].legend()
axs[2].set_xlabel(r"$T$ in natural units")

plt.show()