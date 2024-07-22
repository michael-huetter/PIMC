"""
Some statistics...
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
from helpers import get_filenames, remove_duplicates_floats, HO3DEnergyExact, write_debug_log, HO3DEnergyExactAdiab
from statsmodels.tsa.stattools import acf
import arviz as az
import configparser
config = configparser.ConfigParser()
config.read('input.in')
use_jit = str(config["settings"]["use_jit"]) 
numParticles = int(config["settings"]["numParticles"])  
kinVir = str(config["settings"]["kin_virial"]) 

def jackknife_after_bootstrap(data, num_resamples=1000):
    n = len(data)
    estimates = np.zeros(num_resamples)
    
    # Generate bootstrap resamples and compute the estimator for each resample
    for i in range(num_resamples):
        sample = np.random.choice(data, size=n, replace=True)
        estimates[i] = np.mean(sample)
    
    # True mean estimate using the original data
    true_mean = np.mean(data)
    
    # Bias of the bootstrap estimator
    bias = np.mean(estimates) - true_mean
    
    # Variance of the bootstrap estimator
    variance = np.var(estimates, ddof=1)
    
    adjusted_mean = true_mean - bias
    
    return true_mean, np.mean(estimates), adjusted_mean, bias, variance

def gelman_rubin(chains):
    """
    Gelman-Rubin diagnostic for MCMC chain convergence.
    """
    n = chains.shape[1]  # Number of samples per chain
    m = chains.shape[0]  # Number of chains

   
    means_per_chain = np.mean(chains, axis=1)
    overall_mean = np.mean(means_per_chain)
    
    # Calculate the between-chain variance
    B_over_n = np.sum((means_per_chain - overall_mean)**2) / (m - 1)
    # Calculate the within-chain variances
    W = np.sum([(chains[i] - means_per_chain[i])**2 for i in range(m)]) / (m * (n - 1))
    
    # Estimate of the marginal posterior variance
    var_hat = (n - 1) / n * W + B_over_n

    R_hat = np.sqrt(var_hat / W)
    
    return R_hat

def integrated_autocorrelation_time(chain, max_lag=30):
    """
    Computes the Integrated Autocorrelation Time (IAT) for an MCMC chain.
    max_lag may needs to be adjusted depending on the use case
    """

    autocorr = acf(chain, nlags=max_lag, fft=True)
    return 1 + 2 * np.sum(autocorr[1:])  # summing from lag 1 to max_lag

def eStatePop(data, T):
    """
    Compute the population of the individual eStates.
    """
    total_counts = {}
    for arr in data:
        unique, counts = np.unique(arr, return_counts=True)
        for num, count in zip(unique, counts):
            if num in total_counts:
                total_counts[num] += count
            else:
                total_counts[num] = count

    total_entries = sum(total_counts.values())
    percentages = {num: (count / total_entries) * 100 for num, count in total_counts.items()}
    for num, percentage in percentages.items():
        write_debug_log(f"({T}) eState: {num}, Populated: {percentage:.2f}%")


def rcc():
    files = get_filenames("output")
    T = []
    name = []

    write_debug_log(f"Some statistics on the energy estimator...")
    for file in files[1]:
        temp = file.split("_")[0]    
        type = file.split("_")[1]

        if temp not in T:
            T.append(temp)
        if type not in name:
            name.append(type)

    if len(T) == 0:
            write_debug_log(f"Error: NDF")
            exit()

    # Get toatal E
    E = []
    E_trace = []
    for i in T:
        kin = np.loadtxt("output/"+i+"_KinEnergyTrace.csv")
        pot = np.loadtxt("output/"+i+"_PotEnergyTrace.csv")
        e = np.mean(kin+pot)

        # JKNF resampling
        true_mean, estimated_mean, adjusted_mean, bias, variance = jackknife_after_bootstrap(kin+pot)
        write_debug_log(f"({i}) Mean: {true_mean}(+/-){variance}; Estimated Mean: {estimated_mean}; Bias: {bias}")

        # Compute IAT
        iat = integrated_autocorrelation_time(kin+pot)
        write_debug_log(f"({i}) IAT: {iat:.2f}; (max_lag=30)")
        ess = az.ess(kin+pot)
        write_debug_log(f"({i}) ESS/TOT: {(ess/len(kin+pot)):.2f}")
        #if ess/len(kin+pot) < 0.95 or ess/len(kin+pot) > 1:
        #    write_debug_log(f"Warning: Possible correlation problems for the ({i}) chain")


        E_trace.append(kin+pot)
        E.append([float(i), e])

    E = np.array(E)
    E_trace = np.array(E_trace)

    # Compute R-hat with Gerlman-Rubin statistic
    if len(E_trace) > 1:
        R = gelman_rubin(E_trace)
        write_debug_log(f"Convergance: R-hat = {R:.2f}")
        #if R > 1.1:
        #    write_debug_log(f"Warning: Chain most likely not converged properly") 
    else:
        write_debug_log(f"Warning: GR statistic not possible to compute with one chain. Pleas check convergance manually.")  

    write_debug_log(f"Population of the eStates")
    for i in T:
        # Print eState population
        data = np.loadtxt(f"output/{i}_eStatTrace.csv", delimiter=",")
        eStatePop(data, i)

    # mean bond length for simple diatomic molecules
    if numParticles == 1:
        for i in T:
            r0 = np.loadtxt(f"output/{i}_PositionTrace.csv", delimiter=",")
            write_debug_log(f"({i}) Mean bond length r_0 = {np.mean(r0)}(+/-){np.var(r0)}") 

rcc()
    
