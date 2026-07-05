import numpy as np


def autokorr(observable,max_l):
    mean = np.mean(observable)
    x = observable - mean
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    var = np.dot(x,x)/n

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