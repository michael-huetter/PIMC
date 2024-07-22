# PIMC
Implementation of a Markov Chain Monte Carlo Algorithm to sample static finite temperature properties of molecules.

## Requirements
- Python 2.6 or higher
- numpy
- numba
- tqdm
- arviz
- joblib
- scikit-learn
- scipy
- statsmodels

### Installation
Install the necessary packages using the command below:

```bash
pip install -r requirements.txt
```

## Input Parameters
Define your input parameters in the file input.in. Ensure that all units are atomic units.

T = <t1,t2,...,tn>  # Set the temperature for the simulation
numParticles = <>  # Specify the number of particles
lam = <1/(2*mass)>  # Lambda parameter, calculated as half the inverse of mass
n = <>  # Number of electronic states
delta = <>  # Controls acceptance ratio (Center of Mass move)
numTimeSlices = <>  # Number of beads in the path integral
m = <>  # Stage length (controls acceptance ratio of staging move)
numMCSteps = <>  # Total Monte Carlo sweeps
corrSkip = <20>  # Correlation skip
thermSkip = <2000>  # Thermalization skip
rand_seed = <>  # Random seed
echange = <true/false>  # Set to true if multiple electronic states are present
non_adiabatic_coupling = <true/false>  # Set to true to consider non-adiabatic effects
staging = <true/false>  # Set to true if staging should be enabled
delta_bead = <>  # Controls acceptance ratio for individual bead moves
PoE = <true/false>  # True if propagation of Excitation moves should be used
eCG = <>  # Parameters related to skipping global e-changes
eCL = <>  # Parameters related to skipping local e-changes
