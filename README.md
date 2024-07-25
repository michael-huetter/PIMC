# PIMC
Implementation of a Markov Chain Monte Carlo Algorithm to sample static finite temperature properties of molecules.

## Requirements
- Python 3.6 or higher
- numpy
- numba
- tqdm
- arviz
- joblib
- scikit-learn
- scipy
- statsmodels

To include non-adiabatic effects using PoE moves:
- C compiler (e.g. GCC for LINUX/macOS or MinGW for Windows)

### Installation
Install the necessary python packages using the command below:

```bash
pip install -r requirements.txt
```

#### Compiling the C Library
To perform PoE moves the C code neds to be compiled to create the necessary shared library. Do this by running the following command on Linx/macOS:

```bash
gcc -shared -fPIC -o lib_PoE.so PoE.c
```

or on Windows:

```bash
gcc -shared -o lib_PoE.dll PoE.c
```

Note that if you are using Windows you further have to change how the shared library is loded using ctypes by changing the following line of code in `main.py`:

```python
lib = ctypes.CDLL('./lib_PoE.so')
```

to use a .dll (Dynamic Link Librarie) instead of the usual shared object files of Linux/macOS.

## Input Parameters
Define your input parameters in `input.in`. Ensure that atomic units are used.

| Parameter                 | Value              | Description                                                  |
|---------------------------|--------------------|--------------------------------------------------------------|
| `T`                       | `t_1,...,t_n`| Set the temperature for the simulation. Multiple temperature loops 1,..,n can be run in parallel            |
| `numParticles`            |          | Specify the number of particles                              |
| `lam`                     | `1/(2*m_1), ..., 1/(2*m_n)`     | Lambda, calculated as half the inverse of mass for particle 1,...,n     |
| `n`                       |          | Number of electronic states                                  |
| `delta`                   |           | Controls acceptance ratio (Center of Mass move)              |
| `numTimeSlices`           |          | Number of beads in the path integral                         |
| `m`                       |          | Stage length (controls acceptance ratio of staging move)     |
| `numMCSteps`              |          | Total Monte Carlo sweeps                                     |
| `corrSkip`                | `20`               | Correlation skip                                             |
| `thermSkip`               | `2000`             | Thermalization skip                                          |
| `rand_seed`               |            | Random seed                                                  |
| `echange`                 | `true/false`       | Set to true if multiple electronic states are present        |
| `non_adiabatic_coupling`  | `true/false`       | Set to true to consider non-adiabatic effects                |
| `staging`                 | `true/false`       | Set to true if staging should be enabled                     |
| `delta_bead`              |           | Controls acceptance ratio for individual bead moves          |
| `PoE`                     | `true/false`       | True if propagation of Excitation moves should be used       |
| `eCG`                     |         | Parameter related to skipping global e-changes              |
| `eCL`                     |         | Parameter related to skipping local e-changes               |

## Example Usage

### System composed of two harmonic adiabats

```math
V_{\text{HO}} = \begin{bmatrix} 0.5 \cdot (x^2 + y^2 + z^2) & 0 \\ 0 & 0.5 \cdot (x^2 + y^2 + z^2) + 1 \end{bmatrix}
```

1) Define input parameters
2) Define potential in `potential.py`

```python
# getV is called from the main PIMC code
def getV(R, eState):
	if eState[0] == 0:
		return V_HO(R[0], 0)  # only one particle, thus r = R[0]
	else:
		return V_HO(R[0], 1)
	
# Compute analytic gradient (only necessary if Virial Estimator is used)
def getGradV(R, eState):
	return np.array([R[0][0], R[0][1], R[0][2]]) # x,y,z coordinates of particle
```

3) Run script with: 
```bash
python main.py
```

4) Analyze output in `debug.log` and `output/`

For more details see [pdf](https://ulb-dok.uibk.ac.at/urn/urn:nbn:at:at-ubi:1-148622).
