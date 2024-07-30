# Path Integral Monte Carlo
Implementation of a Path Integral Monte Carlo (PIMC) algorithm using a Markov Chain Monte Carlo method (Metropolis-Hastings) to sample static finite temperature properties of molecules from previously computed Potential Energy Surfaces (PES). Due to a generalized Path Integral expression of the canonical density matrix, given by:

```math
        Z = \lim_{P \to \infty } \prod_{I=1}^{N}\left[ \left( \frac{m_{I}P}{2\pi\beta\hbar ^{2}} \right)^{\frac{3P}{2}} \int dR^{1}_{I}... \int dR^{P}_{I} \right]  \times \text{exp}\left[ -\beta\sum_{k=1}^{P}\sum_{I=1}^{N}\frac{m_{I}}{2}\omega_{P}^{2}\left( R_{I}^{k}-R_{I}^{k-1} \right)^{2} \right]  \times \sum_{i_{1}}...\sum_{i_{P}}\prod_{k=1}^{P}\left\langle\Psi_{i_k}|\Psi_{i_{k-1}}\right\rangle e^{-\frac{\beta}{P} E_{i_k}},
```
both electronic and nuclear degrees of freedom are sampled naturally. Calculations are possible in both the adiabatic limit of uncoupled PESs and the diabatic limit, where non-adiabatic effects are included. The current implementation includes center of mass (CoM) moves, single bead moves, staging moves, global e-change moves, local e-change moves, and propagation of excitation (PoE) moves.

### CoM moves
The entire path is uniformly translated by a random vector. The acceptance propability is controlled with the `delta` parameter.
### Single bead moves
Individual beads are displaced by a random vector. The acceptance propability is controlled with the `delta_bead` parameter.
### Staging moves
Staging moves are applied to a section of path, where stage length and thus the acceptance probability can be controlled with the `m` parameter, while the kinetic action of the displaced path segment is sampled directly using
the Lévy construct.
### Global e-change moves
Electronic states of all beads simultaneously are changed if the parameter `echange` is set to `True`. With this multiple electronic states can be included in the adiabatic limit. 
### Local e-change moves
Electronic states of a single bead is changed if the parameter `non_adiabatic_coupling` is set to `True`. With this non-adiabatic effects can be included in the diabatic limit.
### PoE moves
A randomly choosen number of overlap terms 

```math
\left\langle\Psi_{i_k}|\Psi_{i_{k-1}}\right\rangle
```

is set periodically and excitations are propagated randomly trough the nacklace of beads while keeping the number of overlap terms fixed, if the `PoE` parameter is set to `True`. With this, non-adiabatic effects are included in the diabatic limit and no global/local e-change moves are necessary.

## Requirements
- Python 3.10 or higher
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
Install the necessary Python packages using the command below:

```bash
pip install -r requirements.txt
```
If you are using conda:
```bash
conda create --name pimc_env python=3.11 -y
conda activate pimc_env
conda install pip -y
pip install -r requirements.txt
```

#### Compiling the C Library
To perform PoE moves the PoE.c file needs to be compiled to create the necessary shared library. Do this by running the following command on Linx/macOS:

```bash
gcc -shared -fPIC -o lib_PoE.so PoE.c
```

Or on Windows:

```bash
gcc -shared -o lib_PoE.dll PoE.c
```

Note that if you are using Windows you further have to change how the shared library is loaded using ctypes by changing the following line of code in `main.py`:

```python
lib = ctypes.CDLL('./lib_PoE.so')
```

to use a .dll (Dynamic Link Library) instead of the usual shared object files of Linux/macOS.

## Input Parameters
Define your input parameters in `input.in`. Ensure that atomic units are used.

| Parameter                 | Value              | Description                                                  |
|---------------------------|--------------------|--------------------------------------------------------------|
| `T` | `t_1,...,t_n`| Set the temperature for the simulation. Multiple temperature loops 1,..,n can be run in parallel            |
| `numParticles` |          | Specify the number of particles                              |
| `lam` | `1/(2*m_1), ..., 1/(2*m_n)` | Lambda, calculated as half the inverse of mass for particle 1,...,n     |
| `n` |          | Number of electronic states                                  |
| `delta` |           | Controls acceptance ratio (Center of Mass move)              |
| `numTimeSlices` |          | Number of beads in the path integral                         |
| `m` |          | Stage length (controls acceptance ratio of staging move)     |
| `numMCSteps` |          | Total Monte Carlo sweeps                                     |
| `corrSkip` | `20` | Correlation skip                                             |
| `thermSkip` | `2000` | Thermalization skip                                          |
| `rand_seed` |            | Random seed                                                  |
| `echange` | `true/false` | Set to true if multiple electronic states are present        |
| `non_adiabatic_coupling` | `true/false` | Set to true to consider non-adiabatic effects                |
| `staging` | `true/false` | Set to true if staging should be enabled                     |
| `delta_bead` |           | Controls acceptance ratio for individual bead moves          |
| `PoE` | `true/false` | True if propagation of Excitation moves should be used       |
| `eCG` |         | Every n-th step a global e-change is attempted    (adiabatic limit)        |
| `eCL` |         | Every n-th step a local e-change is attempted  (diabatic limit)             |
| `logging` |    `true/false`     | Logs attempted moves for debugging             |

## Example Usage

The general workflow is as follows:

1) Find a continuous representation of the PESs. This typically involves firstly conducting quantum chemical calculations to sample the PES and subsequantly fitting it using for example simple interpolation schemes or more sophisticated methods like Neural Networks. Note that if non-adiabatic effects should be considered, an additional quasi-diabatization transformation needs to be applied.

2) Define the input parameters in `input.in`. Note that in the code natural units are used, so it is recommended to use atomic units if simulations are run for molecules.

3) Secify your potential in `potential.py`. Firstly, the main PIMC code calls the function,

```python
getV(R, eState)
```

where `R` is a three-dimensional array containing the xyz - coordinates of bead $k$ and particle I with [x,z,z] = R[I,k], and eState is an intager containing the index of the PES on which bead k is currently located. Finally, this function should return the numerical value of the potential consistent with the current positions and occupied electronic states. Depending on how the potentials were computed, the function needs to be adjusted accordingly.

Secondly, if the virial estimator for the kinetic energy is used, the gradient of the potential needs be be computed. Therefor the function 

```python
getGradV(R, eState)
```

is called, where the input parameters are equal to the getV function, and the gradient should be returned. Thirdly, if non-adiabatic effects should be considered, the function 

```python
getDiabV(R)
```

is called and should return the diabatic matrix elements as a tuple, for example, 

```python
def getDiabV(R):
	...
	return V11, V12, V21, V22 
```


in the case of two PESs.

4) Run the simulation with 

```bash
python main.py
```

5) Analyze output in `output.out` and `output/`

### Harmonic Oscillator
Sample input files for `input.in` and `potential.py` can be found in the `sample_input/` folder for the Harmonic Oscillator with one and two adiabats. The `test_script.py` automatically runs PIMC simulations for these input files and plots some of the results using matplotlib. This acts as a test if everything is installed properly and behaves as expected.