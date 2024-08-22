"""
Define your potential here. getV functino is calles from the main MCMC loop!!!
"""

import numpy as np
from numba import njit, jit
import configparser
import joblib
import torch
import torch.nn as nn

from projToINRC import proj_main

from NN.model_architechture import Molecule_NN
from line_profiler import LineProfiler
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
from helpers import wOut

config = configparser.ConfigParser()
config.read('input.in')
use_jit = config.getboolean("PIMC", "use_jit") 

calculate_errors = config.getboolean("NN", "calculateErrors")
use_multiprocessing = config.getboolean("NN", "useMultiprocessing")
err_threshold = config.getfloat("NN", "ErrorThreshold")

############################ Hyperparameters ###########################
# Device configuration
device = "cpu"  # Change to "cuda" or "mps" if using GPU

# Neural network architecture
input_dim = 3
hidden_dims = [20]
output_dim = 2  # Potential energy output

# paths
model_dir = "NN/models/"
pattern = re.compile(r'^model\d+\.pth$')

# List to store matching filenames
matching_files = []


# Iterate over all files in the directory
for filename in os.listdir(model_dir):
    if pattern.match(filename):
        matching_files.append(filename)

if calculate_errors:
    models = [Molecule_NN(input_dim, hidden_dims, output_dim) for _ in matching_files]
else:
    models= [Molecule_NN(input_dim, hidden_dims, output_dim)]
for i, model in enumerate(models):
    model.load_state_dict(torch.load(model_dir+matching_files[i], map_location=torch.device('cpu')))
    model.to(device)  # Send model to the device

# Load the scalers
scalers = joblib.load("NN/models/scalers.pkl")
scaler_X = scalers['scaler_X']
scaler_Y = scalers['scaler_Y']

def cJIT(func):

    if use_jit:
        return njit()(func)
    else:
        return func


############Morse potential of H2@FCI/aug-cc-pvDZ#######################

@cJIT
def H2_morse(R):
   
    r = np.linalg.norm(R[0,:]) 
    alpha = 2.02881752 / 1.8897268777744
    r_e = 0.76345602 * 1.8897268777744
    A = 0.17089793

    return A * (1 - np.exp(-alpha * (r - r_e)))**2

@cJIT
def H2_morse_grad(R):

    r = np.linalg.norm(R[0,:]) 
    alpha = 2.02881752  / 1.8897268777744
    r_e = 0.76345602 * 1.8897268777744
    A = 0.17089793

    return 2 * A * alpha * (1 - np.exp(-alpha * (r - r_e))) * np.exp(-alpha * (r - r_e)) * (R/r)

############ Additional functions #######################
def evaluate_model(model, R_tensor):
    with torch.no_grad():
        E_scaled = model(R_tensor)
        return E_scaled.numpy().reshape(-1, 1) if E_scaled.shape[1] == 1 else E_scaled

############Called from main code#######################   
@cJIT

def getV(R: np.array, eState: int) -> float:
    R_scaled = scaler_X.transform(R.reshape(-1, R.shape[-1]))
    R_tensor = torch.tensor(R_scaled, dtype=torch.float32, device=device)
    
    if calculate_errors:
        if use_multiprocessing:
            with ThreadPoolExecutor() as executor:
                # Run models in parallel using ProcessPoolExecutor
                results = list(executor.map(evaluate_model, models, [R_tensor]*len(models)))
        else:
            # Run models sequentially without multiprocessing
            results = [evaluate_model(model, R_tensor) for model in models]

        # Convert to a numpy array
        E_ar = np.array(results)

        # Error calculation
        errors = np.std(E_ar, axis=0)
        for i in range(errors.shape[0]):  # Loop over positions
            for j in range(errors.shape[1]):  # Loop over energy states
                error = errors[i, j]
                if error > err_threshold:
                    wOut(f'Warning: at bead position {R[i]} and eState {j}, the error reaches {error*100:.3f}%')

        # Convert the first prediction back to the original scale
        E_orig = scaler_Y.inverse_transform(E_ar[0])
    
    else:
        E_scaled = evaluate_model(models[0], R_tensor)
        # Convert prediction back to original scale
        E_orig = scaler_Y.inverse_transform(E_scaled)

    E = E_orig[np.arange(E_orig.shape[0]), eState].flatten() if E_orig.shape[1] != 1 else E_orig.flatten()   
    
    return E
        

    
@cJIT   
def getGradV(R: np.array, eState: int) -> np.array:
    R_scaled = scaler_X.transform(R.reshape(-1, R.shape[-1]))

    R_tensor = torch.tensor(R_scaled, dtype=torch.float32, device=device, requires_grad=True)
    
    # Forward pass to calculate the energy
    E_scaled = models[0](R_tensor)

    grad_outputs = torch.ones_like(E_scaled)

    # Backward pass to calculate the gradient of the energy 
    E_scaled.backward(grad_outputs)
    grad_R_scaled = R_tensor.grad.cpu().numpy()
    
    # Convert the output back to NumPy and inverse transform it to get the original scale
    grad_R_original = grad_R_scaled / scaler_X.scale_
    return grad_R_original




@cJIT
def getDiabV(R: np.array) -> tuple:
    """
    Only needed in the diabatic limit. So if non_adiabatic_coupling or PoE is used.
    """

    pass

""" # Example data
R = np.array([[1,2,3],[4,3,5],[1,6,2],[-5,3,-4],[-1,-2,-3],[4,-4,4],[5,-3,-1],[-6,7,1]])
eState = 1

# Profile the functions
profiler = LineProfiler()
profiler.add_function(getV)
profiler.add_function(getGradV)

# Execute the functions to profile them
profiler.run('getV(R, eState)')
profiler.run('getGradV(R, eState)')

# Print the profiling results
profiler.print_stats() """