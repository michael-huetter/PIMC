"""
Define your potential here. getV functino is calles from the main MCMC loop!!!
"""

import numpy as np
from numba import njit
import configparser
import joblib
import torch
import torch.nn as nn

from projToINRC import proj_main

from NN.model_architechture import Molecule_NN


config = configparser.ConfigParser()
config.read('input.in')
use_jit = config.getboolean("PIMC", "use_jit") 

# Device configuration
device = "cpu"  # Change to "cuda" or "mps" if using GPU

# Neural network architecture
input_dim = 3
hidden_dims = [20]
output_dim = 1  # Potential energy output

# Load the trained model
model = Molecule_NN(input_dim, hidden_dims, output_dim)
model.load_state_dict(torch.load("NN/models/model.pth", map_location=torch.device('cpu')))
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

############Called from main code#######################
    
@cJIT
def getV(R: np.array, eState: int) -> float:
    """ R=torch.from_numpy(R.squeeze()).to(device)
    # Scale the input data
    R_scaled = scaler_X.transform(R)

    # Convert to torch tensor
    R_tensor = torch.from_numpy(R_scaled).float().to(device)
    with torch.no_grad():
        E_scaled = model(R_tensor)
        E = scaler_Y.inverse_transform(E_scaled.cpu().numpy()).flatten() """
    E_ar = []
    for elm in R.squeeze():
        E = 0.5*(elm[0]**2+elm[1]**2+elm[2]**2)
        E_ar.append(E)
    return E_ar

    
@cJIT   
def getGradV(R: np.array, eState: int) -> np.array:
    # Convert R to a PyTorch tensor with gradients enabled
    R_tensor = torch.tensor(R, dtype=torch.float32, device=device, requires_grad=True)
    
    # Scale the input data using the scaler, but keep the tensor's gradient capability
    R_scaled = scaler_X.transform(R_tensor.detach().cpu().numpy())
    R_tensor_scaled = torch.tensor(R_scaled, dtype=torch.float32, device=device, requires_grad=True)
    
    # Forward pass to calculate the energy
    E_scaled = model(R_tensor_scaled)
    
    # If E_scaled is 2D, you may need to flatten it
    # E_scaled_flat = E_scaled.view(-1)  # This may or may not be necessary

    # Create a gradient tensor for backward, all ones for sum of gradients
    grad_outputs = torch.ones_like(E_scaled)
    
    # Backward pass to calculate the gradient of the energy w.r.t R
    E_scaled.backward(grad_outputs)
    
    # Get the gradient from R_tensor_scaled
    grad_R_scaled = R_tensor_scaled.grad.cpu().numpy()

    # Transform the gradient back to the original scale using the inverse of the scaler's transform
    grad_R_original = grad_R_scaled / scaler_X.scale_

    return grad_R_original


@cJIT
def getDiabV(R: np.array) -> tuple:
    """
    Only needed in the diabatic limit. So if non_adiabatic_coupling or PoE is used.
    """

    pass