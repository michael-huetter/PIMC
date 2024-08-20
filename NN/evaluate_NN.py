import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib  # To load the scalers

from model_architechture import Molecule_NN

####################### Hyperparameters #######################

# Device configuration
device = "cpu"  # Change to "cuda" or "mps" if using GPU

# Neural network architecture
input_dim = 3
hidden_dims = [20]
output_dim = 1  # Potential energy output

# Model and scalers paths
model_path = "NN/models/model.pth"
scalers_path = "NN/models/scalers.pkl"  # Path to load scalers

####################### Data Preparation #####################

def calculate_data(num_points: int) -> np.array:
    x = np.linspace(0, 0, num_points)
    y = np.linspace(0, 0, num_points)
    z = np.linspace(-10, 10, num_points)

    PE = 0.5 * (x**2 + y**2 + z**2)

    Positions = np.column_stack((x, y, z))
    potential_energies = PE

    return Positions, potential_energies

X, E = calculate_data(10000) 

####################### Load Models and Scalers #######################

# Load the trained model
model = Molecule_NN(input_dim, hidden_dims, output_dim)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.to(device)  # Send model to the device

# Load the scalers
scalers = joblib.load(scalers_path)
scaler_X = scalers['scaler_X']
scaler_Y = scalers['scaler_Y']

####################### Evaluation #######################

def evaluate_model(model, X_new):
    model.eval()
    with torch.no_grad():
        predictions_scaled = model(X_new)
        predictions = scaler_Y.inverse_transform(predictions_scaled.cpu().numpy())
    return predictions

# Scale the input data
X_scaled = scaler_X.transform(X)

# Convert to torch tensor
X_new_tensor = torch.from_numpy(X_scaled).float().to(device)

# Evaluate the model
predictions = evaluate_model(model, X_new_tensor)

####################### Plotting #######################

def plot_potential(predictions, E):
    plt.figure(figsize=(10, 5))
    plt.title("Potential Energy: Model Predictions vs. harmonic oscillator")
    plt.plot(np.linspace(-10,10, 10000), predictions, color="red", label="Model Predictions")
    plt.plot(np.linspace(-10,10, 10000), E, color="black", label="harmonic oscillator")
    plt.xlabel("Sample Index")
    plt.ylabel("Potential Energy")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_potential(predictions, E)