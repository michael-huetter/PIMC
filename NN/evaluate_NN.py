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
model_path = "NN/models/model2.pth"
scalers_path = "NN/models/scalers.pkl"  # Path to load scalers
energies_path = "NN/data/energies_300K.dat"
positions_path = "NN/data/movie_300K.xyz"

####################### Data Preparation #####################
#Positions = [[[0,0,0], [1.4335500179, 0.000000, 0.95295864922], [1.4335500179, 0.000000, -0.95295864922]]]
def read_energies(file_path):
    lines = np.loadtxt(file_path)
    ar = lines[:,1]
    return ar

def read_positions(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Process the file lines to extract the coordinates
    data = []
    for i, line in enumerate(lines):
        if line.strip().startswith(('O', 'H')):
            data.append([float(x) for x in line.split()[1:]])

    # Convert the data into a NumPy array and reshape it
    num_time_steps = len(data) // 3  # Each time step has 3 coordinate sets (O, H, H)
    ar = np.array(data).reshape((num_time_steps, 3, 3))
    return ar

Energies = read_energies(energies_path)
Positions = np.array(read_positions(positions_path))*1.88973
r_H, avg_OH, prod_OH = [], [], []
for coords in Positions:
    O, H1, H2 = np.array(coords[0]), np.array(coords[1]), np.array(coords[2])
    d_HH = np.linalg.norm(H1 - H2)
    d_OH1 = np.linalg.norm(O - H1)
    d_OH2 = np.linalg.norm(O - H2)

    r_H.append(d_HH)
    avg_OH.append((d_OH1+d_OH2)/2)
    prod_OH.append(d_OH1*d_OH2)

# Convert to numpy arrays
r_H = np.array(r_H).reshape(-1, 1)
avg_OH = np.array(avg_OH).reshape(-1, 1)
prod_OH = np.array(prod_OH).reshape(-1, 1)

# Feature matrix
X = np.hstack((r_H, avg_OH, prod_OH))

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

def plot_potential(predictions):
    plt.figure(figsize=(10, 5))
    plt.title("Potential Energy: Model Predictions vs. harmonic oscillator")
    plt.plot(predictions, color="red", label="Model Predictions")
    plt.plot(Energies, color="blue", label="actual Data")
    plt.xlabel("Sample Index")
    plt.ylabel("Potential Energy")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_potential(predictions)