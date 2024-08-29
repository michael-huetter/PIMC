import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib  # To load the scalers

####################### Hyperparameters #######################

# Device configuration
device = "cpu"  # Change to "cuda" or "mps" if using GPU

# Neural network architecture
input_dim = 3
hidden_dims = [20]
output_dim = 1  # Potential energy output

# Model and scalers paths
model_path = "NN/models/model1.pth"
scalers_path = "NN/models/scalers.pkl"  # Path to load scalers

####################### Data Preparation #####################

def read_scan(file_path, R1_start=0.1, R1_end=5, num_points=1000):

    # Read the file and extract R1 and A1
    with open(file_path, 'r') as file:
        lines = file.readlines()

    R1 = None
    A1 = None

    for line in lines:
        if line.startswith("R1 ="):
            R1 = float(line.split('=')[1].strip())
        if line.startswith("A1 ="):
            A1 = float(line.split('=')[1].strip())

    if R1 is None or A1 is None:
        raise ValueError("R1 or A1 not found in the input file.")

    # Convert angle A1 from degrees to radians
    A1_rad = np.radians(A1)

    # Position of O is at the origin
    O = np.array([0.0, 0.0, 0.0])

    # Position of H2 based on R1 along the x-axis
    H2 = np.array([R1, 0.0, 0.0])

    # Initialize the positions list
    positions = []

    # Generate the range of R1 values for H1
    R1_values = np.linspace(R1_start, R1_end, num_points)

    for R1_val in R1_values:
        # Compute the position of H1 using the angle A1 and varying distance R1_val
        H1_x = R1_val * np.cos(A1_rad)
        H1_y = R1_val * np.sin(A1_rad)
        H1 = np.array([H1_x, H1_y, 0.0])

        # Append the positions to the list
        positions.append(np.array([O, H1, H2]))

    return np.array(positions)

# Calculate distances and features
def calculate_polynomes(positions):
    r_H, r_mean, r_HOxHO = [], [], []
    for coords in positions:
        O, H1, H2 = coords[0], coords[1], coords[2]
        d_HH = np.linalg.norm(H2 - H1)
        d_OH1 = np.linalg.norm(H1 - O)
        d_OH2 = np.linalg.norm(H2 - O)
        r_H.append(d_HH)
        r_mean.append((d_OH1 + d_OH2) / 2)
        r_HOxHO.append(d_OH1 * d_OH2)
    return np.array(r_H), np.array(r_mean), np.array(r_HOxHO)

pos = read_scan('NN/data/waterPES.com')*1.88973
print(pos)

####################### Neural Network Model #######################

class Molecule_NN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(Molecule_NN, self).__init__()
        
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        # Create hidden layers with ReLU activation and Dropout
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.Sigmoid())
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

####################### Load Models and Scalers #######################

# Load the trained models
model = Molecule_NN(input_dim, hidden_dims, output_dim)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

scalers = joblib.load(scalers_path)
scaler_X = scalers['scaler_X']
scaler_Y = scalers['scaler_Y']


####################### Evaluation #######################

def evaluate_model(model, X_new_tensor):
    model.eval()
    with torch.no_grad():
        predictions_scaled = model(X_new_tensor)
        predictions = scaler_Y.inverse_transform(predictions_scaled.cpu().numpy())
    return predictions

# Calculate polynomials for new data
r_H, r_mean, r_HOxHO = calculate_polynomes(pos)
r_H_scaled = r_H.reshape(-1,1)
r_mean_scaled = r_mean.reshape(-1,1)
r_HOxHO_scaled = r_HOxHO.reshape(-1,1)

# Prepare input features for the models
X = np.hstack((r_H_scaled, r_mean_scaled, r_HOxHO_scaled)).astype(np.float32)
X_new = scaler_X.transform(X)  # `new_data` should be your input data array
X_new_tensor = torch.from_numpy(X_new.astype(np.float32)).to(device)

# Evaluate the models
predictions = evaluate_model(model, X_new_tensor)

# Real experimental values for comparison
real_r = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])*1.88973
real_E = [-75.53219, -75.98354, -76.19821, -76.29462, -76.33092, -76.33652, -76.32674, -76.30969, -76.28973, -76.26922, -76.24938]

point = [[ 0,  0,  0 ],
 [ 1.4335500179,  0,  0.95295864922],
 [ 1.4335500179,  0, -0.95295864922]]

pol1, pol2, pol3 = calculate_polynomes(np.array([point]))
pol1 = pol1.reshape(-1,1)
pol2 = pol2.reshape(-1,1)
pol3 = pol3.reshape(-1,1)
pol = np.hstack((pol1,pol2,pol3)).astype(np.float32)
pol = scaler_X.transform(pol)
pol_tensor = torch.from_numpy(pol.astype(np.float32)).to(device)

pred2 = evaluate_model(model, pol_tensor)
####################### Plotting #######################

def plot_potential():
    plt.figure(figsize=(10, 5))
    plt.title("Potential Predictions")
    plt.plot(np.linspace(0.1*1.88973, 5*1.88973, 1000), predictions, color="red", label="Model Predictions")
    plt.scatter(real_r, real_E, color ="black", label="quantum chemical calculations")
    plt.scatter([1],pred2)
    plt.legend()
    plt.show()
plot_potential()