import numpy as np
import joblib
from model_architechture import Molecule_NN
import torch
import glob
import matplotlib.pyplot as plt 
import re

# Device configuration
device = "cpu"  # Change to "cuda" or "mps" if using GPU

# Neural network architecture
input_dim = 3
hidden_dims = [20]
output_dim = 1  # Potential energy output

# Load the trained model
model = Molecule_NN(input_dim, hidden_dims, output_dim)
model.load_state_dict(torch.load("NN/models/model1.pth", map_location=torch.device('cpu')))
model.to(device)  # Send model to the device

# Load the scalers
scalers = joblib.load("NN/models/scalers.pkl")
scaler_X = scalers['scaler_X']
scaler_Y = scalers['scaler_Y']

def getV(R: np.array, eState: int) -> float:
    R=torch.from_numpy(R.squeeze()).to(device)
    # Scale the input data
    R_scaled = scaler_X.transform(R)

    # Convert to torch tensor
    R_tensor = torch.from_numpy(R_scaled).float().to(device)
    with torch.no_grad():
        E_scaled = model(R_tensor)
        E = scaler_Y.inverse_transform(E_scaled.cpu().numpy()).flatten()
        

    return E

#print(getV(np.array([[[1,2,3]], [[2,3,4]], [[3,4,5]], [[4,5,6]]]), 0))

def getGradV(R: np.array, eState: int) -> np.array:
    # Convert R to a PyTorch tensor with gradients enabled
    R_tensor = torch.tensor(R.squeeze(), dtype=torch.float32, device=device, requires_grad=True)
    
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
    grad_R = R_tensor_scaled.grad.cpu().numpy()
    return grad_R

##print(getGradV(np.array([[[1,2,3]], [[2,3,4]], [[3,4,5]], [[4,5,6]]]), 0))
print((3/2)/np.tanh(0.5/0.5))
print((3/2)/np.tanh(0.5/2))
print((3/2)/np.tanh(0.5/4))

Kin_pattern1 = 'output/*_KinEnergyTrace.csv'
Kin_list = glob.glob(Kin_pattern1)
Pot_pattern1 = 'output/*_PotEnergyTrace.csv'
Pot_list = glob.glob(Pot_pattern1)


Pot_pattern = re.compile(r'output/(\d+\.?\d*)\_PotEnergyTrace.csv')
Kin_E = []
Pot_E = []
T_ar = []
for Kin in Kin_list:
    Kin_E.append(np.mean(np.loadtxt(Kin)))
for Pot in Pot_list:
    T = float(Pot.split('\\')[1].split('_')[0])
    T_ar.append(T)
    Pot_E.append(np.mean(np.loadtxt(Pot)))

Tot_E = np.array(Kin_E)+np.array(Pot_E)


T = np.linspace(0.1, 5, 500)
exact = (3/2)/np.tanh(0.5/T) + 2 / (np.exp(2/T) + 1)

plt.figure(figsize=(10, 5))
plt.title("Potential Energy: Model Predictions vs. harmonic oscillator")
plt.scatter(T_ar, Tot_E, color="red", label="Model Predictions")
plt.plot(np.linspace(0.1, 5, 500), (3/2)/np.tanh(0.5/np.linspace(0.1, 5, 500)), color="black", label="harmonic oscillator")
plt.plot(np.linspace(0.1, 5, 500), exact)
plt.xlabel("Sample Index")
plt.ylabel("Potential Energy")
plt.legend()
plt.grid(True)
plt.show()



l = np.loadtxt("output/3.1e-05_PositionTrace.csv", delimiter=',')
print(l[:,0])
for i in range(3):
    plt.hist(l[:,i].flatten(),bins =50, alpha=.5)

plt.show()


E = Tot_E 
print(E)