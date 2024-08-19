import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from time import time
import joblib  # To save and load scalers
from sklearn.preprocessing import StandardScaler  # Scikit-learn scaler
from sklearn.model_selection import train_test_split

from model_architechture import Molecule_NN

#################### HYPERPARAMETERS #######################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Neural Network options
input_dim = 3
hidden_dims = [20]  # List of hidden layer dimensions
output_dim = 1  # Potential energy output

# Training options
learning_rate = 0.015
num_epochs = 6000
patience = 50
batch_size = 1000

# File paths
model_path = "NN/models/model.pth"
scalers_path = "NN/models/scalers.pkl"  # Path to save scalers

################# DATA PREPARATION #####################

def calculate_data(num_points: int) -> np.array:
    x = np.random.uniform(-5, 5, num_points)
    y = np.random.uniform(-5, 5, num_points)
    z = np.random.uniform(-5, 5, num_points)

    PE = 0.5 * (x**2 + y**2 + z**2)

    Positions = np.column_stack((x, y, z))
    potential_energies = PE

    return Positions, potential_energies

X, Energies = calculate_data(10000) 

# Split data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Energies, test_size=0.2, random_state=42)

# Initialize scalers
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

# Scale features and target
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)  # Use the same scaler for validation data
Y_train = scaler_Y.fit_transform(np.array(Y_train).reshape(-1, 1))
Y_val = scaler_Y.transform(np.array(Y_val).reshape(-1, 1))  # Scale validation targets


# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train.astype(np.float32)).to(device)
Y_train = torch.from_numpy(Y_train.astype(np.float32)).to(device)
X_val = torch.from_numpy(X_val.astype(np.float32)).to(device)
Y_val = torch.from_numpy(Y_val.astype(np.float32)).to(device)

# Save the scalers for future use
scalers = {'scaler_X': scaler_X, 'scaler_Y': scaler_Y}
joblib.dump(scalers, scalers_path)

################## MODEL TRAINING ######################

def train_model(model, X_train, Y_train, X_val, Y_val, num_epochs, learning_rate, patience, batch_size):
    t0 = time()
    train_loss_per_epoch = []
    val_loss_per_epoch = []
    best_val_loss =np.Infinity
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,)

    # Prepare data loader
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
        epoch_train_loss = running_train_loss / len(train_loader)
        train_loss_per_epoch.append(epoch_train_loss)

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, Y_val)
            val_loss_per_epoch.append(val_loss.item())
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_improvement = 0
                torch.save(model.state_dict(), model_path)
            else:
                epochs_since_improvement +=1
            if epochs_since_improvement > patience:
                t1 = time()
                print(f"Training stopped early. Total training time: {t1-t0}. Minimum validation loss: {best_val_loss}")
                return train_loss_per_epoch, val_loss_per_epoch
        
        # Print average train and validation loss per epoch
        if epoch % 10 == 9:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss}, Val Loss: {val_loss.item()}')

    t1 = time()
    print(f"Training complete! Total training time: {t1-t0}")
    return train_loss_per_epoch, val_loss_per_epoch

# Initialize the model
model = Molecule_NN(input_dim, hidden_dims, output_dim).to(device)

# Train the model
train_loss_per_epoch, val_loss_per_epoch = train_model(model, X_train, Y_train, X_val, Y_val, num_epochs, learning_rate, patience, batch_size)

# Save the trained model
if(len(train_loss_per_epoch) == num_epochs):
    torch.save(model.state_dict(), model_path)

# Plot the validation and training loss
plt.figure(figsize=(10, 5))
plt.title("Loss per Epoch")
plt.plot(train_loss_per_epoch, color='blue', label='Training Loss per epoch')
plt.plot(val_loss_per_epoch, color='orange', label='Validation Loss per epoch')
plt.legend()
plt.show()