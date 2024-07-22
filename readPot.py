"""
Parse the output of a Gaussian PES scan.
"""

import numpy as np
#import matplotlib.pyplot as plt

def parse_pimc_data(filename, key):
    
    data = {k: [] for k in key}
    

    with open(filename, 'r') as file:
        


        for line in file:
            parts = line.strip().split()
            if not parts:
                continue

            # check if pars[0] is in key
            current_key = parts[0]
            if current_key in key:
                if current_key == "Eigenvalues":
                    data[current_key].append(parts[2:])
                else:   
                    data[current_key].append(parts[1:])
            else:
                continue
        
        return data

            
# Example usage

filename = 'waterPES.txt'
key = ['Eigenvalues', 'R1', 'R2', 'A1']
data = parse_pimc_data(filename, key)
V = []
Vo = []

eig = np.array([float(item) for sublist in data['Eigenvalues'] for item in sublist])

V.append(eig)

for k, v in data.items():
    if k != 'Eigenvalues':
        V.append(np.array([float(item) for sublist in v for item in sublist]))

V = np.array(V)

for i in range(len(V[0])):
    Vo.append([V[0][i], V[1][i], V[2][i], V[3][i]])

"""
# Plot the potential energy surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(V[1], V[2], V[0], c = V[0], cmap='viridis', marker="+", alpha=0.5)
# plot a minima marker where V[0] is minimum with big marker
minima = np.argmin(V[0])
ax.scatter(V[1][minima], V[2][minima], V[0][minima], c='r', marker='o', s=100)
ax.set_xlabel(r'$R_1$ in Å')
ax.set_ylabel(r'$R_2$ in Å')
ax.set_zlabel(r'$E$ in Hartree')
plt.show()
"""

# Normalize data and train a Gaussian process model 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import MinMaxScaler
import joblib

X_train = np.array(Vo)[:, 1:]
Y_train = np.array(Vo)[:, 0]

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(X_train_scaled, Y_train)

joblib.dump(scaler, 'scaler.pkl')
joblib.dump(gp, 'VgaussProcess.pkl')