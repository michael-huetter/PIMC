import numpy as np
import os

import torch

from dataset import train_test_dataset
from pip_nn import PIP_NN, train_nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class molecule:
    def __init__(self, nuc, num_estates):
        self.nuc = nuc
        self.num_estates = num_estates
        self.pos = np.zeros((len(nuc), 3))
        self.path = None
        self.pos_trace = []
        self.energy_trace = []
        self.rij_trace = []
        self.rij_idx = []
    
    def load_data(self, path):
        self.path = path
        print("Loading data from", path, "...")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = f.readlines()
            if len(data) == 0:
                raise ValueError("File is empty")
        else:
            raise FileNotFoundError("File not found")
        n_at, n_estat = data[0].split()
        n_at = int(n_at)
        n_estat = int(n_estat)
        if n_at != len(self.nuc):
            raise ValueError("Number of atoms in the dataset does not match the number of atoms in the molecule")
        if n_estat != self.num_estates:
            raise ValueError("Number of states in the dataset does not match the number of states in the molecule")
        for i in range(1, len(data)):
            tokens = data[i].split()
            positions = [(tokens[i], float(tokens[i+1]), float(tokens[i+2]), float(tokens[i+3])) 
             for i in range(0, n_at * 4, 4)]
            energies = [(int(tokens[i]), float(tokens[i+1])) 
            for i in range(n_at * 4, n_at * 4 + n_estat * 2, 2)]
            self.pos_trace.append(positions)
            self.energy_trace.append(energies)
        print("Data loaded successfully")

    def compute_rij(self):
        print("Computing r_ij ...")
        for i in range(len(self.pos_trace)):
            pos = self.pos_trace[i]
            rij = []
            for i in range(len(pos)):
                for j in range(i+1, len(pos)):
                    if i < j:
                        if (i, j) not in self.rij_idx:
                            self.rij_idx.append((i, j))
                        rij.append(np.linalg.norm(np.array(pos[i][1:]) - np.array(pos[j][1:])))
            self.rij_trace.append(rij)
        print("r_ij computed successfully")


if __name__ == "__main__":

    my_mol = molecule(["H", "H"], num_estates=2)
    my_mol.load_data("test_dataset_h2.txt")  
    
    my_mol.compute_rij() 
    print(f"Found r_ij index: {my_mol.rij_idx}")

    # load data for training
    p_ij = np.exp(-np.array(my_mol.rij_trace))
    E = np.array(my_mol.energy_trace)

    # prepare data for training
    X = p_ij[:,0]
    Y0 = E[:, 0, 1]
    Y1 = E[:, 1, 1]
    Y = np.array([Y0, Y1]).T

    train_loader, test_loader = train_test_dataset(X, Y, batch_size=128)

    # set up model
    n_pip = X.reshape(-1, 1).shape[1]
    n_energy = Y.shape[1]
    model = PIP_NN(n_pip, n_energy)
    print("Used model:", model)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)
    print("Used device:", device)

    learning_rate = 1e-2
    epoches = 300
    
    loss_fn = MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    schedular = ReduceLROnPlateau(optimizer)
    
    train_nn(epoches, optimizer, device, loss_fn, model, train_loader, schedular)
