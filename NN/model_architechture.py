import torch
import torch.nn as nn

class Molecule_NN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super(Molecule_NN, self).__init__()
        
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        # Create hidden layers with Sigmoid activation
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.Sigmoid())
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x