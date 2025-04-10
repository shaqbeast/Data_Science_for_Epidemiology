import torch
import torch.nn as nn
import torch.optim as optim

class GRUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUNetwork, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)  # GRU layer
        self.fc = nn.Linear(hidden_size, output_size)  # Dense layer
    
    def forward(self, x):
        out, _ = self.gru(x)  # Pass through GRU
        out = out[:, -1, :]  # Use the last output of the GRU
        out = self.fc(out)  # Dense layer to predict the output
        return out
