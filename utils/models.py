# import libs
import torch.nn as nn

# gaia model (pytorch version)
class GaiaModel(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(GaiaModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 472),
            nn.ReLU(),
            nn.Linear(472, 235),
            nn.ReLU(),
            nn.Linear(235, 118),
            nn.ReLU(),
            nn.Linear(118, 59),
            nn.ReLU(),
            nn.Linear(59, output_size),
        )
        
    def forward(self, x):
        return self.layers(x)

# spira model (pytorch version)
class SpiraModel(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(SpiraModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 472),
            nn.ReLU(),
            nn.Linear(472, 235),
            nn.ReLU(),
            nn.Linear(235, 118),
            nn.ReLU(),
            nn.Linear(118, 59),
            nn.ReLU(),
            nn.Linear(59, output_size),
        )
        
    def forward(self, x):
        return self.layers(x)

