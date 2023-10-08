import torch
import torch.nn as nn
import torch.nn.functional as func
from networks.Embedding import Embedding

class DensityNetwork(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.embedding = Embedding(channels=channels)
        self.density_input = nn.Linear(channels*3,channels*3)
        self.density_output = nn.Linear(channels*3,1)
    
    def forward(self,direction,feature):
        direction = self.embedding(direction)
        feature = feature + direction
        feature = func.relu(self.density_input(feature))
        density = func.relu(self.density_output(feature))
        return density