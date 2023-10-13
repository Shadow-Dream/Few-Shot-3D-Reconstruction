import torch
import torch.nn as nn
import torch.nn.functional as func
from networks.embeddings import Embedding

class ColorNetwork(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.embedding = Embedding(channels=channels)
        self.color_input = nn.Linear(channels*3,channels*3)
        self.color_output = nn.Linear(channels*3,3)
    
    def forward(self,direction,feature):
        direction = self.embedding(direction)
        feature = feature + direction
        feature = func.relu(self.color_input(feature))
        color = func.sigmoid(self.color_output(feature))
        return color