import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

class FeatureVoxel(nn.Module):
    def __init__(self,resolution,channels,density_network,color_network):
        super(FeatureVoxel,self).__init__()
        self.voxel = torch.Tensor(resolution[0] + 1,resolution[1] + 1,resolution[2] + 1,channels)
        self.voxel = nn.parameter.Parameter(self.voxel)
        self.density_network = density_network
        self.color_network = color_network
        self.grid_matrix = [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]
        self.grid_martix = torch.tensor(np.array(self.grid_matrix))
        self.grid_martix = self.grid_martix.unsqueeze(1)
        self.channels = channels
        self.resolution = resolution

    def forward(self,points):
        with torch.no_grad():
            batch_size = points.shape[0]
            channels = self.channels 
            indices_on_grid = points.to(torch.long)
            points_in_grid = points - indices_on_grid
            points_x,points_y,points_z = torch.unbind(points_in_grid,1)
            points_x = torch.stack([1-points_x,points_x]).permute(1,0).unsqueeze(1).unsqueeze(1).repeat(1,channels,1,1)
            points_y = torch.stack([1-points_y,points_y]).permute(1,0).unsqueeze(1).unsqueeze(1).repeat(1,channels,1,1)
            points_z = torch.stack([1-points_z,points_z]).permute(1,0).unsqueeze(1).unsqueeze(1).repeat(1,channels,1,1)

            indices_on_grid = indices_on_grid.unsqueeze(0).repeat(8,1,1)
            indices_on_grid = indices_on_grid + self.grid_martix.to(self.voxel.device).to(torch.long)

        features = self.voxel[indices_on_grid[:,:,0],indices_on_grid[:,:,1],indices_on_grid[:,:,2]].permute(1,2,0)#n*c*8
        features = features.reshape(batch_size,channels,2,4)
        features = torch.matmul(points_x,features)
        features = features.reshape(batch_size,channels,2,2)
        features = torch.matmul(points_y,features)
        features = features.reshape(batch_size,channels,2,1)
        features = torch.matmul(points_z,features)
        features = features.squeeze(-1).squeeze(-1)
        return features

voxel = FeatureVoxel((1,1,1),1,None,None)
point = [(0,0,0.5),(0,0.5,0),(0.5,0,0),(0,0.5,0.5),(0,0.5,0.5),(0.5,0,0.5),(0.5,0.5,0.5)]
point = np.array(point)
point = torch.tensor(point,dtype = torch.float32)