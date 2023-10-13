import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

class FeatureVoxel4D(nn.Module):
    def __init__(self,factors,resolution,coord_range,channels):
        super(FeatureVoxel4D,self).__init__()
        self.plane_tx = torch.Tensor(factors[0],resolution[2] + 1,resolution[3] + 1,channels)
        self.plane_ty = torch.Tensor(factors[1],resolution[1] + 1,resolution[3] + 1,channels)
        self.plane_tz = torch.Tensor(factors[2],resolution[1] + 1,resolution[2] + 1,channels)
        self.plane_xy = torch.Tensor(factors[2],resolution[0] + 1,resolution[3] + 1,channels)
        self.plane_xz = torch.Tensor(factors[1],resolution[0] + 1,resolution[2] + 1,channels)
        self.plane_yz = torch.Tensor(factors[0],resolution[0] + 1,resolution[1] + 1,channels)
        self.plane_tx = nn.parameter.Parameter(self.plane_tx)
        self.plane_ty = nn.parameter.Parameter(self.plane_ty)
        self.plane_tz = nn.parameter.Parameter(self.plane_tz)
        self.plane_xy = nn.parameter.Parameter(self.plane_xy)
        self.plane_xz = nn.parameter.Parameter(self.plane_xz)
        self.plane_yz = nn.parameter.Parameter(self.plane_yz)

        self.grid_matrix = [
            (0,0,0,0),(0,0,0,1),(0,0,1,0),(0,0,1,1),
            (0,1,0,0),(0,1,0,1),(0,1,1,0),(0,1,1,1),
            (1,0,0,0),(1,0,0,1),(1,0,1,0),(1,0,1,1),
            (1,1,0,0),(1,1,0,1),(1,1,1,0),(1,1,1,1),
        ]
        self.grid_martix = torch.tensor(np.array(self.grid_matrix))
        self.grid_martix = self.grid_martix.unsqueeze(1)
        self.channels = channels
        self.resolution = resolution
        self.coord_range = coord_range
        self.factors = factors

    @torch.no_grad()
    def convert_to_voxel_coordinate(self,positions):
        coord_range = self.coord_range
        coord_range = coord_range.to(self.voxel.device)
        resolution = self.resolution
        resolution = resolution.to(self.voxel.device)
        positions = positions - coord_range[0].unsqueeze(0)
        positions = positions / ((coord_range[1] - coord_range[0]).unsqueeze(0))
        mask = (positions<1) & (positions>=0)
        mask = mask[:,0] & mask[:,1] & mask[:,2] & mask[:,3]
        positions = positions * resolution
        return positions, mask

    def forward(self,positions):
        with torch.no_grad():
            origin_batch_size = positions.shape[0]
            channels = self.channels
            positions,mask = self.convert_to_voxel_coordinate(positions)
            positions = positions[mask]
            batch_size = positions.shape[0]
            indices_on_grid = positions.to(torch.long)
            positions_in_grid = positions - indices_on_grid
            positions_t,positions_x,positions_y,positions_z = torch.unbind(positions_in_grid,1)
            positions_t = torch.stack([1-positions_t,positions_t]).permute(1,0).unsqueeze(1).unsqueeze(1).repeat(1,channels,1,1)
            positions_x = torch.stack([1-positions_x,positions_x]).permute(1,0).unsqueeze(1).unsqueeze(1).repeat(1,channels,1,1)
            positions_y = torch.stack([1-positions_y,positions_y]).permute(1,0).unsqueeze(1).unsqueeze(1).repeat(1,channels,1,1)
            positions_z = torch.stack([1-positions_z,positions_z]).permute(1,0).unsqueeze(1).unsqueeze(1).repeat(1,channels,1,1)

            indices_on_grid = indices_on_grid.unsqueeze(0).repeat(8,1,1)
            indices_on_grid = indices_on_grid + self.grid_martix.to(self.voxel.device).to(torch.long)

        features_0 = self.plane_tx[:,indices_on_grid[:,:,0],indices_on_grid[:,:,1]] * self.plane_yz[:,indices_on_grid[:,:,2],indices_on_grid[:,:,3]]
        features_1 = self.plane_ty[:,indices_on_grid[:,:,0],indices_on_grid[:,:,2]] * self.plane_xz[:,indices_on_grid[:,:,1],indices_on_grid[:,:,3]]
        features_2 = self.plane_tz[:,indices_on_grid[:,:,0],indices_on_grid[:,:,3]] * self.plane_xy[:,indices_on_grid[:,:,1],indices_on_grid[:,:,2]]
        features_0 = torch.sum(features_0,1)
        features_1 = torch.sum(features_1,1)
        features_2 = torch.sum(features_2,1)
        features = features_0 + features_1 + features_2
        features = features.permute(1,2,0)#n*c*16
        features = features.reshape(batch_size,channels,2,8)
        features = torch.matmul(positions_t,features)
        features = features.reshape(batch_size,channels,2,4)
        features = torch.matmul(positions_x,features)
        features = features.reshape(batch_size,channels,2,2)
        features = torch.matmul(positions_y,features)
        features = features.reshape(batch_size,channels,2,1)
        features = torch.matmul(positions_z,features)
        features = features.squeeze(-1).squeeze(-1)
        final_features = torch.zeros((origin_batch_size,channels)).to(self.voxel.device).to(self.voxel.dtype)
        final_features[mask] = features
        return final_features

class DynamicNeRF(nn.Module):
    def __init__(self,resolution,coord_range,channels):
        super(FeatureVoxel4D,self).__init__()
        self.voxel = torch.Tensor(resolution[0] + 1,resolution[1] + 1,resolution[2] + 1,resolution[3] + 1,channels)
        self.voxel = nn.parameter.Parameter(self.voxel)
        self.grid_matrix = [
            (0,0,0,0),(0,0,0,1),(0,0,1,0),(0,0,1,1),
            (0,1,0,0),(0,1,0,1),(0,1,1,0),(0,1,1,1),
            (1,0,0,0),(1,0,0,1),(1,0,1,0),(1,0,1,1),
            (1,1,0,0),(1,1,0,1),(1,1,1,0),(1,1,1,1),
        ]
        self.grid_martix = torch.tensor(np.array(self.grid_matrix))
        self.grid_martix = self.grid_martix.unsqueeze(1)
        self.channels = channels
        self.resolution = resolution
        self.coord_range = coord_range

    @torch.no_grad()
    def convert_to_voxel_coordinate(self,positions):
        coord_range = self.coord_range
        coord_range = coord_range.to(self.voxel.device)
        resolution = self.resolution
        resolution = resolution.to(self.voxel.device)
        positions = positions - coord_range[0].unsqueeze(0)
        positions = positions / ((coord_range[1] - coord_range[0]).unsqueeze(0))
        mask = (positions<1) & (positions>=0)
        mask = mask[:,0] & mask[:,1] & mask[:,2] & mask[:,3]
        positions = positions * resolution
        return positions, mask

    def forward(self,positions):
        with torch.no_grad():
            origin_batch_size = positions.shape[0]
            channels = self.channels
            positions,mask = self.convert_to_voxel_coordinate(positions)
            positions = positions[mask]
            batch_size = positions.shape[0]
            indices_on_grid = positions.to(torch.long)
            positions_in_grid = positions - indices_on_grid
            positions_t,positions_x,positions_y,positions_z = torch.unbind(positions_in_grid,1)
            positions_t = torch.stack([1-positions_t,positions_t]).permute(1,0).unsqueeze(1).unsqueeze(1).repeat(1,channels,1,1)
            positions_x = torch.stack([1-positions_x,positions_x]).permute(1,0).unsqueeze(1).unsqueeze(1).repeat(1,channels,1,1)
            positions_y = torch.stack([1-positions_y,positions_y]).permute(1,0).unsqueeze(1).unsqueeze(1).repeat(1,channels,1,1)
            positions_z = torch.stack([1-positions_z,positions_z]).permute(1,0).unsqueeze(1).unsqueeze(1).repeat(1,channels,1,1)

            indices_on_grid = indices_on_grid.unsqueeze(0).repeat(8,1,1)
            indices_on_grid = indices_on_grid + self.grid_martix.to(self.voxel.device).to(torch.long)

        features = self.voxel[indices_on_grid[:,:,0],indices_on_grid[:,:,1],indices_on_grid[:,:,2],indices_on_grid[:,:,3]].permute(1,2,0)#n*c*16
        features = features.reshape(batch_size,channels,2,8)
        features = torch.matmul(positions_t,features)
        features = features.reshape(batch_size,channels,2,4)
        features = torch.matmul(positions_x,features)
        features = features.reshape(batch_size,channels,2,2)
        features = torch.matmul(positions_y,features)
        features = features.reshape(batch_size,channels,2,1)
        features = torch.matmul(positions_z,features)
        features = features.squeeze(-1).squeeze(-1)
        final_features = torch.zeros((origin_batch_size,channels)).to(self.voxel.device).to(self.voxel.dtype)
        final_features[mask] = features
        return final_features