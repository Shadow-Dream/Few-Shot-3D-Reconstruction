import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

class ImplicitField(nn.Module):
    def __init__(self,
                 feature_voxel,
                 density_network,
                 color_network,
                 image_width,
                 image_height,
                 focal, 
                 near, 
                 far,
                 num_samples = 128):
        self.feature_voxel = feature_voxel
        self.density_network = density_network
        self.color_network = color_network
        self.near = near
        self.far = far
        self.num_samples = num_samples
        self.image_width = image_width
        self.image_height = image_height
        self.focal = focal
    
    def get_points(self, positions, directions, distances):
        features = self.feature_voxel(positions)
        densities = self.density_network(positions,features)
        colors = self.color_network(directions,features)

        densities = densities.squeeze()

        delta = distances[..., 1:] - distances[..., :-1]
        delta = torch.cat([delta, torch.ones_like(delta[..., :1]).cuda().fill_(1e10)], -1)
        delta = delta * torch.norm(directions, dim=-1)

        alpha = 1 - torch.exp(-densities * delta)
        one_minus_alpha = torch.cat([torch.ones_like(delta[...,:1]).cuda(), 1 - alpha], dim=-1)
        weights = alpha * torch.cumprod(one_minus_alpha, dim=-1)[..., :-1]
        weights = weights.unsqueeze(-1)
        return colors, weights
    
    def forward(self, positions, forwards, ups):
        with torch.no_grad():
            batch_size = positions.shape[0]
            rays_positions,rays_directions = self.get_rays(positions, forwards, ups)
            normal = torch.sort(torch.rand((batch_size, self.num_samples,)).cuda())[0]
            distances = self.near + normal * (self.far - self.near)
            distances = distances.unsqueeze(1).unsqueeze(1)
            rays_positions = rays_positions[..., None, :] + rays_directions[..., None, :] * distances[..., None]
            rays_directions = rays_directions[...,None,:].broadcast_to(rays_positions.shape)

        colors, weights = self.get_points(rays_positions, rays_directions, distances)
        color_map = torch.sum(weights * colors, -2)
        depth_map = torch.sum(weights, -2)
        return color_map, depth_map
    
    def get_rays(self, positions, forwards, ups):
        batch_size = positions.shape[0]
        right = torch.cross(ups,forwards)
        transition = torch.concat([right.unsqueeze(1),ups.unsqueeze(1),forwards.unsqueeze(1)],dim=1)
        transition = transition.transpose(-1,-2)

        x = torch.arange(self.image_width, dtype=torch.float32).cuda()
        y = torch.arange(self.image_height, dtype=torch.float32).cuda()
        x, y = torch.meshgrid([x, y],indexing="xy")
        directions_local = torch.stack([
            (y-self.image_height/2+0.5)/self.focal,
            -(x-self.image_width/2+0.5)/self.focal,
            torch.ones_like(x)], -1)
        
        directions_local = directions_local.unsqueeze(0).repeat(batch_size,1,1,1)
        transition = transition.unsqueeze(1).unsqueeze(1)

        rays_directions = torch.sum(directions_local[..., None, :] * transition, -1)
        rays_positions = positions.unsqueeze(1).unsqueeze(1).broadcast_to(rays_directions.shape)
        return rays_positions, rays_directions

