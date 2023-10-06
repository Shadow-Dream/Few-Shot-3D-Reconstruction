import torch
import torch.nn as nn
import torch.nn.functional as func

class FeatureProjector(nn.Module):
    def __init__(self):
        super(FeatureProjector,self).__init__()

    def forward(self, features, projections, depth_values):
        with torch.no_grad():
            batch_size, view_size = features.shape[0], features.shape[1]
            channels = features.shape[2]
            height, width = features.shape[3], features.shape[4]
            rotations = projections[:, :, :3, :3]
            translations = projections[:, :, :3, 3:4]

            if len(depth_values.shape) == 1:
                depth_values = depth_values.reshape(batch_size,1,1,1)
            else:
                depth_values = depth_values.reshape(batch_size,1,1,height*width)

            x = torch.arange(0, width, dtype=torch.float32).cuda()
            y = torch.arange(0, height, dtype=torch.float32).cuda()
            x, y = torch.meshgrid([x,y],indexing="xy")
            x = x.contiguous()
            y = y.contiguous()
            x = x.ravel()
            y = y.ravel()

            coord_grid = torch.stack([x, y, torch.ones_like(x)]) 
            coord_grid = coord_grid.unsqueeze(0).unsqueeze(0).repeat(batch_size, view_size, 1, 1)
            coord_grid = torch.matmul(rotations, coord_grid)
            coord_grid *= depth_values
            coord_grid += translations

            coord_grid = coord_grid[:, :, :2, :] / coord_grid[:, :, 2:3, :]
            x = coord_grid[:, :, 0, :] / (width - 1) * 2 - 1
            y = coord_grid[:, :, 1, :] / (height - 1) * 2 - 1
            coord_grid = torch.stack((x, y), dim = 3)

        features = features.reshape(batch_size*view_size,channels,height,width)
        coord_grid = coord_grid.reshape(batch_size*view_size, height, width, 2)
        features = func.grid_sample(features, coord_grid, mode='bilinear',padding_mode='zeros')
        features = features.reshape(batch_size, view_size, channels, height, width)
        return features