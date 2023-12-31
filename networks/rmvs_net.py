import torch
import torch.nn as nn
import torch.nn.functional as func
from networks.feature_extractor import FeatureExtractor
from networks.feature_projector import FeatureProjector
from networks.gate_recurrent_unit import GateRecurrentUnit
import matplotlib.pyplot as plt
import numpy as np

class RMVSNet(nn.Module):
    def __init__(self,gru1,gru2,gru3,regressive_network,feature_extractor,feature_projector,feature_batch_size = 1):
        super(RMVSNet,self).__init__()
        self.feature_batch_size = feature_batch_size
        
        self.channels = feature_extractor.output_channels
        self.channels_0 = self.gru1.in_channels
        self.channels_1 = self.gru1.out_channels
        self.channels_2 = self.gru2.out_channels
        self.channels_3 = self.gru3.out_channels

        self.feature_extractor = feature_extractor
        self.feature_projector = feature_projector
        self.gru1 = gru1
        self.gru2 = gru2
        self.gru3 = gru3
        self.regressive_network = regressive_network

    def test_feature_projector(self,images,projections,depths):
        batch_size,view_size,channels,height,width = images.shape
        features = []

        images = images.reshape(batch_size*view_size,channels,height,width)
        features = images[:,:,::4,::4]
        height = height // 4
        width = width // 4
        features = features.reshape(batch_size,view_size,channels,height,width)

        features_ref = features[:,:1]
        projections_ref = projections[:,:1]
        features_src = features[:,1:]
        projections_src = projections[:,1:]

        projections_src = torch.matmul(projections_src, torch.inverse(projections_ref))
        features_pro = self.feature_projector(features_src,projections_src,depths)
        with torch.no_grad():
            feature_var = features_pro.var(2)
            zeros = torch.zeros_like(feature_var).to(feature_var.device)
            ones = torch.ones_like(feature_var).to(feature_var.device)
            mask_pro = torch.where(feature_var==0,zeros,ones)
            mask_pro = mask_pro.unsqueeze(2)
        test_image = features_pro[0,0].permute(1,2,0)
        test_image = np.array(test_image.detach().cpu())
        features_ref = features_ref[0,0].permute(1,2,0)
        features_ref = np.array(features_ref.detach().cpu())
        features_src = features_src[0,0].permute(1,2,0)
        features_src = np.array(features_src.detach().cpu())
        plt.subplot(1,3,1)
        plt.imshow(features_src)
        plt.subplot(1,3,2)
        plt.imshow(test_image)
        plt.subplot(1,3,3)
        plt.imshow(features_ref)
        plt.show()

    def forward(self,images,projections,depth_layers):
        batch_size,view_size,channels,height,width = images.shape
        num_depth_layers = depth_layers.shape[1]
        features = []

        images = images.reshape(batch_size*view_size,channels,height,width)
        if self.feature_batch_size <= batch_size * view_size:
            features = torch.unbind(self.feature_extractor(images),0)
        else:
            for i in range((batch_size*view_size)//self.feature_batch_size):
                start_index = self.feature_batch_size * i
                end_index = min(batch_size,self.feature_batch_size * (i + 1))
                features = features + torch.unbind(self.feature_extractor(images[start_index:end_index]),0)
        
        features = torch.stack(features)
        height = height // 4
        width = width // 4
        channels = self.feature_extractor.output_channels
        features = features.reshape(batch_size,view_size,channels,height,width)

        features_ref = features[:,:1]
        projections_ref = projections[:,:1]
        features_src = features[:,1:]
        projections_src = projections[:,1:]

        cost_1 = torch.zeros((batch_size, self.channels_1, height,width), dtype=torch.float).cuda()
        cost_2 = torch.zeros((batch_size, self.channels_2, height,width), dtype=torch.float).cuda()
        cost_3 = torch.zeros((batch_size, self.channels_3, height,width), dtype=torch.float).cuda()
        depth_costs = []
        projections_src = torch.matmul(projections_src, torch.inverse(projections_ref))

        for i in range(num_depth_layers):
            features_pro = self.feature_projector(features_src,projections_src,depth_layers[:,i])
            depth_features = torch.cat((features_ref,features_pro),1)

            cost = torch.mean(depth_features ** 2,1) - (torch.mean(depth_features,1) ** 2)
            cost_1 = self.gru1(- cost, cost_1)
            cost_2 = self.gru2(cost_1, cost_2)
            cost_3 = self.gru3(cost_2, cost_3)
            regressive_cost = self.regressive_network(cost_3)
            depth_costs.append(regressive_cost)

        volume = torch.cat(depth_costs, 1)
        volume = torch.softmax(volume, 1)
        volume = volume * depth_layers.reshape(batch_size,num_depth_layers,1,1)
        depth_map = torch.sum(volume,1)
        return depth_map

