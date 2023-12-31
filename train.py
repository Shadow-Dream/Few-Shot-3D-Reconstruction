from torch.utils.data import DataLoader
from networks.rmvs_net import RMVSNet
import torch
from datasets.dtu_yao import MVSDataset
import cv2 as cv
from torch.optim import Adam
import tqdm
import numpy
import matplotlib.pyplot as plt
import os
import torch.nn.functional as func

from networks.feature_projector import FeatureProjector
from networks.feature_extractor import FeatureExtractor
from networks.gate_recurrent_unit import GateRecurrentUnit
from networks.regressive_network import RegressiveNetwork

BATCH_SIZE = 1
VIEW_SIZE = 5
SAVE_DELTA = 64
EPOCHES = 100

gru1 = GateRecurrentUnit(32,16)
gru2 = GateRecurrentUnit(16,4)
gru3 = GateRecurrentUnit(4,2)
regressive_network = RegressiveNetwork(2)
feature_extractor = FeatureExtractor(output_channels=32)
feature_projector = FeatureProjector(scale=2)

model = RMVSNet(gru1,gru2,gru3,regressive_network,feature_extractor,feature_projector)
model.cuda().train()

dataset = MVSDataset("./datasets/dtu/", "lists/dtu/train.txt", "train", VIEW_SIZE, 128, 1.59)
optimizer = Adam(model.parameters(), lr=0.001)

DATASET_SIZE = len(dataset.metas)

if os.path.exists("model.ckpt"):
    model.load_state_dict(torch.load("model.ckpt"))

def mvsnet_loss(depth_pred, depth_real, mask):
    mask = mask == 1
    return func.smooth_l1_loss(depth_pred[mask], depth_real[mask], size_average=True)

for epoch in range(EPOCHES):
    losses = []
    pbar = tqdm.tqdm(range(DATASET_SIZE//BATCH_SIZE))
    for i in pbar:
        with torch.no_grad():
            batch_images = []
            batch_projections = []
            batch_depthes = []
            batch_depth_values = []
            batch_masks = []
            for j in range(BATCH_SIZE):
                sample = dataset[0 * BATCH_SIZE + j]
                batch_images.append(sample["imgs"])
                batch_projections.append(sample["proj_matrices"])
                batch_depthes.append(sample["depth"])
                batch_depth_values.append(sample["depth_values"])
                batch_masks.append(sample["mask"])
            
            batch_images = torch.stack(batch_images)
            batch_projections = torch.stack(batch_projections)
            batch_depthes = torch.stack(batch_depthes)
            batch_depth_values = torch.stack(batch_depth_values)
            batch_masks = torch.stack(batch_masks)
        
        # model.test_feature_projector(batch_images,batch_projections,batch_depthes)
        optimizer.zero_grad()
        depth_map = model(batch_images,batch_projections, batch_depth_values)
        
        loss = mvsnet_loss(depth_map,batch_depthes,batch_masks)
        loss.backward()
        optimizer.step()
        loss = float(loss.detach().cpu())
        losses.append(loss)
        average_loss = numpy.mean(numpy.array(losses))
        pbar.set_postfix({"epoch":epoch,"loss":loss,"average loss":average_loss,})
        if i % SAVE_DELTA == SAVE_DELTA - 1:
            torch.save(model.state_dict(),"model.ckpt")
        plt.subplot(1,2,1)
        plt.imshow(batch_masks[0].detach().cpu()*depth_map[0].detach().cpu(),vmin=batch_depth_values.min(), vmax=batch_depth_values.max())
        plt.subplot(1,2,2)
        plt.imshow(batch_depthes[0].detach().cpu(),vmin=batch_depth_values.min(), vmax=batch_depth_values.max())
        plt.savefig("results/{}.png".format(i))