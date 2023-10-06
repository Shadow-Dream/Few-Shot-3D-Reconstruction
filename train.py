from torch.utils.data import DataLoader
from networks.RMVSNet import RMVSNet
import torch
from datasets.dtu_yao import MVSDataset
import cv2 as cv
from torch.optim import Adam
import tqdm
import numpy
import matplotlib.pyplot as plt

from networks.FeatureProjector import FeatureProjector

BATCH_SIZE = 1
VIEW_SIZE = 2
SAVE_DELTA = 64
EPOCHES = 100

model = RMVSNet()
model.cuda().train()
dataset = MVSDataset("./datasets/dtu/", "lists/dtu/train.txt", "train", VIEW_SIZE, 128, 1.59)
optimizer = Adam(model.parameters(), lr=0.0001)

DATASET_SIZE = len(dataset.metas)

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
                sample = dataset[i * BATCH_SIZE + j]
                batch_images.append(sample["imgs"])
                batch_projections.append(sample["proj_matrices"])
                batch_depthes.append(sample["depth"])
                batch_depth_values.append(sample["depth_values"])
                batch_masks.append(sample["mask"])\
            
            batch_images = torch.stack(batch_images)
            batch_projections = torch.stack(batch_projections)
            batch_depthes = torch.stack(batch_depthes)
            batch_depth_values = torch.stack(batch_depth_values)
            batch_masks = torch.stack(batch_masks)

            batch_masks = torch.where(batch_masks==1,torch.ones_like(batch_masks),torch.zeros_like(batch_masks))
            max_delta_depth = batch_depth_values.max() - batch_depth_values.min()
        
        optimizer.zero_grad()
        depth_map = model(batch_images,batch_projections, batch_depth_values)
        
        loss = torch.mean((batch_masks * (depth_map - batch_depthes) / max_delta_depth)**2)
        loss.backward()
        optimizer.step()
        loss = float(loss.detach().cpu())
        losses.append(loss)
        average_loss = numpy.mean(numpy.array(losses))
        pbar.set_postfix({"epoch":epoch,"loss":loss,"average loss":average_loss,})
        if i % SAVE_DELTA == SAVE_DELTA - 1:
            torch.save(model.state_dict(),"model.ckpt")