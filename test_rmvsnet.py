from rmvsnet import RMVSNet
import torch
from datasets.dtu_yao import MVSDataset
from matplotlib import pyplot as plt
import numpy as np

model = RMVSNet()

model.to(torch.device('cuda'))

dataset = MVSDataset("./datasets/dtu/", "lists/dtu/train.txt", "train", 3, 192, 1.59)
sample = dataset[100]
images = sample["imgs"]
intrinsics = sample["intrinsics"]
extrinsics = sample["extrinsics"]
depth_values = sample["depth_values"]
depth_start = float(depth_values[0])
depth_interval = float(depth_values[1] - depth_values[0])
depth_num = depth_values.shape[0]
depths, probs = model(images, intrinsics, extrinsics, depth_start, depth_interval, depth_num)
test_image = images[0].permute(1,2,0).detach().cpu()
test_image = np.array(test_image)
depths = np.array(depths.detach().cpu())
probs = np.array(probs.detach().cpu())
plt.subplot(1,3,1)
plt.imshow(test_image)
plt.subplot(1,3,2)
plt.imshow(depths)
plt.subplot(1,3,3)
plt.imshow(probs)
plt.show()