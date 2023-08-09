import os

import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import pylab
import matplotlib

matplotlib.use("Qt5Agg")  # 或者使用 "Qt5Agg"

# class OHLC(Dataset):
#     def __int__(self, csv_file):
#         self.data = pd.read_csv(csv_file)
#
#     def __getitem__(self, index):
#         r = self.data.iloc[index]
#         label = torch.tensor(r.is_up_day, dtype=torch.long)
#         sample = self.normalize(torch.tensor([r.open, r.high, r.low, r.close]))
#         return sample, label
#
#     def __len__(self):
#         return len(self.data)

# root表示下载路径，train表示用于训练集（整个数据集有七万数据，六万作为训练集） download表示若路径下没有数据则下载  transform表示转换数据类型
train_set = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST', train=True, download=True,
                                              transform=transforms.Compose([transforms.ToTensor()]))

# train_loader = torch.utils.data.DataLoader(train_set, batch_size=10)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=50)
torch.set_printoptions(linewidth=120)
print(len(train_set))
print(train_set.targets)
print(train_set.targets.bincount())
sample = next(iter(train_set))
print("len(sample): ", len(sample))
print("type(sample): ", type(sample))
image, label = sample
print(image.shape)
# print(label.shape)
# plt.imshow(image.squeeze(), cmap="gray")
# print("label:", label)
# plt.show()
batch = next(iter(train_loader))
print(len(batch))
print(type(batch))

images, labels = batch
print(images.shape)
print(labels.shape)
grid = torchvision.utils.make_grid(images, nrow=10)
plt.figure(figsize=(15, 15), dpi=55)
plt.imshow(np.transpose(grid, (1, 2, 0)))
plt.show()
