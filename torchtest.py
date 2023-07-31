import torch
import numpy as np

# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.version.cuda)
t = torch.tensor([1, 2, 3])
print(t)
t = t.cuda()  # 用gpu计算
print(t)
dd = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
t = torch.tensor(dd)
print(t)
print(type(t))
print(t.shape)
t = t.reshape(1, 9)  # 需要接收一下返回值
print("------------------")
print(t)
print(t.shape)
print("------------------")
t = torch.Tensor()
print(type(t))
print(t.dtype)
print(t.device)
print(t.layout)
device = torch.device('cuda:0')
print("------------------")
print(device.type, device.index)
print(device)
print("------------------")
data = np.array([1, 2, 3])
print(type(data))
print(torch.Tensor(data))
print(torch.tensor(data))
print(torch.as_tensor(data))
print(torch.from_numpy(data))
print("------------------")
print(torch.eye(3))  # 单位阵
print(torch.zeros(2, 2))  # 零阵
print(torch.ones(2, 2))  # 全1
print(torch.rand(3, 3))  # 随机
print("------------------")
