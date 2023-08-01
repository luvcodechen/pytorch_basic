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
print("data数据：", data)
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
t1 = torch.Tensor(data)  # 构造类
t2 = torch.tensor(data)  # 工厂函数
t3 = torch.as_tensor(data)  # 工厂函数
t4 = torch.from_numpy(data)  # 工厂函数
print("torch_default_dtype: ", torch.get_default_dtype())
print(t1, t1.dtype, '\n', t2, t2.dtype, '\n', t3, t3.dtype, '\n', t4, t4.dtype)
print("------------------")
print("修改原data数据后：")
data[0] = data[1] = data[2] = 0
print(t1, '\n', t2, '\n', t3, '\n', t4)
print("------------------")
t = torch.tensor([
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3],
], dtype=torch.float32)
print(t.shape)
print(t.size())
print(len(t.shape))
print(torch.tensor(t.shape).prod())
print(t.numel())
print("------------------")

