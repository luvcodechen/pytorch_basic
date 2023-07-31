import torch

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
