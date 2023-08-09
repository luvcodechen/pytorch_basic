import torch.nn as nn


class Lizard:
    def __init__(self, name):
        self.name = name

    def set_name(self, name):
        self.name = name


# class Network:
#     def __init__(self):
#         self.layer = None
#
#     def forward(self, t):
#         t = self.layer(t)
#         return t


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 44, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        # self.layer = None

    def forward(self, t):
        # t = self.layer(t)
        # implement the forward pass
        return t


#
# lizard = Lizard('abc')
# print(lizard.name)
# lizard.set_name('dsd')
# print(lizard.name)
network = Network()
print(network)
