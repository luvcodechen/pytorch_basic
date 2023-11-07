import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils import data

torch.set_printoptions(linewidth=120)  # Display options for output
torch.set_grad_enabled(True)
train_set = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        # self.layer = None

    def forward(self, t):
        # t = self.layer(t)
        # implement the forward pass

        # (1) input layer
        t = t

        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) hidden liner layer
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden liner layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output layer
        t = self.out(t)
        # t = F.softmax(t, dim=1)

        return t


# torch.set_grad_enabled(False)

network = Network()

# sample = next(iter(train_set))
#
# image, label = sample
#
# pred = network(image.unsqueeze(0))  # image shape needs to be (batch_size x in_channels x H x W)
# print(pred.shape)
# print(pred)
# print(label)
# print(pred.argmax(dim=1))
# print(F.softmax(pred, dim=1))
# print(F.softmax(pred, dim=1).sum())

data_loader = torch.utils.data.DataLoader(train_set, batch_size=100)

batch = next(iter(data_loader))

images, labels = batch

# Calculating the loss
preds = network(images)
loss = F.cross_entropy(preds, labels)  # Calculating the loss
loss.item()

# calculating the gradients
print(network.conv1.weight.grad)
loss.backward()  # Calculating the gradients
print(network.conv1.weight.grad.shape)

# Updating the weights
optimizer = optim.Adam(network.parameters(), lr=0.01)
print("loss:", loss.item())
print(get_num_correct(preds, labels))
optimizer.step()  # Updating the weights
preds = network(images)
loss = F.cross_entropy(preds, labels)
print("loss:", loss.item())

print(get_num_correct(preds, labels))

# print(preds)
# print(preds.argmax(dim=1))
# print(labels)

# print(network)
# print(network.conv1.weight)
# print(network.conv1.weight.shape)
# print(network.conv2.weight.shape)
# print(network.fc1.weight.shape)
# print(network.fc2.weight.shape)
# print(network.out.weight.shape)
# print(network.conv2.weight[0].shape)
# in_features = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
# weight_matrix = torch.tensor([
#     [1, 2, 3, 4],
#     [2, 3, 4, 5],
#     [3, 4, 5, 6],
# ], dtype=torch.float32)
# print(weight_matrix.matmul(in_features))
# # print(weight_matrix.mv(in_features))
# fc = nn.Linear(in_features=4, out_features=3, bias=False)
# fc.weight = nn.Parameter(weight_matrix)
# print(fc(in_features))

for epoch in range(5):
    # 所有批次循环训练
    total_loss = 0
    total_correct = 0
    for batch in data_loader:
        images, labels = batch

        preds = network(images)  # pass batch
        loss = F.cross_entropy(preds, labels)  # calculate loss

        optimizer.zero_grad()
        loss.backward()  # calculate gradients
        optimizer.step()  # update weights

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)
    print("epoch:", epoch, "total_correct: ", total_correct, " loss :", total_loss)
    print(total_correct / len(train_set))


def get__all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds),
            dim=0
        )
    return all_preds


prediction_loder = torch.utils.data.DataLoader(train_set, batch_size=10000)
train_preds = get__all_preds(network, prediction_loder)
print(train_preds.shape)
print(train_preds.requires_grad)
print(train_preds.grad)
print(train_preds.grad_fn)
with torch.no_grad():
    prediction_loder = torch.utils.data.DataLoader(train_set, batch_size=10000)
    train_preds = get__all_preds(network, prediction_loder)
print(train_preds.requires_grad)
print(train_preds.grad)
print(train_preds.grad_fn)
preds_correct = get_num_correct(train_preds, train_set.targets)

print("total correct:", preds_correct)
print('accuracy:', preds_correct / len(train_set))
print(train_set.targets)
print(train_preds.argmax(dim=1))
stacked = torch.stack(
    (
        train_set.targets,
        train_preds.argmax(dim=1)
    )
    , dim=1
)
print(stacked.shape)
print(stacked)
print(stacked[0].tolist())
cmt = torch.zeros(10, 10, dtype=torch.int32)
print(cmt)
for p in stacked:
    tl, pl = p.tolist()
    cmt[tl, pl] = cmt[tl, pl] + 1
print(cmt)

# plotting a confusion matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from resources.plotcm import plot_confusion_matrix
cm = confusion_matrix(train_set.targets, train_preds.argmax(dim=1))
print(type(cm))
print(cm)
names = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm, names)
