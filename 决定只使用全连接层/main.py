import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

epochs = 5
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)


# residual模块
# 残差模块
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#
#     def forward(self, x):
#         # 保存输入数据，采用恒等
#         # 映射
#
#         identity = x
#
#         #
#         out = self.conv1(x)
#
#         out = self.relu(out)
#
#         x = self.conv2(out)
#
#         out = torch.nn.functional.relu(x + identity)
#
#         # 还原结果
#         return out
#
#
# # ...
#
# # 构建包含ResidualBlock的CNN
# class ResNet_CNN(nn.Module):
#     def __init__(self, num_class=10):
#         super(ResNet_CNN, self).__init__()
#
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
#
#         self.relu = nn.ReLU()
#
#         self.res1 = ResidualBlock(16, 16)
#
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#
#         self.relu = nn.ReLU()
#
#         self.res2 = ResidualBlock(32, 32)
#
#         # 这个数据有点难
#
#         self.fc = nn.Linear(25088, num_class)
#
#     def forward(self, x):
#         x = self.conv1(x)
#
#         x = self.relu(x)
#
#         x = self.res1(x)
#
#         x = self.conv2(x)
#
#         x = self.relu(x)
#
#         x = self.res2(x)
#
#         # n个通道，每个通道1*1，输出1*1
#         # x = self.avg_pool(x)
#         # 将数据拉成一维
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#
#         return x

# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(64 * 3 * 3, 512)
#         self.fc2 = nn.Linear(512, 10)
#         self.dropout = nn.Dropout(0.5)
#
#     def forward(self, x):
#         x = nn.functional.relu(self.conv1(x))
#         x = nn.functional.max_pool2d(x, 2)
#         x = nn.functional.relu(self.conv2(x))
#         x = nn.functional.max_pool2d(x, 2)
#         x = nn.functional.relu(self.conv3(x))
#         x = nn.functional.max_pool2d(x, 2)
#         x = x.view(-1, 64 * 3 * 3)
#         x = self.dropout(x)
#         x = nn.functional.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
loss_f = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 训练模型
def train(epochs):
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_f(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()
                ))



def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('Accuracy on test set: %d %% [%d/%d]' % (100 * correct / total, correct, total))



if __name__ == '__main__':
    epochs = 5
    train(epochs)
    test()

# 保存模型 state_dict()是一个字典，保存了网络中所有的参数
# 转换并保存为torch.jit的模型

example_input = torch.rand(1, 1, 28, 28).to(device)
traced_model = torch.jit.trace(model, example_input)
torch.jit.save(traced_model, "traced_model.pt")
