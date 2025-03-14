
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义超参数
learning_rate = 0.001
batch_size = 64
epochs = 10

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载 KMNIST 数据集
train_dataset = datasets.KMNIST(root='./kmnist_data', train=True,
                                download=True, transform=transform)
test_dataset = datasets.KMNIST(root='./kmnist_data', train=False,
                               download=True, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义简单的神经网络模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 初始化模型、损失函数和优化器
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}')

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Accuracy on test set: {100 * correct / total}%')


# 尝试调整模型结构，增加隐藏层和神经元数量
class AdvancedNet(nn.Module):
    def __init__(self):
        super(AdvancedNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 初始化新模型、损失函数和优化器
advanced_model = AdvancedNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(advanced_model.parameters(), lr=learning_rate)

# 训练新模型
for epoch in range(epochs):
    advanced_model.train()
    running_loss = 0.0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = advanced_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}')

# 测试新模型
advanced_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = advanced_model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Accuracy on test set with advanced model: {100 * correct / total}%')


# 调试超参数：学习率和批次大小
learning_rates = [0.0001, 0.001, 0.01]
batch_sizes = [32, 64, 128]

for lr in learning_rates:
    for bs in batch_sizes:
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

        model = SimpleNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        print(f'Learning rate: {lr}, Batch size: {bs}, Accuracy: {100 * correct / total}%')
    
