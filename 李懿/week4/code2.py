import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset 
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#2. 结合归一化和正则化来优化网络模型结构，观察对比loss结果。

#设置参数
lr = 0.01
epoch = 10
batch_size = 128

#加载数据集
olivetti = fetch_olivetti_faces(data_home='olivetti_data', shuffle=True)

#拆分数据集
X_train, X_test, y_train, y_test = train_test_split(olivetti.data, olivetti.target, test_size=0.3, shuffle=True)

#加载归一化器
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#数据转成torch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


#将data, target整合一起
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

#数据加载器
train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_data, batch_size = batch_size, shuffle=True)

#定义模型
class NeuralNet(nn.Module):
    def __init__(self, use_dropout=True, use_batchnorm=True):
        super(NeuralNet, self).__init__()
        self.fn1 = nn.Linear(4096, 1024)
        self.bn1 = nn.BatchNorm1d(1024) if use_batchnorm else nn.Identity()
        self.fn2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512) if use_batchnorm else nn.Identity()
        self.fn3 = nn.Linear(512, 40)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.4) if use_dropout else nn.Identity()
    
    def forward(self, x):
        x = x.view(-1, 4096)
        x = self.act(self.fn1(x))
        x = self.bn1(x)
        x = self.act(self.fn2(x))
        x = self.bn2(x)
        x = self.drop(x)
        x = self.fn3(x)
        return x
    
#初始化模型
def init_model(use_dropout=True, use_batchnorm=True, weight_decay=0.0):
    model = NeuralNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, loss_fn, optimizer

#训练模型
def train_model(train_dl, model, loss_fn, optimizer):
    model.train()
    
    loss_history = []
    for i in range(epoch):
        running_loss = 0
        for data, target in train_dl:
            output = model(data)
            loss = loss_fn(output, target)
            running_loss += loss.item()

            #梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss = running_loss / len(train_data)
        loss_history.append(epoch_loss)
        print(f'Loss:{epoch_loss}')
    
    return loss_history


def test_model(test_dl, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_dl:
            output = model(data)
            _, pred = torch.max(output, 1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    print(f'acc:{correct / total}')

#无正则化，无归一化
model, loss_fn, optimizer = init_model(use_dropout=False, use_batchnorm=False, weight_decay=0.0)
loss_history_baseline = train_model(train_dl, model, loss_fn, optimizer)
test_model(test_dl, model)

#使用正则化和归一化
model, loss_fn ,optimizer = init_model(use_batchnorm=True, use_dropout=True, weight_decay=0.01)
loss_history_improved = train_model(train_dl, model, loss_fn, optimizer)
test_model(test_dl, model)

import matplotlib.pyplot as plt
plt.plot(loss_history_baseline, label='Baseline Model')
plt.plot(loss_history_improved, label='Improved Model')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.show()
