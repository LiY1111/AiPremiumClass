import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces

#超参数
lr = 1e-3
epoch=100

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=64,
            hidden_size=56,
            num_layers=2,
            batch_first=True
        )
        self.fn = nn.Linear(56, 40)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fn(out[:, -1, :])
        return out
    
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=64,
            hidden_size=56,
            num_layers=2,
            batch_first=True
        )
        self.fn = nn.Linear(56, 40)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fn(out[:, -1, :])
        return out
    
class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=64,
            hidden_size=56,
            num_layers=2,
            batch_first=True
        )
        self.fn = nn.Linear(56, 40)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fn(out[:, -1, :])
        return out
    
class BiRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=64,
            hidden_size=56,
            num_layers=2,
            batch_first=True, 
            bidirectional=True
        )
        self.fn = nn.Linear(56 * 2, 40)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fn(out[:, -1, :])
        return out



#加载数据
def load_data():
    olivetti = fetch_olivetti_faces(data_home='./olivetti_data', shuffle=True)

    images, target = olivetti.images, olivetti.target

    train_X, test_X, train_y, test_y= train_test_split(images, target, test_size=0.3, shuffle=True)

    train_X, train_y = torch.tensor(train_X, dtype=torch.float), torch.tensor(train_y, dtype=torch.long)
    test_X, test_y = torch.tensor(test_X, dtype=torch.float), torch.tensor(test_y, dtype=torch.long)

    train_set, test_set = TensorDataset(train_X, train_y), TensorDataset(test_X, test_y)
    x, y = train_set.tensors
    print(x)

    train_dl, test_dl = DataLoader(train_set, batch_size=128), DataLoader(test_set, batch_size=128)
    return train_dl, test_dl


#模型初始化
def init_model(lr, model_name):
    models = {
        "RNN": RNN(),
        "LSTM": LSTM(),
        "GRU": GRU(),
        "BiRNN": BiRNN()
    }
    model = models[model_name]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    return model, optimizer, loss_fn

#模型训练
def train_model(epoch, dl, model, optimizer, loss_fn):
    model.train()
    for i in range(epoch):
        loss_val = 0
        for images, target in dl:
            out = model(images)
            loss = loss_fn(out, target)
            loss_val += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss = loss_val / len(dl)
        print(f"loss: {epoch_loss}, epoch: {i +1}")

def test_model(dl, model):
    model.eval()
    correct = 0
    for images, target in dl:
        out = model(images)
        pred = torch.argmax(out, dim=1)
        correct += pred.eq(target).sum().item()
    print(f"acc: {correct / len(dl.dataset)}")

if __name__ == "__main__":
    model_name = ["RNN", "LSTM","GRU"]
    for name in model_name:
        model, optimizer, loss_fn = init_model(lr, name)
        print(f"------------------------{name} Model Train Start--------------------")
        train_dl, test_dl = load_data()
        train_model(epoch, train_dl, model, optimizer, loss_fn)
        test_model(test_dl, model)
