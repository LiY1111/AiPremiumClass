import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

# 配置参数
class Config:
    data_path = 'Summary of Weather.csv'  # 数据集路径
    seq_length = 30                      # 输入序列长度(天)
    pred_length = 5                      # 预测长度(天)
    batch_size = 64
    hidden_size = 128
    num_layers = 2
    dropout = 0.3
    lr = 0.001
    epochs = 150
    test_size = 0.2
    target_col = 'MaxTemp'               # 预测目标列
    features = ['MaxTemp', 'MinTemp', 'Precipitation', 'WindSpeed', 'Humidity']  # 使用特征

# 数据加载和预处理
class WeatherDataset(Dataset):
    def __init__(self, data, mode='single'):
        self.data = data
        self.mode = mode  # 'single'或'multi'
        
    def __len__(self):
        return len(self.data) - Config.seq_length - Config.pred_length + 1
        
    def __getitem__(self, idx):
        seq = self.data[idx:idx+Config.seq_length]
        
        if self.mode == 'single':
            # 单日预测：预测第seq_length+1天的温度
            target = self.data[idx+Config.seq_length, 0]  # 第一列是MaxTemp
            return torch.FloatTensor(seq), torch.FloatTensor([target])
        else:
            # 多日预测：预测接下来5天的温度
            target = self.data[idx+Config.seq_length:idx+Config.seq_length+Config.pred_length, 0]
            return torch.FloatTensor(seq), torch.FloatTensor(target)

def load_data():
    # 加载数据
    df = pd.read_csv(Config.data_path)
    
    # 转换日期并排序
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 选择特征列
    df = df[['Date'] + Config.features].dropna()
    
    # 标准化
    scaler = StandardScaler()
    df[Config.features] = scaler.fit_transform(df[Config.features])
    
    return df, scaler

# RNN模型
class WeatherRNN(nn.Module):
    def __init__(self, input_size, output_size=1):
        super().__init__()
        self.rnn = nn.GRU(  # 使用GRU单元
            input_size=input_size,
            hidden_size=Config.hidden_size,
            num_layers=Config.num_layers,
            batch_first=True,
            dropout=Config.dropout if Config.num_layers > 1 else 0
        )
        self.regressor = nn.Sequential(
            nn.Linear(Config.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(Config.dropout),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        out, _ = self.rnn(x)  # [batch, seq_len, hidden_size]
        out = self.regressor(out[:, -1, :])  # 取最后时间步
        return out

# 训练和评估
def train_and_evaluate(model, train_loader, test_loader, writer, mode='single'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    
    best_test_loss = float('inf')
    
    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 测试集评估
        test_loss = evaluate(model, test_loader, criterion, device)
        
        # 记录到TensorBoard
        writer.add_scalars(f'Loss/{mode}', 
                         {'Train': train_loss, 'Test': test_loss}, epoch)
        
        # 保存最佳模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), f'best_{mode}_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'{mode} Epoch [{epoch+1}/{Config.epochs}], '
                  f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    return model

def evaluate(model, loader, criterion, device):
    model.eval()
    loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, targets).item()
    
    return loss / len(loader)

# 主函数
def main():
    # 创建TensorBoard记录目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/weather_prediction_{timestamp}')
    
    # 加载数据
    df, scaler = load_data()
    data = df[Config.features].values
    
    # 单日预测
    print("\nTraining single-day prediction model...")
    single_dataset = WeatherDataset(data, mode='single')
    train_idx, test_idx = train_test_split(
        range(len(single_dataset)), test_size=Config.test_size, shuffle=False)
    
    train_loader = DataLoader(
        torch.utils.data.Subset(single_dataset, train_idx),
        batch_size=Config.batch_size, shuffle=True)
    
    test_loader = DataLoader(
        torch.utils.data.Subset(single_dataset, test_idx),
        batch_size=Config.batch_size)
    
    single_model = WeatherRNN(len(Config.features), output_size=1)
    train_and_evaluate(single_model, train_loader, test_loader, writer, 'single')
    
    # 多日预测
    print("\nTraining multi-day prediction model...")
    multi_dataset = WeatherDataset(data, mode='multi')
    train_idx, test_idx = train_test_split(
        range(len(multi_dataset)), test_size=Config.test_size, shuffle=False)
    
    train_loader = DataLoader(
        torch.utils.data.Subset(multi_dataset, train_idx),
        batch_size=Config.batch_size, shuffle=True)
    
    test_loader = DataLoader(
        torch.utils.data.Subset(multi_dataset, test_idx),
        batch_size=Config.batch_size)
    
    multi_model = WeatherRNN(len(Config.features), output_size=Config.pred_length)
    train_and_evaluate(multi_model, train_loader, test_loader, writer, 'multi')
    
    writer.close()
    print("训练完成！使用以下命令查看TensorBoard：")
    print(f"tensorboard --logdir=runs/weather_prediction_{timestamp}")

if __name__ == '__main__':
    main()