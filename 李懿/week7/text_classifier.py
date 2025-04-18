import torch
import torch.nn as nn
import jieba
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import pandas as pd
import sentencepiece as spm


def load_data(file_name):
    data = pd.read_csv(file_name)
    preprocess = []
    
    for _, row in data.iterrows():
        star = int(row['Star'])  # 获取评分
        comment = str(row['Comment']).strip()  # 获取评论并去除首尾空格
        # 只保留1-2星和4-5星的评论
        if star in [1, 2]:
            preprocess.append((comment, star))
        elif star in [4, 5]:
            preprocess.append((comment, star))
    return preprocess

# 构建词汇表函数
def build_vocab(data, tokenizer, max_size=10000, min_freq=1):
    vocab_freq = {}  # 词频统计字典
    # 统计每个词出现的频率
    for comment, _ in data:
        words = tokenizer(comment)  # 分词
        for word in words:
            vocab_freq[word] = vocab_freq.get(word, 0) + 1
    
    # 按词频排序并截取前max_size个高频词
    vocab_list = sorted([(word, count) for word, count in vocab_freq.items() if count >= min_freq],
                       key=lambda x: x[1], reverse=True)[:max_size]
    
    # 创建词汇字典，为每个词分配唯一ID
    vocab_dict = {word: idx+2 for idx, (word, _) in enumerate(vocab_list)}
    vocab_dict['PAD'] = 0  # 填充符ID
    vocab_dict['UNK'] = 1   # 未知词ID
    return vocab_dict

# 获取分词器函数
def get_tokenizer(tokenizer_type='jieba'):
    if tokenizer_type == 'jieba':
        return jieba.lcut  # 使用jieba分词
    elif tokenizer_type == 'spm':
        sp = spm.SentencePieceProcessor()
        sp.load('comment.model')  # 加载sentencepiece模型
        return sp.encode_as_pieces  # 使用sentencepiece分词
    else:
        raise ValueError(f"未知的分词器类型: {tokenizer_type}")

# 模型训练和评估函数
def train_eval_model(model, train_loader, vocab, tokenizer, epochs=10):
    criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 定义Adam优化器

    # 训练循环
    for epoch in range(epochs):
        model.train()  # 设置为训练模式
        # 遍历每个batch
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()  # 梯度清零
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            # 每10个batch打印一次训练信息
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # 评估模式
        model.eval()
        with torch.no_grad():  # 禁用梯度计算
            # 测试两个示例评论
            comment1 = '这部电影太好看了，我非常喜欢！'
            comment2 = '这部电影太烂了，我非常不喜欢！'
            # 将评论转换为ID序列
            seq1 = torch.LongTensor([vocab.get(word, vocab['UNK']) for word in tokenizer(comment1)])
            seq2 = torch.LongTensor([vocab.get(word, vocab['UNK']) for word in tokenizer(comment2)])
            
            # 添加batch维度
            seq1 = seq1.unsqueeze(0)
            seq2 = seq2.unsqueeze(0)
            
            # 模型预测
            output1 = model(seq1)
            output2 = model(seq2)
            # 打印预测结果
            print(f"Comment1: {comment1}, Predicted Label: {torch.argmax(output1).item()}")
            print(f"Comment2: {comment2}, Predicted Label: {torch.argmax(output2).item()}")

# 自定义数据集类
class CommentDataset(Dataset):
    def __init__(self, data, vocab, tokenizer, pad_size=20):
        self.data = data  # 原始数据
        self.vocab = vocab  # 词汇表
        self.tokenizer = tokenizer  # 分词器
        self.pad_size = pad_size  # 填充长度
    
    def __len__(self):
        return len(self.data)  # 返回数据集大小
    
    # 将文本转换为ID序列
    def text_to_sequence(self, data):
        words = self.tokenizer(data)  # 分词
        seq = [self.vocab.get(word, self.vocab['UNK']) for word in words]  # 转换为ID序列
        # 填充或截断序列
        if len(seq) < self.pad_size:
            seq += [self.vocab['PAD']] * (self.pad_size - len(seq))
        else:
            seq = seq[:self.pad_size]
        return seq
    
    # 获取单个样本
    def __getitem__(self, index):
        comm, label = self.data[index]  
        seq = self.text_to_sequence(comm)  
        return torch.LongTensor(seq), torch.LongTensor([label])  

# 文本分类模型
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=100, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, 0)  
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, 1, batch_first=True)  
        self.fc = nn.Linear(hidden_dim, num_classes) 

    def forward(self, com_idx):
        out = self.embedding(com_idx)  
        out, _ = self.rnn(out)  
        out = self.fc(out[:, -1, :]) 
        return out

if __name__ == "__main__":
    # 加载数据
    data = load_data('./douban_MLS.csv')
    # 要测试的分词器类型
    tokenizer_type = ['jieba', 'spm']
    
    # 遍历每种分词器
    for tokenizer_name in tokenizer_type:
        print(f"\nUsing tokenizer: {tokenizer_name}")
        # 获取分词器
        tokenizer = get_tokenizer(tokenizer_name)
        # 构建词汇表
        vocab = build_vocab(data, tokenizer)
        # 创建数据集
        dataset = CommentDataset(data, vocab, tokenizer)
        # 创建数据加载器
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
        # 创建模型
        model = TextClassifier(len(vocab))
        # 训练和评估模型
        train_eval_model(model, dataloader, vocab, tokenizer)
