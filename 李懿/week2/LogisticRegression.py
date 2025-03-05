from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

#数据集
X, y = load_iris(return_X_y=True)
print(X.shape)
print(y.shape)

#取数据集的前100行
X = X[:100]
y = y[:100]

#拆分训练和测试集
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, shuffle=True)

#初始化模型参数
theta = np.random.randn(1, 4)
bais = 0
lr = 1e-3
epoch = 2000

#向前计算
def forward(X, theta, bais):
    z = np.dot(theta, X.T) + bais
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat

#损失函数
def loss(y, y_hat):
    e = 1e-8
    return -y * np.log(y_hat + e) -(1 - y) * np.log(1 - y_hat + e)

#梯度计算
def gradient(x, y, y_hat):
    m = x.shape[-1]
    delta_theta = np.dot(y_hat - y, x) / m
    delta_bais = np.mean(y_hat - y)
    return delta_theta, delta_bais

for i in range(epoch):
    y_hat = forward(X, theta, bais)
    loss_val = np.mean(loss(y, y_hat))

    delta_theta , delta_bais = gradient(X, y, y_hat)
    theta = theta - lr * delta_theta
    bais = bais - lr * delta_bais

    if i % 100 == 0:
        acc = np.mean([np.round(y_hat) == y])
        print(f"loss:{loss_val}, acc:{acc}")



np.savez("model_params.npz", theta=theta, bais=bais, test_X=test_X, test_y=test_y)

