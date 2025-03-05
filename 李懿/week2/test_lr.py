
import numpy as np


def forward(X, theta, bais):
    z = np.dot(theta, X.T) + bais
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat

loaded_parames = np.load("model_params.npz")
theta = loaded_parames["theta"]
bais = loaded_parames["bais"]
test_X = loaded_parames["test_X"]
test_y = loaded_parames["test_y"]


ind = np.random.randint(len(test_X))
x = test_X[ind]
y = test_y[ind]
pred = np.round(forward(x, theta, bais))

print(f"预测值:{pred}, 真实值:{y}")
