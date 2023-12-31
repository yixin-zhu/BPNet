import math
import numpy as np
from Util.function import mean_squared_error, sigmoid, softmax, cross_entropy_error, numerical_gradient
from Net.MyNet import MyNet
import matplotlib.pyplot as plt

iter_num = 10000
train_size = 100
learning_rate = 0.001
test_size = 1000 
X = np.linspace(-np.pi, np.pi, train_size).reshape(1, train_size).T
t = np.sin(X)

mynet = MyNet(1, 10, 1)


mynet.train(X, t, iter_num, learning_rate, continue_train=True)

test_X = 2 * np.pi * (np.random.rand(test_size, 1) - 0.5)
test_y = mynet.predict(test_X)
test_t = np.sin(test_X)
div = abs(test_y - test_t)
# print("div=", div)
avg_div = np.sum(div) / test_size
print("avg_div=", avg_div)
plt.scatter(test_X, test_t,  linewidths=0.05, marker='o', color='blue')
plt.scatter(test_X, test_y,  linewidths=0.05, marker='o', color='red')
plt.title('test: +, train: o')
plt.show()
