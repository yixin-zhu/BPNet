import numpy as np
from Util.function import sigmoid, softmax, cross_entropy_error, numerical_gradient, mean_squared_error, relu
from Layer.Affine import Affine
from Layer.Sigmoid import Sigmoid
from Layer.SoftmaxWithLoss import SoftmaxWithLoss
from collections import OrderedDict
import pickle
import os


class MyNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(
            self.params['W1'], self.params['b1'])
        self.layers['Sigmoid1'] = Sigmoid()
        self.layers['Affine2'] = Affine(
            self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)

        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        return np.sum(y == t) / float(x.shape[0])

    def numerical_gradient(self, x, t):
        def loss_W(W): return self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设置
        grads = {}
        grads['W1'] = self.layers['Affine1'].dw
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dw
        grads['b2'] = self.layers['Affine2'].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        if not os.path.exists(file_name):
            self.save_params(file_name)

        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]

    def train(self, X, t, iter_num, learning_rate=0.01, continue_train=False):
        if (continue_train):
            self.load_params()
        for i in range(iter_num):
            loss = self.loss(X, t)
            grads = self.gradient(X, t)
            for key in ("W1", "b1", "W2", "b2"):
                self.params[key] -= learning_rate * grads[key]
            if (i % 100 == 0):
                print(i, loss)
        self.save_params()
