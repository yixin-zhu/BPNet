import numpy as np


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out

    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx
