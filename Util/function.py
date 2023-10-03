import numpy as np


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]

        # 计算f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # 计算f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        # 计算数值微分
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        # 恢复原始值
        x[idx] = tmp_val
        it.iternext()

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x
