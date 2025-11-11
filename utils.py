import math
import numpy as np



# TODO
def sigmoid(xs):
    return 1.0 / (1.0 + np.exp(-xs))

def sigmoid_derivative(xs):
    return xs * (1.0 - xs)

def relu(xs):
    return np.maximum(0.0, xs)

def relu_derivative(xss):
    #(np.array([[1.0 if x > 0.0 else 0.0 for x in xs] for xs in xss]))
    return xss > 0.0

def softmax(xs):
    x = np.exp(xs)
    y = np.sum(np.exp(xs))
    return x / y


def no_activation(xs):
    return np.ones(xs.shape)


def get_derivative_fn(fn):
    if fn == sigmoid:
        return sigmoid_derivative
    elif fn == relu:
        return relu_derivative
    return no_activation



def to_one_hot(n, max):
    out = np.zeros(max)
    out[n] = 1.0
    return out