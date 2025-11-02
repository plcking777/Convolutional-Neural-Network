import math


# TODO
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)


def relu(x):
    return max(0.0, x)

def relu_derivative(x):
    if x <= 0.0:
        return 0.0
    return 1.0


def map_activation(xs, activation):
        # TODO optimize
        for i in range(len(xs)):
            for j in range(len(xs[i])):
                xs[i][j] = activation(xs[i][j])
        return xs

def get_derivative_fn(fn):
    if fn == sigmoid:
        return sigmoid_derivative
    elif fn == relu:
        return relu_derivative
