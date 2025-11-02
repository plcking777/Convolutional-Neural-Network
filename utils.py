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