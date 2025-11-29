import math
import numpy as np



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

def stable_softmax(xs):
    xs = xs - np.max(xs, axis=-1, keepdims=True)
    exp_xs = np.exp(xs)
    return exp_xs / np.sum(exp_xs, axis=-1, keepdims=True)


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


def apply_clip(xs, clip):
    return np.clip(xs, -clip, clip)

def apply_norm_clip(xs, clip):
    mag = np.linalg.norm(xs)
    if mag == 0 or np.isnan(mag):
        return apply_clip(xs, clip)
    return (xs / mag) * clip


def select_random_mini_batch(data, labels, size):
    random_idxs = np.random.randint(0, len(data), size)
    batch_data = np.array([data[idx] for idx in random_idxs])
    batch_labels = np.array([labels[idx] for idx in random_idxs])

    return batch_data, batch_labels