import numpy as np
from layers import *
from utils import *
import pandas as pd


df = pd.read_csv("data/mnist-small.csv")


train_data = df[0:]
train_labels = df.label




print("shape:  ", train_data.shape)

model = Model([
    Convolution(1, (3, 3), 1, (28, 28)), # 28x28 -> convolution size 26x26 (x kernel_count)
    Flatten(), # 26x26 -> 676
    Dense(676, None),
    Dense(256, relu),
    Dense(10, softmax),
])


"""
model.set_input(train_data)
model.forward()
print("output= ", model.get_output())
print("cost = ", model.cost(train_labels))


for i in range(1000):
    model.set_input(train_data)
    model.forward()
    print("cost = ", np.sum(model.cost(train_labels)))
    model.backward(train_labels)
"""