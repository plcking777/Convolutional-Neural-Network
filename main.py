import numpy as np
from layers import *
from utils import *




train_data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

train_labels = np.array([
    [0],
    [1],
    [1],
    [0],
])


print("shape:  ", train_data.shape)

model = Model([
    Dense(2, None),
    Dense(5, relu),
    Dense(1, sigmoid),
])



model.set_input(train_data)
model.forward()
print("output= ", model.get_output())
print("cost = ", model.cost(train_labels))


for i in range(1000):
    model.set_input(train_data)
    model.forward()
    print("cost = ", model.cost(train_labels))
    model.backward(train_labels)