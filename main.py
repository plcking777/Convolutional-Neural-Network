import numpy as np
from layers import *




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
    Dense(2),
    Dense(5),
    Dense(1),
])



model.set_input(train_data)
model.forward()
print("output= ", model.get_output())
