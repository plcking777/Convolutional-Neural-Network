import numpy as np
from layers import *
from utils import *
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/mnist.csv")


train_data = np.array(df).T[1:].T  # pixels (removed the labels)
train_labels = np.array([to_one_hot(label, 10) for label in df.label])



train_data = np.array([[np.reshape(entry, (28, 28))] for entry in train_data]) / 255.0  # restore structure of the image


#train_data = np.array([train_data[0], train_data[1]])
#train_labels = np.array([train_labels[0], train_labels[1]])


print("shape:  ", train_data.shape)

model = Model([
    Convolution(1, (3, 3), 1, (28, 28)), # {data_count}x{prev_kernel_count}x28x28 -> convolution size {data_count}x{kernel_count * prev_kernel_count}x26x26
    Flatten(), # {data_count}x{kernel_count}x26x26 -> {data_count}x676
    Dense(676, None),
    Dense(128, relu),
    Dense(10, stable_softmax),
])


graph_x = []
graph_y = []


for i in range(500):
    batch_data, batch_label = select_random_mini_batch(train_data, train_labels, 64)

    model.set_input(batch_data)
    model.forward()
    cost = np.sum(model.cost(batch_label))
    graph_x.append(i)
    graph_y.append(cost)
    print("cost = ", cost)
    model.backward(batch_label)


print("OUT: ", model.get_output())
print("LABEL: ", train_labels)
plt.plot(graph_x, graph_y)
plt.show()
