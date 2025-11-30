import numpy as np
from layers import *
from utils import *
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/mnist-small.csv")


train_data = np.array(df).T[1:].T  # pixels (removed the labels)
train_labels = np.array([to_one_hot(label, 10) for label in df.label])



train_data = np.array([[np.reshape(entry, (28, 28))] for entry in train_data]) / 255.0  # restore structure of the image



print("shape:  ", train_data.shape)

model = Model([
    Convolution(16, (3, 3), 2, (28, 28)),
    Flatten(),
    Dense(169 * 16, None),
    Dense(512 * 4, relu),
    Dense(10, stable_softmax),
])


graph_x = []
graph_y = []


for i in range(10000):
    batch_data, batch_label = select_random_mini_batch(train_data, train_labels, 10)

    model.set_input(batch_data)
    model.forward()
    cost = np.sum(model.cost(batch_label))
    if i % 500 == 0:
        graph_x.append(i)
        graph_y.append(cost)
        print("cost = ", cost)
    model.backward(batch_label)



model.print_state()


plt.plot(graph_x, graph_y)
plt.show()
