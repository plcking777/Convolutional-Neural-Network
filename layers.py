import numpy as np
from utils import *

class Model():

    def __init__(self, layers):
        self.layers = layers
        self.learning_rate = 0.1

        prev_output_size = None
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.init_weights_and_biases(prev_output_size)
                prev_output_size = layer.get_output_shape()
            elif isinstance(layer, Convolution):
                layer.init_weights_and_biases()



    
    def forward(self):
        
        prev_output = self.model_input

        for idx in range(len(self.layers)):
            layer = self.layers[idx]

            current_output = layer.forward(prev_output)

            prev_output = current_output
    
        self.model_output = prev_output


    def cost(self, target):
        diff = self.get_output() - target
        return diff ** 2

    def cost_derivative(self, target):
        return (self.get_output() - target) * 2

    def backward(self, target):

        part_deriv = self.cost_derivative(target).T

        for idx in range(len(self.layers)):
            layer = self.layers[len(self.layers) - 1 - idx]
            if layer._is_input:
                continue
            
            dact = get_derivative_fn(layer.activation)(layer.get_activations()).T

            # update weights & biases

            new_part_deriv = layer.weights.dot(part_deriv)

            # TODO
            next_layer = self.layers[len(self.layers) - 2 - idx]
            layer.weights = layer.weights - (self.learning_rate * (part_deriv * dact).dot(next_layer.get_activations())).T
            layer.biases = layer.biases - (self.learning_rate * part_deriv).T
            
            part_deriv = new_part_deriv


    def set_input(self, input):
        self.model_input = input

    def get_output(self):
        return self.model_output


class Dense():
    
    def __init__(self, size, activation):
        self.weights = None
        self.biases = None
        self._is_input = False
        self.size = size
        self.activation = activation

        # state that stores the last output from the forward step of this layer
        self.activations = []


    def forward(self, input):
        if self._is_input:
            self.activations = input
            return input
        
        self.activations = self.activation(input.dot(self.weights) + self.biases)
        return self.activations


    def init_weights_and_biases(self, input_size):
        if input_size == None:
            self._is_input = True
        else:
            self.weights = np.random.rand(input_size, self.size) - 0.5
            self.biases = np.random.rand(1, self.size) - 0.5

    def get_activations(self):
        return self.activations
    
    def get_output_shape(self):
        return self.size
    

class Flatten():
    def __init__(self):
        pass
    
    def forward(self, input):
        return input.flatten()

    def init_weights_and_biases(self, _):
        pass

    def get_output_shape(self):
        pass #TODO

class Convolution():
    
    def __init__(self, kernel_count, kernel_shape, stride, input_shape):
        self.kernel_count = kernel_count
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.input_shape = input_shape

        self.kernel_weights = []
    
    def init_weights_and_biases(self):
        self.kernel_weights = np.random.rand(self.kernel_count, self.kernel_shape[0], self.kernel_shape[1])

    def get_output_shape(self):
        pass #TODO

    def get_convolution_shape(self):
        def relu(x):
            return max(0.0, x)
        
        return (relu(self.input_shape[0] - self.kernel_shape[0]) // self.stride + 1, relu(self.input_shape[1] - self.kernel_shape[1]) // self.stride + 1)
        

    def forward(self, input):

        conv_shape = self.get_convolution_shape()
        out = []
        for data in input:
            output_convolutions = []
            for input_convolution in data:
                #shape = 28x28
                row = 0
                col = 0

                conv = np.zeros(conv_shape)
                for conv_row in range(conv_shape[0]):
                    for conv_col in range(conv_shape[1]):
                        conv_value = np.sum(input_convolution[row:row + self.kernel_shape[0], col:col + self.kernel_shape[1]].dot(self.kernel_weights))
                        conv[conv_row][conv_col] = conv_value
                        col += self.stride
                    col = 0
                    row += self.stride
                output_convolutions.append(conv)
            out.append(output_convolutions)

        out = np.array(out)
        return out

    
