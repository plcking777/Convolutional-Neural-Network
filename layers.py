import numpy as np


class Model():

    def __init__(self, layers):
        self.layers = layers
        

        prev_output_size = None
        for layer in self.layers:
            layer.init_weights_and_biases(prev_output_size)
            prev_output_size = layer.size



    
    def forward(self):
        
        prev_output = self.model_input

        for idx in range(len(self.layers)):
            layer = self.layers[idx]

            if isinstance(layer, Dense):
                current_output = layer.forward(prev_output)

            prev_output = current_output
    
        self.model_output = prev_output


    def cost(self, target):
        #TODO maybe average
        diff = self.get_output() - target
        return diff ** 2

    def cost_derivative(self, target):
        return (self.get_output() - target) * 2

    def backward(self, target):
        part_deriv = self.cost_derivative(target)

        for idx in range(len(self.layers)):
            pass



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


    def forward(self, input):
        if self._is_input:
            return input
        
        return self.map_activation(input.dot(self.weights) + self.biases)
        
    
    def map_activation(self, xs):
        # TODO optimize
        for i in range(len(xs)):
            for j in range(len(xs[i])):
                xs[i][j] = self.activation(xs[i][j])
        return xs        


    def init_weights_and_biases(self, input_size):
        if input_size == None:
            self._is_input = True
        else:
            self.weights = np.random.rand(input_size, self.size) - 0.5
            self.biases = np.random.rand(1, self.size) - 0.5


    


class Convolution():
    
    def __init__(self, kernel_count, kernel_shape):
        self.kernel_count = kernel_count
        self.kernel_shape = kernel_shape
    

