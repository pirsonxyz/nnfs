import numpy as np
import math

class Activation_Softmax:
    def forward(self, inputs):
        '''
        exp_values = []
        for input in inputs:
            exp_values.append(math.e ** input)
        # The probabilites 
        confidence_values = []
        sum_of_exp = sum(exp_values)
        for exp_value in exp_values:
            confidence_values.append(exp_value / sum_of_exp)
        '''
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilites = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilites

class Activation_ReLu:
    def forward(self, inputs):
        # Iterate every element of the list and append if more than 0
        self.output = np.maximum(0, inputs)
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
