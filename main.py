#!/usr/bin/env python

import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
from nn import Layer_Dense, Activation_ReLu, Activation_Softmax

nnfs.init()

x, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3)
activation_1 = Activation_ReLu()

dense2 = Layer_Dense(3, 3)

activation_2 = Activation_Softmax()
dense1.forward(x)
activation_1.forward(dense1.output)
dense2.forward(activation_1.output)
activation_2.forward(dense2.output)
print(activation_2.output[:5])


'''
X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()
inputs = [[1.0, 2.0, 3.0, 2.5], 
          [2.0, 5.0, -1.0, 2.0], 
          [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0], 
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
weights2 = [[0.1, -0.14, 0.5], 
           [-0.5, 0.12, -0.33], 
           [-0.44, 0.73, -0.13]]
biases2 = [-1, 2 , -0.5]
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(layer2_outputs)
'''
