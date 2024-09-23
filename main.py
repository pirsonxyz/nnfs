#!/usr/bin/env python
import numpy as np
import math
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
from nn import Layer_Dense, Activation_ReLu, Activation_Softmax, Loss_CategoricalCrossentropy, acc

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
loss_fn = Loss_CategoricalCrossentropy()
loss = loss_fn.calculate(activation_2.output, y)
print('loss:', loss)
accuracy = acc(activation_2, y)
print('acc: ', accuracy)


