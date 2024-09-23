#!/usr/bin/env python
import nnfs
from nnfs.datasets import vertical_data
from nn import Layer_Dense, Activation_ReLu, Activation_Softmax, Loss_CategoricalCrossentropy, train
nnfs.init()

dataset = vertical_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3)
activation_1 = Activation_ReLu()
dense2 = Layer_Dense(3, 3)
activation_2 = Activation_Softmax()
lowest_loss = 9999999
loss_fn = Loss_CategoricalCrossentropy()

train(dense1, dense2, loss_fn, lowest_loss, 100000, dataset, activation_1, activation_2)



