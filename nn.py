from typing import Any
import numpy as np
import math

from numpy._typing import NDArray

class Loss:
    def forward(self, y_pred, y_true) -> NDArray[Any]:
        pass
    def calculate(self, output: np.ndarray, y: np.ndarray) -> np.floating[Any]:
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> NDArray[Any]:
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true
        ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true,
                                         axis=1
            )
        negative_loss_likelihoods = -np.log(correct_confidences)
        return negative_loss_likelihoods

class Activation_Softmax:
    def forward(self, inputs: np.ndarray):
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
    def forward(self, inputs: np.ndarray):
        # Iterate every element of the list and append if more than 0
        self.output = np.maximum(0, inputs)
class Layer_Dense:
    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs: np.ndarray):
        self.output = np.dot(inputs, self.weights) + self.biases
def acc(softmax_output: Activation_Softmax, y: np.ndarray) -> float:
    preds = np.argmax(softmax_output.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    return np.mean(preds==y)
