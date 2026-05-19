import numpy as np

class Dense:
    def __init__(self, input_dim, output_dim, alpha=0.1):
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / ((1.0 + alpha**2) * input_dim))
        self.biases = np.zeros((1, output_dim))
        self.inputs = None
        self.dweights = None
        self.dbiases = None

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, output_gradient):
        self.dweights = np.dot(self.inputs.T, output_gradient)
        self.dbiases = np.sum(output_gradient, axis=0, keepdims=True)
        return np.dot(output_gradient, self.weights.T)