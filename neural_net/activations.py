import numpy as np

class LeakyReLU:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return np.where(inputs > 0, inputs, inputs * self.alpha)

    def backward(self, output_gradient):
        dx = np.ones_like(self.inputs)
        dx[self.inputs <= 0] = self.alpha
        return output_gradient * dx