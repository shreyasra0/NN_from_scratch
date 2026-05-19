import numpy as np

class SGDMomentum:
    def __init__(self, lr=0.001, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v_weights = {}
        self.v_biases = {}

    def update(self, layer, layer_idx):
        if layer_idx not in self.v_weights:
            self.v_weights[layer_idx] = np.zeros_like(layer.weights)
            self.v_biases[layer_idx] = np.zeros_like(layer.biases)

        self.v_weights[layer_idx] = self.beta * self.v_weights[layer_idx] + (1.0 - self.beta) * layer.dweights
        self.v_biases[layer_idx] = self.beta * self.v_biases[layer_idx] + (1.0 - self.beta) * layer.dbiases

        layer.weights -= self.lr * self.v_weights[layer_idx]
        layer.biases -= self.lr * self.v_biases[layer_idx]