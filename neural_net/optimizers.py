import numpy as np

class AdamW:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        self.m_weights = {}
        self.v_weights = {}
        self.m_biases = {}
        self.v_biases = {}
        self.t = 0

    def update(self, layer, layer_idx):
        if layer_idx not in self.m_weights:
            self.m_weights[layer_idx] = np.zeros_like(layer.weights)
            self.v_weights[layer_idx] = np.zeros_like(layer.weights)
            self.m_biases[layer_idx] = np.zeros_like(layer.biases)
            self.v_biases[layer_idx] = np.zeros_like(layer.biases)

        dw = layer.dweights
        db = layer.dbiases

        self.m_weights[layer_idx] = self.beta1 * self.m_weights[layer_idx] + (1.0 - self.beta1) * dw
        self.m_biases[layer_idx]  = self.beta1 * self.m_biases[layer_idx]  + (1.0 - self.beta1) * db

        self.v_weights[layer_idx] = self.beta2 * self.v_weights[layer_idx] + (1.0 - self.beta2) * (dw ** 2)
        self.v_biases[layer_idx]  = self.beta2 * self.v_biases[layer_idx]  + (1.0 - self.beta2) * (db ** 2)

        m_w_corrected = self.m_weights[layer_idx] / (1.0 - self.beta1 ** self.t)
        m_b_corrected = self.m_biases[layer_idx]  / (1.0 - self.beta1 ** self.t)
        
        v_w_corrected = self.v_weights[layer_idx] / (1.0 - self.beta2 ** self.t)
        v_b_corrected = self.v_biases[layer_idx]  / (1.0 - self.beta2 ** self.t)

        layer.weights -= self.lr * (m_w_corrected / (np.sqrt(v_w_corrected) + self.epsilon) + self.weight_decay * layer.weights)
        layer.biases  -= self.lr * (m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon))