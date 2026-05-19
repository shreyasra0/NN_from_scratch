import numpy as np

class MSE:
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2) / 2.0

    def backward(self, y_pred, y_true):
        return (y_pred - y_true) / y_pred.shape[0]