import numpy as np

class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        out = inputs
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, loss_gradient):
        grad = loss_gradient
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def fit(self, X, y, epochs, batch_size, optimizer, loss_fn):
        num_samples = X.shape[0]
        
        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0.0
            num_batches = int(np.ceil(num_samples / batch_size))
            
            for b in range(num_batches):
                start_idx = b * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                y_pred = self.forward(X_batch)
                
                loss = loss_fn.forward(y_pred, y_batch)
                epoch_loss += loss * (end_idx - start_idx)
                
                loss_grad = loss_fn.backward(y_pred, y_batch)
                
                self.backward(loss_grad)
                
                for idx, layer in enumerate(self.layers):
                    if hasattr(layer, 'weights'):
                        optimizer.update(layer, idx)
                        
            epoch_loss /= num_samples
            if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")