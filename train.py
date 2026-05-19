import numpy as np
import os
from sklearn.model_selection import train_test_split

from neural_net.model import Sequential
from neural_net.layers import Dense
from neural_net.activations import LeakyReLU
from neural_net.losses import MSE
from neural_net.optimizers import SGDMomentum

if __name__ == "__main__":
    X_path = os.path.join("data", "parkinsons_X_processed.npy")
    y_path = os.path.join("data", "parkinsons_y_processed.npy")
    
    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        raise FileNotFoundError("Processed arrays not found. Run preprocess.py first!")
        
    X = np.load(X_path)
    y = np.load(y_path)
    
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
        
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    ALPHA = 0.1
    model = Sequential()
    
    model.add(Dense(input_dim=19, output_dim=64, alpha=ALPHA))
    model.add(LeakyReLU(alpha=ALPHA))
    
    model.add(Dense(input_dim=64, output_dim=32, alpha=ALPHA))
    model.add(LeakyReLU(alpha=ALPHA))
    
    model.add(Dense(input_dim=32, output_dim=1, alpha=ALPHA))
    
    optimizer = SGDMomentum(lr=0.001, beta=0.9)
    loss_function = MSE()
    
    EPOCHS = 100
    BATCH_SIZE = 32
    
    print(f"Training initialized on {X_train.shape[0]} samples. Validating on {X_val.shape[0]} samples...")
    print("-" * 60)
    
    for epoch in range(EPOCHS):
        model.fit(X_train, y_train, epochs=1, batch_size=BATCH_SIZE, optimizer=optimizer, loss_fn=loss_function)
        
        y_val_pred = model.forward(X_val)
        val_loss = loss_function.forward(y_val_pred, y_val)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"--- Epoch {epoch+1:3d}/{EPOCHS} Evaluation -> Validation MSE Loss: {val_loss:.4f}")
            print("-" * 60)