import sys
import os

# Go up one level from 'scripts/' to the '01_MLP' root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

from src.model import ScratchCNN, VecScratchCNN
from src.train import train, trainVec
from utils.losses import SoftmaxCrossEntropyLoss

if __name__ == "__main__":
    # Load MNIST
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False
    )

    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int64)

    # Reshape flat 784 vectors into 28x28 2D images for the CNN!
    X = X.reshape(-1, 28, 28)

    # Standard MNIST train/test split
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # One-hot encode labels
    Y_train = np.eye(10)[y_train]
    Y_test = np.eye(10)[y_test]

    # Initialize model with proper CNN dimensions
    np.random.seed(42)
    model = VecScratchCNN(
        input_dim=28,          # 28x28 image
        convolution_dim=5,     # 3x3 kernel
        convolution_stride=2,
        pooling_dim=3,         # 2x2 max pool
        pooling_stride=2,
        hidden_dim=128,
        output_dim=10,         # 10 digit classes
        loss=SoftmaxCrossEntropyLoss()
    )

    # Train using your training loop
    loss_history = trainVec(
        model, X_train, Y_train,
        lr=0.1, n_epochs=50, trace=True
    )

    # Evaluate on test set (looping sample-by-sample to match forward pass)
    predictions = []
    for x in X_test:
        y_pred, _ = model.forward(x)
        predictions.append(np.argmax(y_pred))
        
    predictions = np.array(predictions)
    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    print(f"Test accuracy: {accuracy:.4%}")

    # Plot loss
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("MNIST MLP Training Loss")
    plt.show()
