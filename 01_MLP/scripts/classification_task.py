import sys
import os

# Go up one level from 'scripts/' to the '01_MLP' root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

from src.model import VecScratchMLP_CSE
from src.train import trainVec
from utils.losses import CrossEntropyLoss

if __name__ == "__main__":
    # Load MNIST
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False
    )

    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int64)

    # Standard MNIST train/test split
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # One-hot encode labels
    Y_train = np.eye(10)[y_train]
    Y_test = np.eye(10)[y_test]

    # Initialize model
    np.random.seed(42)
    modelVecCSE = VecScratchMLP_CSE(
        N=len(X_train),
        input_dim=784,
        hidden_dim=128,
        output_dim=10,
        loss=CrossEntropyLoss()
    )

    # Train
    loss_history = trainVec(
        modelVecCSE, X_train, Y_train,
        lr=0.05, n_epochs=100, trace=True
    )

    # Evaluate
    Y_pred, _ = modelVecCSE.forward(X_test)
    predictions = np.argmax(Y_pred, axis=1)
    accuracy = np.mean(predictions == y_test)

    print(f"Test accuracy: {accuracy:.4%}")

    # Plot loss
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("MNIST MLP Training Loss")
    plt.show()
