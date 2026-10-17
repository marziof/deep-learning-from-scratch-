import sys
import os

# Go up one level from 'scripts/' to the '01_MLP' root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# Now Python can find both src and utils from the root
import numpy as np
import matplotlib.pyplot as plt

from src.model import ScratchMLP, VecScratchMLP
from src.train import train, trainVec
from utils.losses import MSELoss
from utils.activations import ReLU

if __name__ == "__main__":
    # Generate synthetic regression data
    np.random.seed(42)
    N=100
    X = np.random.randn(N, 3)
    true_weights = np.array([2.0, 0.0, -1.5])
    y = X @ true_weights + 0.5 + np.random.randn(N) * 0.1
    y = y.reshape(-1, 1)

    # Initialize and train
    model = ScratchMLP(input_dim=3, hidden_dim=5, output_dim=1)
    modelVec = VecScratchMLP(N=N,input_dim=3, hidden_dim=5, output_dim=1)
    #loss_history = train(model, X, y, lr=0.05, n_epochs=500, trace=True)
    loss_history = trainVec(modelVec, X, y, lr=0.05, n_epochs=500, trace=True)

    

    # Plot results
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("ScratchMLP Training Loss")
    plt.show()