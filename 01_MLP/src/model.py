import numpy as np

from utils.losses import MSELoss
from utils.activations import ReLU

class ScratchMLP:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int=1, loss = None, activation = None):
        # initialize dimensions 
        self.d = input_dim # dimension of input data (1 sample)
        self.h = hidden_dim # dimension of hidden layer (number of neurons)
        self.m = output_dim # dimension of output label 

        # initialize weights
        self.W1 = np.random.rand(self.d, self.h)
        self.b1 = np.zeros(self.h)
        self.W2 = np.random.rand(self.h, self.m)
        self.b2 = np.zeros(self.m)

        # Gradient attributes (initialized to None or zeros)
        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.b2)

        # loss and activation functions
        self.loss = loss if loss is not None else MSELoss()
        self.activation = activation if activation is not None else ReLU()

    def forward(self, x: np.ndarray): 
        # computes forward pass y=W2(sigma(W1*x + b1)) + b2
        z = self.W1.T @ x + self.b1 # dim h
        a = self.activation.forward(z) # dim h
        y = self.W2.T @ a + self.b2 # dim m

        cache = (x, z, a) # to use for backpropagation when done right after forward (avoid recomputing)
        return y, cache

    def backpropagation(self, y_label, cache):
        # computes partial derivatives of y wrt weights
        x, z, a = cache
        y = self.W2.T @ a + self.b2

        # derivative of MSE loss wrt output y
        dy = self.loss.backward(y,y_label)  # shape (m,)

        db2 = dy # shape (m,)
        dW2 = np.outer(a, dy) # shape (h,m)

        dz = (self.W2 @ dy) * self.activation.backward(z)  # Shape (h,)

        db1 = dz
        dW1 = np.outer(x, dz)

        return dW1, db1, dW2, db2




class VecScratchMLP:
    def __init__(self, N: int, input_dim: int, hidden_dim: int, output_dim: int=1, loss = None, activation = None):
        # initialize dimensions 
        self.N = N
        self.d = input_dim # dimension of input data (1 sample)
        self.h = hidden_dim # dimension of hidden layer (number of neurons)
        self.m = output_dim # dimension of output label 

        # initialize weights
        self.W1 = np.random.rand(self.d, self.h)
        self.b1 = np.zeros(self.h)
        self.W2 = np.random.rand(self.h, self.m)
        self.b2 = np.zeros(self.m)

        # Gradient attributes (initialized to None or zeros)
        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.b2)

        # loss and activation functions
        self.loss = loss if loss is not None else MSELoss()
        self.activation = activation if activation is not None else ReLU()

    def forward(self, X: np.ndarray): 
        # Z shape: (N, h) = (N, d) @ (d, h) + (h,)
        Z = X @ self.W1 + self.b1 
        
        # A shape: (N, h)
        A = self.activation.forward(Z) 
        
        # Y shape: (N, m) = (N, h) @ (h, m) + (m,)
        Y = A @ self.W2 + self.b2 

        cache = (X, Z, A) 
        return Y, cache

    def backpropagation(self, Y_label: np.ndarray, cache):
        X, Z, A = cache
        N = X.shape[0]
        Y = A @ self.W2 + self.b2  # shape (N, m)

        # 1. Derivative of loss wrt output Y
        dY = self.loss.backward(Y, Y_label)  # shape (N, m)

        # 2. Gradients for Layer 2 (W2 and b2)
        db2 = np.sum(dY, axis=0)             # shape (m,) - sum over batch
        dW2 = A.T @ dY                       # shape (h, m) = (h, N) @ (N, m)

        # 3. Gradients for Layer 1 (W1 and b1)
        dZ = (dY @ self.W2.T) * self.activation.backward(Z)  # shape (N, h)
        
        db1 = np.sum(dZ, axis=0)             # shape (h,) - sum over batch
        dW1 = X.T @ dZ                       # shape (d, h) = (d, N) @ (N, h)

        return dW1, db1, dW2, db2


class VecScratchMLP_CSE:
    def __init__(self, N: int, input_dim: int, hidden_dim: int, output_dim: int=1, loss = None, activation = None):
        # initialize dimensions 
        self.N = N
        self.d = input_dim # dimension of input data (1 sample)
        self.h = hidden_dim # dimension of hidden layer (number of neurons)
        self.m = output_dim # dimension of output label 

        # initialize weights
        self.W1 = np.random.rand(self.d, self.h)
        self.b1 = np.zeros(self.h)
        self.W2 = np.random.rand(self.h, self.m)
        self.b2 = np.zeros(self.m)

        # Gradient attributes (initialized to None or zeros)
        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.b2)

        # loss and activation functions
        self.loss = loss if loss is not None else MSELoss()
        self.activation = activation if activation is not None else ReLU()

    def forward(self, X: np.ndarray): 
        # Z shape: (N, h) = (N, d) @ (d, h) + (h,)
        Z = X @ self.W1 + self.b1 
        
        # A shape: (N, h)
        A = self.activation.forward(Z) 
        
        Z2 = A @ self.W2 + self.b2
        Y = softmax(Z2)
        cache = (X, Z, A, Y)
        return Y, cache

    def backpropagation(self, Y_label, cache):
        X, Z, A, Y = cache
        N = X.shape[0]

        dZ2 = (Y - Y_label) / N

        dW2 = A.T @ dZ2
        db2 = np.sum(dZ2, axis=0)

        dZ1 = (dZ2 @ self.W2.T) * self.activation.backward(Z)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)

        return dW1, db1, dW2, db2


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)