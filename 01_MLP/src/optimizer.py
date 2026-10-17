import numpy as np

class SGD:
    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self, model):
        # Unpack gradients
        model.W1 -= self.lr * model.dW1
        model.b1 -= self.lr * model.db1
        model.W2 -= self.lr * model.dW2
        model.b2 -= self.lr * model.db2

    def zero_grad(self, model):
        model.dW1 = np.zeros_like(model.W1)
        model.db1 = np.zeros_like(model.db1)
        model.dW2 = np.zeros_like(model.W2)
        model.db2 = np.zeros_like(model.db2)
