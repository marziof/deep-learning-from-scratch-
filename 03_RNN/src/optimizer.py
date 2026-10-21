import numpy as np

class SGD:
    def __init__(self, model, lr=0.1):
        self.model = model
        self.lr = lr

    def step(self):
        # Update Dense Layers
        self.model.W_x -= self.lr * self.model.dW_x
        self.model.W_h -= self.lr * self.model.dW_h
        self.model.W_z -= self.lr * self.model.dW_z
        self.model.b_h -= self.lr * self.model.db_h
        self.model.b_z -= self.lr * self.model.db_z

    def zero_grad(self):
        self.model.dW_x = np.zeros_like(self.model.W_x)
        self.model.dW_h = np.zeros_like(self.model.W_h)
        self.model.dW_z = np.zeros_like(self.model.W_z)
        self.model.db_h = np.zeros_like(self.model.b_h)
        self.model.db_z = np.zeros_like(self.model.b_z)
