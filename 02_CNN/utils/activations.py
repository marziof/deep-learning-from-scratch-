import numpy as np


class ReLU:
    def forward(self, z): return np.maximum(0, z)
    def backward(self, z): return np.where(z > 0, 1, 0)