import numpy as np


class MSELoss:
    def forward(self, y_pred: np.ndarray, y_label: np.ndarray) -> float:
        return 0.5 * np.mean((y_pred - y_label) ** 2)
    
    def backward(self, y_pred: np.ndarray, y_label: np.ndarray) -> np.ndarray:
        N = y_pred.shape[0]
        # Divide by N to match the np.mean in forward()
        return (y_pred - y_label) / N


class CrossEntropyLoss:
    def forward(self, y_pred: np.ndarray, y_label: np.ndarray) -> float:
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1.0)
        return -np.sum(y_label * np.log(y_pred)) / y_pred.shape[0]

    def backward(self, y_pred: np.ndarray, y_label: np.ndarray) -> np.ndarray:
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1.0)
        return -y_label / y_pred / y_pred.shape[0]