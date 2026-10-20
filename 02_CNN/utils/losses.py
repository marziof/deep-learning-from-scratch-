import numpy as np


class MSELoss:
    def forward(self, y_pred: np.ndarray, y_label: np.ndarray) -> float:
        return 0.5 * np.mean((y_pred - y_label) ** 2)
    
    def backward(self, y_pred: np.ndarray, y_label: np.ndarray) -> np.ndarray:
        N = y_pred.shape[0]
        # Divide by N to match the np.mean in forward()
        return (y_pred - y_label) / N


import numpy as np

# class SoftmaxCrossEntropyLoss:
#     def forward(self, logits: np.ndarray, y_label: np.ndarray) -> float:
#         """
#         logits: raw scores from the final layer of shape (m,)
#         y_label: one-hot encoded true labels of shape (m,)
#         """
#         # Numerically stable softmax
#         exp_logits = np.exp(logits - np.max(logits))
#         self.probs = exp_logits / np.sum(exp_logits)
        
#         # Cross-entropy loss
#         eps = 1e-12
#         return -np.sum(y_label * np.log(self.probs + eps))

#     def backward(self, logits: np.ndarray, y_label: np.ndarray) -> np.ndarray:
#         """
#         The beautiful derivative of Softmax + Cross-Entropy combined:
#         d(Loss) / d(logits) = probabilities - true_labels
#         """
#         return self.probs - y_label


class SoftmaxCrossEntropyLoss:
    def forward(self, logits: np.ndarray, y_label: np.ndarray) -> float:
        """
        logits: shape (N, m)
        y_label: shape (N, m)
        """
        # Numerically stable softmax per row (axis=-1)
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        self.probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Cross-entropy loss averaged over the batch
        eps = 1e-12
        loss = -np.sum(y_label * np.log(self.probs + eps))
        return loss / logits.shape[0]

    def backward(self, logits: np.ndarray, y_label: np.ndarray) -> np.ndarray:
        """
        Returns shape (N, m). Gradient division by N is handled 
        when computing weight gradients in backpropagation.
        """
        return self.probs - y_label