
import numpy as np

class softmax_CSE_loss:
    def forward(self, logits, y_label):
        # logits: (T, m)
        # y_label: (T,), integer class indices

        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probas = exp_logits / np.sum(exp_logits, axis=1, keepdims=True) # dim (T,m)

        T = logits.shape[0]
        true_probas = self.probas[np.arange(T), y_label]
        loss = -np.mean(np.log(np.clip(true_probas, 1e-12, 1.0)))

        self.y_label = y_label
        return loss

    def backward(self):
        # Gradient of mean softmax cross-entropy w.r.t. logits
        T = self.probas.shape[0]
        grad = self.probas.copy()
        grad[np.arange(T), self.y_label] -= 1
        grad /= T
        return grad # dim (m,)


class Vec_softmax_CSE_loss:
    def forward(self, logits, y_label):
        # logits: (N, T, m)
        # y_label: (N, T)
        N, T, m = logits.shape
        self.N, self.T, self.m = N, T, m

        self.logits_flat = logits.reshape(-1, m)
        self.y_label_flat = y_label.reshape(-1)

        exp_logits = np.exp(
            self.logits_flat
            - np.max(self.logits_flat, axis=1, keepdims=True)
        )
        self.probas = exp_logits / np.sum(
            exp_logits, axis=1, keepdims=True
        )

        total_elements = N * T
        true_probas = self.probas[
            np.arange(total_elements), self.y_label_flat
        ]

        return -np.mean(
            np.log(np.clip(true_probas, 1e-12, 1.0))
        )

    def backward(self):
        grad = self.probas.copy()
        total_elements = self.N * self.T

        grad[
            np.arange(total_elements), self.y_label_flat
        ] -= 1

        grad /= total_elements

        return grad.reshape(self.N, self.T, self.m)
