import sys
import os

# Go up one level from 'scripts/' to the '01_MLP' root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

from src.model import ScratchRNN, VecScratchRNN
from src.train import train, trainVec
from utils.losses import Vec_softmax_CSE_loss
from utils.activations import tanh_activation
from src.optimizer import SGD


text = (
    "hello world! "
    "deep learning is interesting. "
    "I am learning recurrent neural networks. "
) * 200
chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}

encoded = np.array([char_to_idx[c] for c in text])

T = 5  # sequence length

X_idx = np.array([
    encoded[i:i+T] for i in range(len(encoded) - T)
])
y_idx = np.array([
    encoded[i+1:i+T+1] for i in range(len(encoded) - T)
])

# One-hot encode inputs
V = len(chars)
X = np.eye(V)[X_idx]  # (N, T, V)
print(np.shape(X))
y = y_idx             # (N, T)


T = X.shape[1]

model = VecScratchRNN(
    time=T,
    input_dim=V,
    hidden_dim=16,
    output_dim=V,
    activation_function=tanh_activation(),
    loss_function=Vec_softmax_CSE_loss()
)

optimizer = SGD(model=model, lr=0.1)


loss_history = trainVec(
    model, X, y,
    lr=0.1,
    n_epochs=400,
    trace=True
)

loss, cache = model.forward(X[:1], y[:1])
z = cache[-1]  # (1, T, V)

predicted_idx = np.argmax(z[0], axis=-1)

predicted_text = "".join(chars[i] for i in predicted_idx)
target_text = "".join(chars[i] for i in y[0])

print("Target:    ", target_text)
print("Prediction:", predicted_text)

plt.figure(figsize=(8,6))
plt.plot(loss_history)
plt.title("Loss curve")
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.show()