from src.model import ScratchRNN
from utils.activations import tanh_activation
from utils.losses import softmax_CSE_loss, Vec_softmax_CSE_loss
from src.optimizer import SGD

import numpy as np

def train(model: ScratchRNN, inputs: np.ndarray, targets: np.ndarray, lr=0.1, n_epochs = 50, trace=True):
    # inputs: dim (N, T, d)
    # targets: dim (N, T, 1)

    N = np.shape(inputs)[0]
    loss_history = []
    optimizer = SGD(model=model, lr=lr)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        epoch_loss = 0
        for i in range(N):
            X = inputs[i]
            y_label = targets[i]
            sample_loss, cache =  model.forward(X, y_label)
            epoch_loss += sample_loss
            dW_x, dW_h, dW_z, db_h, db_z = model.backpropagation(cache)
            model.dW_x = dW_x
            model.dW_h = dW_h
            model.dW_z = dW_z
            model.db_h = db_h
            model.db_z = db_z

        epoch_loss /= N
        model.dW_x /= N
        model.dW_h /= N
        model.dW_z /= N
        model.db_h /= N
        model.db_z /= N
        loss_history.append(epoch_loss)
        optimizer.step()

        return loss_history



def trainVec(model, inputs, targets, lr=0.1, n_epochs=50, trace=True):
    N = inputs.shape[0]
    loss_history = []
    optimizer = SGD(model=model, lr=lr)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Forward pass for the entire batch
#        print("Input dim: ", np.shape(inputs))
        loss, cache = model.forward(inputs, targets)

        # Backward pass for the entire batch
        dW_x, dW_h, dW_z, db_h, db_z = model.backpropagation(cache)

        # Store gradients for the optimizer
        model.dW_x = dW_x
        model.dW_h = dW_h
        model.dW_z = dW_z
        model.db_h = db_h
        model.db_z = db_z

        loss_history.append(loss)
        optimizer.step()

        if trace and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, loss: {loss:.4f}")

    return loss_history