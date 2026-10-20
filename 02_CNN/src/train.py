import numpy as np
from src.model import ScratchCNN, VecScratchCNN
from src.optimizer import SGD

def train(model: ScratchCNN, inputs: np.ndarray, targets: np.ndarray, lr=0.1, n_epochs = 50, trace=True):
    """
    Receives: model (MLP), input X (Nxd), targets Y (Nxm), learning rate lr, number of epochs n_epochs
    """
    N= np.shape(inputs)[0]
    loss_history = []
    optimizer = SGD(lr=lr)

    for epoch in range(n_epochs):
        # at every epoch, reset gradient to 0 to avoid accumulation
        optimizer.zero_grad(model)

        epoch_loss = 0.0

        for i, x in enumerate(inputs):
            y_label = targets[i]
            # forward pass
            y_pred, cache = model.forward(x)
            # Compute and accumulate loss for this sample
            sample_loss = model.loss.forward(y_pred, y_label)
            epoch_loss += sample_loss
            # compute gradients
            dW_conv, db_conv, dW1, db1, dW2, db2 = model.backpropagation(y_label, cache)
            # Accumulate gradients
            model.dW_conv += dW_conv
            model.db_conv += db_conv
            model.dW1 += dW1
            model.db1 += db1
            model.dW2 += dW2
            model.db2 += db2

        # Average loss and gradients over the batch size (N)
        epoch_loss /= N
        loss_history.append(epoch_loss)

        if trace and epoch % 5 == 0:
            print(f"Epoch {epoch}: current loss is {epoch_loss}")
        model.dW_conv /= N
        model.db_conv /= N
        model.dW1 /= N
        model.db1 /= N
        model.dW2 /= N
        model.db2 /= N
        # update weights
        optimizer.step(model)
        #print(f"done with epoch {epoch}, moving to epoch {epoch+1}")

    return loss_history

def trainVec(model, inputs: np.ndarray, targets: np.ndarray, lr=0.1, n_epochs=50, trace=True):
    """
    Receives: model, input X (N, H, W), targets Y (N, m), learning rate lr, number of epochs n_epochs
    """
    N = inputs.shape[0]  # Safe batch size extraction for 3D tensors
    loss_history = []
    optimizer = SGD(lr=lr)

    for epoch in range(n_epochs):
        # Reset gradients to 0
        optimizer.zero_grad(model)

        # 1. Full batch forward pass
        Y_pred, cache = model.forward(inputs)
        
        # 2. Compute batch loss
        epoch_loss = model.loss.forward(Y_pred, targets)
        loss_history.append(epoch_loss)

        # 3. Full batch backpropagation (unpacking all 6 gradient components)
        model.dW_conv, model.db_conv, model.dW1, model.db1, model.dW2, model.db2 = model.backpropagation(targets, cache)

        if trace and epoch % 5 == 0:
            print(f"Epoch {epoch}: current loss is {epoch_loss}")

        # 4. Update weights using optimizer
        optimizer.step(model)

    return loss_history