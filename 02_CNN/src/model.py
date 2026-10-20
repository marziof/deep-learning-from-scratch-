import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from utils.losses import MSELoss, SoftmaxCrossEntropyLoss
from utils.activations import ReLU


class ScratchCNN:
    def __init__(self, input_dim: int, 
                 convolution_dim: int,
                 convolution_stride: int,
                 pooling_dim: int,
                 pooling_stride: int,
                 hidden_dim: int,
                 output_dim: int=1, 
                 loss = None, activation = None
        ):
        # initialize dimensions 
        self.d = input_dim # dimension of input data (1 sample)
        self.k = convolution_dim
        self.s_c = convolution_stride
        self.p = pooling_dim
        self.s_p = pooling_stride
        self.h = hidden_dim # dimension of hidden layer (number of neurons)
        self.m = output_dim # dimension of output label 
        self.conv_dim = (self.d-self.k)//self.s_c + 1
        self.pool_dim = (self.conv_dim-self.p)//self.s_p + 1

        # initialize weights
        self.W_conv = np.random.rand(self.k, self.k)
        self.b_conv = 0.0
    
        self.W1 = np.random.rand(self.pool_dim*self.pool_dim, self.h)
        self.b1 = np.zeros(self.h)
        self.W2 = np.random.rand(self.h, self.m)
        self.b2 = np.zeros(self.m)

        # Gradient attributes (initialized to None or zeros)
        self.dW_conv = np.zeros_like(self.W_conv)
        self.db_conv = 0.0
        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.b2)

        # loss and activation functions
        self.loss = loss if loss is not None else SoftmaxCrossEntropyLoss()
        self.activation = activation if activation is not None else ReLU()

    def forward_loops(self, X: np.ndarray): 
        # computes forward pass y=W2(sigma(W1*x + b1)) + b2
        Z = np.zeros((self.conv_dim, self.conv_dim))
        for u in range(self.conv_dim):
            for v in range(self.conv_dim):
                Z[u,v] = np.sum(self.W_conv * X[u*self.s_c:u*self.s_c+self.k, v*self.s_c:v*self.s_c+self.k])+ self.b_conv
        A = self.activation.forward(Z)
        A_max_pooling = np.zeros((self.pool_dim, self.pool_dim))
        for u in range(self.conv_dim):
            for v in range(self.conv_dim):
                A_max_pooling[u,v] = np.max(A[u*self.s_p:u*self.s_p+self.p, v*self.s_p:v*self.s_p+self.p])
        x = A_max_pooling.flatten() # pxp -> 1 layer, p^2
        z = self.W1.T @ x + self.b1 # dim h
        a = self.activation.forward(z) # dim h
        y = self.W2.T @ a + self.b2 # dim m

        cache = (X, Z, A, A_max_pooling, x, z, a, y)
        return y, cache

    def forward(self, X: np.ndarray): 
        # computes forward pass y=W2(sigma(W1*x + b1)) + b2
        conv_window = sliding_window_view(X, (self.k, self.k))[::self.s_c, ::self.s_c]
        Z = np.sum(self.W_conv * conv_window, axis=(-2, -1))+ self.b_conv
        A = self.activation.forward(Z)
        pool_window = sliding_window_view(A, (self.p, self.p))[::self.s_p, ::self.s_p]
        # shape becomes: (pool_dim, pool_dim, p*p)
        pool_flat = pool_window.reshape(self.pool_dim, self.pool_dim, self.p * self.p)
        max_indices = np.argmax(pool_flat, axis=-1) # indices of max in each window
        
        A_max_pooling = np.max(pool_flat, axis=-1)
        x = A_max_pooling.flatten() # pxp -> 1 layer, p^2
        z = self.W1.T @ x + self.b1 # dim h
        a = self.activation.forward(z) # dim h
        y = self.W2.T @ a + self.b2 # dim m

        cache = (X, conv_window, Z, A, pool_window, max_indices, x, z, a)
        return y, cache

    def backpropagation(self, y_label, cache):
        # computes partial derivatives of y wrt weights
        X, conv_window, Z, A, pool_window, max_indices, x, z, a = cache
        y = self.W2.T @ a + self.b2
        # derivative of MSE loss wrt output y
        dy = self.loss.backward(y,y_label)  # shape (m,)
        db2 = dy # shape (m,)
        dW2 = np.outer(a, dy) # shape (h,m)
        dz = (self.W2 @ dy) * self.activation.backward(z)  # Shape (h,)
        db1 = dz
        dW1 = np.outer(x, dz)

        dx =self.W1 @ dz # shape(pool*pool,)
        d_pool = dx.reshape(self.pool_dim, self.pool_dim) 
        dA = np.zeros((self.conv_dim, self.conv_dim))
        # Create grid of (u, v) coordinates for the pooling layer
        U, V = np.indices((self.pool_dim, self.pool_dim))
        # Convert flat argmax indices to local window offsets (pi, pj)
        pi, pj = divmod(max_indices, self.p)
        # Calculate absolute coordinates in the full dA grid
        orig_u = U * self.s_p + pi
        orig_v = V * self.s_p + pj
        # np.add.at safely accumulates gradients (handling any potential overlapping pools)
        np.add.at(dA, (orig_u.flatten(), orig_v.flatten()), d_pool.flatten())
        dZ = dA * self.activation.backward(Z)
        db_conv = np.sum(dZ)
        dW_conv = np.sum(conv_window * dZ[..., np.newaxis, np.newaxis], axis=(0, 1))
        return dW_conv, db_conv, dW1, db1, dW2, db2



import numpy as np

class ReLU:
    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, Z):
        return (self.Z > 0).astype(float)

class VecScratchCNN:
    def __init__(self, input_dim=28, convolution_dim=3, convolution_stride=1, 
                 pooling_dim=2, pooling_stride=2, hidden_dim=128, output_dim=10, loss=None):
        self.k = convolution_dim
        self.s_c = convolution_stride
        self.p = pooling_dim
        self.s_p = pooling_stride
        self.h = hidden_dim
        self.m = output_dim
        self.loss = loss
        # Use two separate activation instances so they don't overwrite each other's cache!
        self.conv_activation = ReLU()
        self.dense_activation = ReLU()
        
        # Calculate spatial dimensions
        self.H = input_dim
        self.conv_dim = (self.H - self.k) // self.s_c + 1
        self.pool_dim = (self.conv_dim - self.p) // self.s_p + 1
        self.flat_dim = self.pool_dim * self.pool_dim
        
        # Initialize Weights & Biases (floats)
        self.W_conv = np.random.randn(self.k, self.k) * 0.1
        self.b_conv = 0.0
        
        self.W1 = np.random.randn(self.flat_dim, self.h) * 0.1
        self.b1 = np.zeros(self.h, dtype=np.float32)
        
        self.W2 = np.random.randn(self.h, self.m) * 0.1
        self.b2 = np.zeros(self.m, dtype=np.float32)
        
        # Initialize Gradient placeholders
        self.dW_conv = np.zeros_like(self.W_conv)
        self.db_conv = 0.0
        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.b2)

    def forward(self, X: np.ndarray):
        """
        X shape: (N, H, W)
        """

        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)
            
        N = X.shape[0]
        
        # 1. Convolution Layer (Batch Sliding Window)
        conv_window = sliding_window_view(X, (self.k, self.k), axis=(-2, -1))
        conv_window = conv_window[:, ::self.s_c, ::self.s_c, :, :]
    
        
        # Z shape: (N, conv_dim, conv_dim)
        Z = np.sum(self.W_conv * conv_window, axis=(-2, -1)) + self.b_conv
        A = self.conv_activation.forward(Z)  # Use conv_activation
        
        # 2. Max Pooling Layer
        pool_window = sliding_window_view(A, (self.p, self.p), axis=(-2, -1))
        pool_window = pool_window[:, ::self.s_p, ::self.s_p, :, :]
        
        pool_flat = pool_window.reshape(N, self.pool_dim, self.pool_dim, self.p * self.p)
        max_indices = np.argmax(pool_flat, axis=-1)  # shape: (N, pool_dim, pool_dim)
        A_max_pooling = np.max(pool_flat, axis=-1)   # shape: (N, pool_dim, pool_dim)
        
        # Flatten for Dense Layers: shape (N, flat_dim)
        x_flat = A_max_pooling.reshape(N, -1)
        
        # 3. Dense Layers
        z1 = x_flat @ self.W1 + self.b1  # shape: (N, h)
        a1 = self.dense_activation.forward(z1)  # Use dense_activation
        
        y = a1 @ self.W2 + self.b2       # shape: (N, m)
        
        cache = (X, conv_window, Z, A, pool_window, max_indices, x_flat, z1, a1, y)
        return y, cache

    def backpropagation(self, targets: np.ndarray, cache):
        X, conv_window, Z, A, pool_window, max_indices, x_flat, z1, a1, y = cache
        N = X.shape[0]
        
        # Loss gradient wrt output logits
        dy = self.loss.backward(y, targets)  # shape: (N, m)
        
        # Dense Layer 2 Gradients (averaged over batch N)
        dW2 = (a1.T @ dy) / N
        db2 = np.sum(dy, axis=0) / N
        
        # Dense Layer 1 Gradients
        dz1 = (dy @ self.W2.T) * self.dense_activation.backward(z1)  # Use dense_activation
        dW1 = (x_flat.T @ dz1) / N
        db1 = np.sum(dz1, axis=0) / N
        
        # Backprop into flattened pooling output
        dx_flat = dz1 @ self.W1.T  # shape: (N, flat_dim)
        d_pool = dx_flat.reshape(N, self.pool_dim, self.pool_dim)
        
        # Max Pooling Backward (Loop-free accumulation per batch item using add.at)
        dA = np.zeros_like(A)  # shape: (N, conv_dim, conv_dim)
        U, V = np.indices((self.pool_dim, self.pool_dim))
        
        for b in range(N):
            pi, pj = divmod(max_indices[b], self.p)
            orig_u = U * self.s_p + pi
            orig_v = V * self.s_p + pj
            np.add.at(dA[b], (orig_u.flatten(), orig_v.flatten()), d_pool[b].flatten())
            
        # Activation Backward for Convolution
        dZ = dA * self.conv_activation.backward(Z)  # Use conv_activation
        
        # Convolution Gradients (averaged over batch N)
        db_conv = np.sum(dZ) / N
        dW_conv = np.sum(conv_window * dZ[..., np.newaxis, np.newaxis], axis=(0, 1, 2)) / N
        
        return dW_conv, db_conv, dW1, db1, dW2, db2