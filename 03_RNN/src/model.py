import numpy as np


class ScratchRNN():
    def __init__(self, time, input_dim, hidden_dim, output_dim, activation_function, loss_function):
        self.T = time
        self.d = input_dim
        self.h = hidden_dim
        self.m = output_dim

        self.W_x = np.random.rand(self.d, self.h)
        self.W_h = np.random.rand(self.h, self.h)
        self.W_z = np.random.rand(self.h, self.m)
        self.b_h = np.zeros(self.h)
        self.b_z = np.zeros(self.m)

        self.dW_x = np.random.rand(self.d, self.h)
        self.dW_h = np.random.rand(self.h, self.h)
        self.dW_z = np.random.rand(self.h, self.m)
        self.db_h = np.zeros(self.h)
        self.db_z = np.zeros(self.m)

        self.activation = activation_function
        self.loss = loss_function

    def forward(self, X, y_label):
        # X: array (T, d): each row is an input vector -> x_t = X[t,:]
        u = np.zeros((self.T, self.h))
        a = np.zeros((self.T, self.h))
        h = np.zeros((self.T, self.h))
        z = np.zeros((self.T, self.m))
        for t in range(self.T):
            u[t,:] = self.W_x.T @ X[t,:] # dim (h,)
            a[t,:] = u[t,:] + self.W_h.T @ h[t-1,:] + self.b_h if t>0 else u[t,:] + self.b_h
            h[t,:] = self.activation.forward(a[t,:])
            z[t,:] = self.W_z.T @ h[t,:] +self.b_z
        cache = (X,u,a,h, z)
        loss_value = self.loss.forward(z, y_label)
        return loss_value, (z, cache)

    def backpropagation(self, cache):
        X, u, a, h, z = cache
        T = self.T

        grad = self.loss.backward()  # (T, m): dL/dz

        # Output layer gradients
        dW_z = h.T @ grad            # (h, m)
        db_z = np.sum(grad, axis=0)  # (m,)

        # Initialize hidden-layer gradients
        dW_x = np.zeros_like(self.W_x)  # (d, h)
        dW_h = np.zeros_like(self.W_h)  # (h, h)
        db_h = np.zeros_like(self.b_h)  # (h,)

        da_next = np.zeros(self.h)  # Future timestep's preactivation gradient

        # Backpropagation through time
        for t in reversed(range(T)):
            # Gradient arriving at h_t from current output and future timestep
            dh = self.W_z @ grad[t] + self.W_h @ da_next

            # Gradient through activation
            da = dh * self.activation.backward(h[t])

            # Accumulate parameter gradients
            x_t = X[t]
            h_prev = h[t - 1] if t > 0 else np.zeros(self.h)

            dW_x += np.outer(x_t, da)
            dW_h += np.outer(h_prev, da)
            db_h += da

            # Pass gradient to previous timestep
            da_next = da

        # self.dW_x = dW_x
        # self.dW_h = dW_h
        # self.dW_z = dW_z
        # self.db_h = db_h
        # self.db_z = db_z
            
        return dW_x, dW_h, dW_z, db_h, db_z





class VecScratchRNN:
    def __init__(
        self, time, input_dim, hidden_dim, output_dim,
        activation_function, loss_function
    ):
        self.T = time
        self.d = input_dim
        self.h = hidden_dim
        self.m = output_dim

        self.W_x = np.random.randn(self.d, self.h) * 0.1
        self.W_h = np.random.randn(self.h, self.h) * 0.1
        self.W_z = np.random.randn(self.h, self.m) * 0.1

        self.b_h = np.zeros(self.h)
        self.b_z = np.zeros(self.m)

        self.activation = activation_function
        self.loss = loss_function

    def forward(self, X, y_label):
        # X: (N, T, d)
        N = X.shape[0]

        u = np.zeros((N, self.T, self.h))
        a = np.zeros((N, self.T, self.h))
        h = np.zeros((N, self.T, self.h))
        z = np.zeros((N, self.T, self.m))

        for t in range(self.T):
            # All N samples at timestep t
            u[:, t, :] = X[:, t, :] @ self.W_x

            h_prev = (
                h[:, t - 1, :]
                if t > 0
                else np.zeros((N, self.h))
            )

            a[:, t, :] = (
                u[:, t, :] + h_prev @ self.W_h + self.b_h
            )

            h[:, t, :] = self.activation.forward(a[:, t, :])

            z[:, t, :] = h[:, t, :] @ self.W_z + self.b_z

        # z: (N, T, m), labels: (N, T)
        loss_value = self.loss.forward(z, y_label)

        cache = (X, u, a, h, z)
        return loss_value, cache

    def backpropagation(self, cache):
        X, u, a, h, z = cache
        N, T, _ = X.shape

        # dL/dz: (N, T, m)
        grad = self.loss.backward()

        # Output-layer gradients: sum across batch and time
        H = h.reshape(-1, self.h)
        G = grad.reshape(-1, self.m)

        dW_z = H.T @ G
        db_z = np.sum(grad, axis=(0, 1))

        dW_x = np.zeros_like(self.W_x)
        dW_h = np.zeros_like(self.W_h)
        db_h = np.zeros_like(self.b_h)

        # Gradient carried backward through time, (N, h)
        da_next = np.zeros((N, self.h))

        for t in reversed(range(T)):
            h_t = h[:, t, :]

            # Gradient from current output + future recurrence
            dh = grad[:, t, :] @ self.W_z.T
            dh += da_next @ self.W_h.T

            # dL/da_t
            da = dh * self.activation.backward(h_t)

            # Accumulate parameter gradients over the batch
            dW_x += X[:, t, :].T @ da

            h_prev = (
                h[:, t - 1, :]
                if t > 0
                else np.zeros((N, self.h))
            )

            dW_h += h_prev.T @ da
            db_h += np.sum(da, axis=0)

            da_next = da

        return dW_x, dW_h, dW_z, db_h, db_z
