import numpy as np

class Attention:
    def __init__(self, N, d_m, d_k, d_v):
        # dimensions
        self.d_model = d_m
        self.d_k = d_k
        self.d_v = d_v

        # weights
        self.W_Q = np.random.rand(self.d_model, self.d_k)
        self.W_K = np.random.rand(self.d_model, self.d_k)
        self.W_V = np.random.rand(self.d_model, self.d_v)

    def softmax(self, S, mask=True):
        #S: NxN
        if mask == True:
            S = self.causal_mask(S)
        max_ = np.max(S, axis=1, keepdims=True)
        exp_S = np.exp(S-max_)
        sum_ = np.sum(exp_S, axis=1, keepdims=True)
        eps = 1e-12
        sum_ += eps
        score = exp_S / sum_
        return score

    def causal_mask(self, S):
        N = np.shape(S)[0]
        M = np.triu(np.ones_like(S)) - np.eye(N, N)
        return np.where(M, -np.inf, S)


    def attention_score(self, X):
        # X is input, Nxd_m
        Q = X @ self.W_Q # dim (N, d_k)
        K = X @ self.W_K # dim (N, d_k)
        prod = Q @ K.T / np.sqrt(self.d_k)
        A = self.softmax(prod)  
        return A      


    def compute_H(self, X):
        A = self.attention_score(X)
        V = X @ self.W_V # dim (N, d_v)
        H = A @ V
        return H
        

