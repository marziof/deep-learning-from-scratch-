import torch
from torch import nn

class MLM_head(nn.Module):
    def __init__(self, model_dim, vocab_size):
        super().__init__()
        self.linear_proj = nn.Linear(model_dim, vocab_size)

    def forward(self, X):
        # X: B, T, D
        y = self.linear_proj(X) #B, T, vocab size 
        return y
