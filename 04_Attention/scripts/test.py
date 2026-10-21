
import sys
import os
import numpy as np
# Go up one level from 'scripts/' to the '01_MLP' root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.model import Attention


N, d_m, d_k, d_v = 5, 8, 4, 6
X = np.random.randn(N, d_m)

attn = Attention(N, d_m, d_k, d_v)
A = attn.attention_score(X)
H = attn.compute_H(X)

print(A.shape)  # (5, 5)
print(H.shape)  # (5, 6)
print(A.sum(axis=1))  # Each row should sum to 1

S = np.random.randn(5, 5)
A = attn.softmax(S, mask=True)

print(A)
print(A.sum(axis=1))