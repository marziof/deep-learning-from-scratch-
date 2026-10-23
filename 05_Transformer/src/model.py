import torch
from torch import nn
import math


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.linear_stack(x)


class Attention(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, output_dim):
        super().__init__()

        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        head_dim = model_dim // num_heads

        self.W_q = nn.Parameter(torch.rand(num_heads, input_dim, head_dim))
        self.W_k = nn.Parameter(torch.rand(num_heads, input_dim, head_dim))
        self.W_v = nn.Parameter(torch.rand(num_heads, input_dim, head_dim))
        self.W_o = nn.Parameter(torch.rand(model_dim, output_dim))

    def forward(self, X, mask=False, attention_mask=None):
        # X: (B, T, D)
        Q = torch.einsum('bti,hid->bhtd', X, self.W_q)
        K = torch.einsum('bti,hid->bhtd', X, self.W_k)
        V = torch.einsum('bti,hid->bhtd', X, self.W_v)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(Q.shape[-1])

        # Ignore PAD positions as attention keys
        if attention_mask is not None:
            padding_mask = ~attention_mask[:, None, None, :].bool()
            scores = scores.masked_fill(padding_mask, float('-inf'))

        # Optional causal mask (not used for BERT)
        if mask:
            T = X.shape[1]
            causal_mask = torch.triu(
                torch.ones(T, T, device=X.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))

        attention = torch.softmax(scores, dim=-1)
        prod = attention @ V
        prod = prod.transpose(1, 2).contiguous()
        prod = prod.reshape(X.shape[0], X.shape[1], -1)

        return prod @ self.W_o

class TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads, hidden_dim, causal=False):
        super().__init__()

        self.causal = causal

        self.ln1 = nn.LayerNorm(model_dim)
        self.mha = Attention(
            input_dim=model_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            output_dim=model_dim
        )

        self.ln2 = nn.LayerNorm(model_dim)
        self.mlp = MLP(
            input_dim=model_dim,
            hidden_dim=hidden_dim,
            output_dim=model_dim
        )

    def forward(self, X, attention_mask=None):
        h = X + self.mha(
            self.ln1(X),
            mask=self.causal,
            attention_mask=attention_mask,
        )
        y = h + self.mlp(self.ln2(h))
        return y



class BertEncoder(nn.Module):
    def __init__(self, vocab_size, model_dim, max_seq_len, num_heads, hidden_dim):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.pos_emb = nn.Embedding(max_seq_len, model_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(model_dim, num_heads, hidden_dim, causal=False)
        ])

    def forward(self, X, attention_mask=None):
        # X: (B, T), token IDs

        E_emb = self.tok_emb(X)

        position_ids = torch.arange(X.shape[1], device=X.device)
        E_pos = self.pos_emb(position_ids)

        y = E_emb + E_pos

        for block in self.transformer_blocks:
            y = block(y, attention_mask=attention_mask)

        return y






#======



# class Attention(nn.Module):
#     def __init__(self, input_dim, model_dim, num_heads, output_dim):
#         super().__init__()
#         if model_dim % num_heads != 0:
#             raise ValueError("model_dim must be divisible by num_heads")
#         head_dim = model_dim // num_heads
#         self.W_q = nn.Parameter(torch.rand(num_heads, input_dim, head_dim))
#         self.W_k = nn.Parameter(torch.rand(num_heads, input_dim, head_dim))
#         self.W_v = nn.Parameter(torch.rand(num_heads, input_dim, head_dim))
#         self.W_o = nn.Parameter(torch.rand(model_dim, output_dim))      

#     def forward(self, X, mask):
#         Q = torch.einsum('ij, hjk -> hik', X, self.W_q) # shape (H, N, dm/H)
#         K = torch.einsum('ij, hjk -> hik', X, self.W_k)
#         V = torch.einsum('ij, hjk -> hik', X, self.W_v)
#         N = X.shape[0]
#         M = torch.triu(torch.ones(N, N, device=X.device, dtype=torch.bool), diagonal=1)
#         dot_prod = torch.einsum('hni, hmi -> hnm', Q, K)/math.sqrt(Q.shape[-1]) # dim (H, N, N)
#         if mask:
#             dot_prod = dot_prod.masked_fill(M, float('-inf'))
#         attention = torch.softmax(dot_prod, dim=-1) # shape (H,N,N)
#         prod = attention @ V                  # (H, N, head_dim)
#         prod = prod.permute(1, 0, 2)          # (N, H, head_dim)
#         prod = prod.reshape(N, self.W_o.shape[0])     # (N, model_dim)
#         out = prod @ self.W_o
#         return out


