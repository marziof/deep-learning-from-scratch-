
import torch

import sys
import os
import numpy as np
# Go up one level from 'scripts/' to the '01_MLP' root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
from src.model import BertEncoder


def main():
    torch.manual_seed(42)

    # Tiny configuration for testing
    vocab_size = 100
    model_dim = 32
    max_seq_len = 16
    num_heads = 4
    hidden_dim = 64

    model = BertEncoder(
        vocab_size=vocab_size,
        model_dim=model_dim,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
    )

    # Two sequences, five tokens each
    input_ids = torch.randint(0, vocab_size, (2, 5))

    # Run the encoder
    output = model(input_ids)

    print("Input shape:", input_ids.shape)
    print("Output shape:", output.shape)

    assert output.shape == (2, 5, model_dim)
    assert torch.isfinite(output).all()

    print("Encoder test passed!")

if __name__ == "__main__":
    main()




# def main():
#     # Reproducibility
#     torch.manual_seed(42)

#     # Model configuration
#     model_dim = 64
#     num_heads = 4
#     hidden_dim = 256
#     seq_len = 10

#     # Create the Transformer block
#     block = TransformerBlock(
#         model_dim=model_dim,
#         num_heads=num_heads,
#         hidden_dim=hidden_dim
#     )

#     # Random input: (sequence_length, model_dim)
#     X = torch.randn(seq_len, model_dim, requires_grad=True)

#     # Forward pass
#     Y = block(X)

#     # Check output shape
#     print("Input shape: ", X.shape)
#     print("Output shape:", Y.shape)

#     assert Y.shape == X.shape, (
#         f"Expected output shape {X.shape}, got {Y.shape}"
#     )

#     # Check that outputs are finite
#     assert torch.isfinite(Y).all(), "Output contains NaN or Inf"

#     # Backward pass
#     loss = Y.square().mean()
#     loss.backward()

#     # Check gradients
#     assert X.grad is not None, "Input gradients are missing"

#     for name, param in block.named_parameters():
#         assert param.grad is not None, f"Missing gradient for {name}"
#         assert torch.isfinite(param.grad).all(), (
#             f"Non-finite gradient for {name}"
#         )

#     print("Forward pass: OK")
#     print("Backward pass: OK")
#     print("All shapes and gradients look good!")


# if __name__ == "__main__":
#     main()
