

import torch

import sys
import os
import numpy as np
# Go up one level from 'scripts/' to the '01_MLP' root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
from src.model import BertEncoder
from src.mlm_head import MLM_head
from src.train import train, evaluate

from pathlib import Path
from src.tokenizer import WordPieceTokenizer

import matplotlib.pyplot as plt

def main():
    torch.manual_seed(42)

    #=======DATASET==========

    corpus = Path("05_Transformer/data/corpus.txt").read_text(encoding="utf-8")


    VOCAB_SIZE = 500
    tokenizer = WordPieceTokenizer.from_corpus(
        corpus,
        vocab_size=VOCAB_SIZE,
    )

    sentences = corpus.splitlines()
    sentences = [s for s in sentences if s.strip()]

    import random

    encoded = [
        tokenizer.encode(sentence)
        for sentence in sentences
    ]
    random.seed(42)
    random.shuffle(encoded)

    split_idx = int(0.8 * len(encoded))
    train_encoded = encoded[:split_idx]
    test_encoded = encoded[split_idx:]
    #========================


    ### INIT
    # Tiny configuration for testing
    vocab_size = len(tokenizer.vocab)
    model_dim = 16
    max_seq_len = max(len(seq) for seq in encoded)
    num_heads = 2
    hidden_dim = 32

    model = BertEncoder(
        vocab_size=vocab_size,
        model_dim=model_dim,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
    )

    mlm_head = MLM_head(model_dim = model_dim, vocab_size=vocab_size)

    optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(mlm_head.parameters()),
            lr=1e-3,
        )

    print("Total sentences:", len(encoded))
    print("Train sentences:", len(train_encoded))
    print("Test sentences:", len(test_encoded))
    print("Vocab size:", len(tokenizer.vocab))
    
    ### TRAIN LOOP
    NUM_EPOCHS = 100
    BATCH_SIZE = 8
    train_losses = []
    test_losses = []
    for epoch in range(NUM_EPOCHS):
        train_loss = train(
            model, mlm_head, optimizer,
            train_encoded, tokenizer, BATCH_SIZE
        )

        test_loss = evaluate(
            model, mlm_head,
            test_encoded, tokenizer, BATCH_SIZE
        )
        if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} "
                f"| Train loss: {train_loss:.4f} "
                f"| Test loss: {test_loss:.4f}"
            )
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    # plot train and test loss vs epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Train Loss")
    plt.plot(range(1, NUM_EPOCHS + 1), test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Test Loss vs Epochs")
    plt.legend()
    plt.grid()
    plt.savefig("05_Transformer/train_test_loss.png")





if __name__ == "__main__":
    main()