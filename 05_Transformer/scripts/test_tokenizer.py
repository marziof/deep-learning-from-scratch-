

import sys
import os
# Go up one level from 'scripts/' to the '01_MLP' root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from pathlib import Path
from src.tokenizer import WordPieceTokenizer

corpus = Path("05_Transformer/data/corpus.txt").read_text(encoding="utf-8")

tokenizer = WordPieceTokenizer.from_corpus(
    corpus,
    vocab_size=500,
)

text = "The cat chased the dog."

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)
decoded = tokenizer.decode(token_ids)

print("Vocabulary size:", len(tokenizer.vocab))
print("Tokens:", tokens)
print("Token IDs:", token_ids)
print("Decoded:", decoded)

print("Special token IDs:")
print("PAD:", tokenizer.vocab["[PAD]"])
print("UNK:", tokenizer.vocab["[UNK]"])
print("MASK:", tokenizer.vocab["[MASK]"])
