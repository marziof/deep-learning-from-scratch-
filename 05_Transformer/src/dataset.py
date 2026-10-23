import sys
import os
# Go up one level from 'scripts/' to the '01_MLP' root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from pathlib import Path
from src.tokenizer import WordPieceTokenizer

import torch
import random

def prepare_batch(batch, tokenizer, max_seq_len=None, mask_prob=0.15):
    pad_id = tokenizer.vocab["[PAD]"]
    mask_id = tokenizer.vocab["[MASK]"]
    cls_id = tokenizer.vocab["[CLS]"]
    sep_id = tokenizer.vocab["[SEP]"]

    # 1. Pad sequences to the longest sequence in this batch.
    if max_seq_len is None:
        max_len = max(len(seq) for seq in batch)
    else:
        max_len = min(max(len(seq) for seq in batch), max_seq_len)

    padded = [
        (seq[:max_len] + [pad_id] * max(0, max_len - len(seq)))
        for seq in batch
    ]

    input_ids = torch.tensor(padded, dtype=torch.long)

    # 2. Initially, ignore every position in the loss.
    labels = torch.full_like(input_ids, -100)

    # 3. Select eligible tokens for masking.
    special_ids = {pad_id, mask_id, cls_id, sep_id}

    for b in range(input_ids.shape[0]):
        eligible_positions = [
            t for t in range(input_ids.shape[1])
            if input_ids[b, t].item() not in special_ids
        ]

        if not eligible_positions:
            continue

        # Select approximately 15%, but at least one token
        # when the sentence contains eligible tokens.
        n_to_mask = max(1, round(len(eligible_positions) * mask_prob))
        n_to_mask = min(n_to_mask, len(eligible_positions))

        selected = random.sample(eligible_positions, n_to_mask)

        for t in selected:
            labels[b, t] = input_ids[b, t]
            input_ids[b, t] = mask_id

    # True means this position is real text, not padding.
    attention_mask = input_ids.ne(pad_id)

    return input_ids, labels, attention_mask