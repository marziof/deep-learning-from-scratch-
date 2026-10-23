
import random
import torch
import torch.nn.functional as F

from src.dataset import prepare_batch


def train(model, mlm_head, optimizer, encoded_txt, tokenizer,
          batch_size):
    model.train()
    mlm_head.train()

    random.shuffle(encoded_txt)

    total_loss = 0.0
    total_masked = 0

    for i in range(0, len(encoded_txt), batch_size):
        batch = encoded_txt[i:i + batch_size]

        input_ids, labels, attention_mask = prepare_batch(
            batch, tokenizer
        )

        # Forward pass
        hidden_states = hidden_states = model(
            input_ids,
            attention_mask=attention_mask,
        )
        logits = mlm_head(hidden_states)

        num_masked = (labels != -100).sum().item()
        if num_masked == 0:
            continue

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * num_masked
        total_masked += num_masked

    return total_loss / max(total_masked, 1)


def evaluate(model, mlm_head, encoded_txt, tokenizer, batch_size):
    model.eval()
    mlm_head.eval()

    total_loss = 0.0
    total_masked = 0

    # Do not calculate gradients during evaluation.
    with torch.no_grad():
        for i in range(0, len(encoded_txt), batch_size):
            batch = encoded_txt[i:i + batch_size]

            input_ids, labels, attention_mask = prepare_batch(
                batch, tokenizer
            )

            hidden_states = model(
                input_ids,
                attention_mask=attention_mask,
            )
            logits = mlm_head(hidden_states)

            num_masked = (labels != -100).sum().item()
            if num_masked == 0:
                continue

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )

            total_loss += loss.item() * num_masked
            total_masked += num_masked

    return total_loss / max(total_masked, 1)
