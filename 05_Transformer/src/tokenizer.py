

import re
from collections import Counter


class WordPieceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.ids_to_tokens = {
            token_id: token
            for token, token_id in vocab.items()
        }

        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"

    @classmethod
    def from_corpus(
        cls,
        corpus,
        vocab_size=500,
        min_freq=1,
        max_piece_len=10,
    ):
        # Split text into words and punctuation.
        words = re.findall(r"\w+|[^\w\s]", corpus.lower())

        # Count possible WordPiece candidates.
        counts = Counter()

        for word in words:
            for start in range(len(word)):
                for end in range(
                    start + 1,
                    min(len(word), start + max_piece_len) + 1,
                ):
                    piece = word[start:end]

                    if start == 0:
                        counts[piece] += 1
                    else:
                        counts["##" + piece] += 1

        special_tokens = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
        ]

        vocab = {
            token: i
            for i, token in enumerate(special_tokens)
        }

        # Keep single characters so unseen words can often
        # be split into smaller pieces.
        for word in words:
            if word:
                for i, char in enumerate(word):
                    piece = char if i == 0 else "##" + char
                    if piece not in vocab:
                        vocab[piece] = len(vocab)

        # Add frequent pieces until the vocabulary is full.
        candidates = [
            (piece, count)
            for piece, count in counts.items()
            if count >= min_freq and piece not in vocab
        ]
        candidates.sort(key=lambda item: item[1], reverse=True)

        for piece, _ in candidates:
            if len(vocab) >= vocab_size:
                break
            vocab[piece] = len(vocab)

        return cls(vocab)

    def tokenize_word(self, word):
        pieces = []
        start = 0

        while start < len(word):
            end = len(word)
            found = None

            # Try the longest possible piece first.
            while end > start:
                piece = word[start:end]

                if start > 0:
                    piece = "##" + piece

                if piece in self.vocab:
                    found = piece
                    break

                end -= 1

            # If any part cannot be tokenized, use [UNK]
            # for the entire word.
            if found is None:
                return [self.unk_token]

            pieces.append(found)
            start = end

        return pieces

    def tokenize(self, text):
        words = re.findall(r"\w+|[^\w\s]", text.lower())
        pieces = []

        for word in words:
            pieces.extend(self.tokenize_word(word))

        return pieces

    def encode(self, text, add_special_tokens=True):
        tokens = self.tokenize(text)

        if add_special_tokens:
            tokens = [self.cls_token] + tokens + [self.sep_token]

        return [
            self.vocab.get(token, self.vocab[self.unk_token])
            for token in tokens
        ]

    def decode(self, token_ids):
        tokens = [
            self.ids_to_tokens.get(i, self.unk_token)
            for i in token_ids
        ]

        words = []

        for token in tokens:
            if token in {
                self.pad_token,
                self.cls_token,
                self.sep_token,
                self.mask_token,
            }:
                continue

            if token == self.unk_token:
                words.append(token)
            elif token.startswith("##"):
                if words:
                    words[-1] += token[2:]
                else:
                    words.append(token[2:])
            elif re.fullmatch(r"[^\w\s]", token):
                if words:
                    words[-1] += token
                else:
                    words.append(token)
            else:
                words.append(token)

        return " ".join(words)
