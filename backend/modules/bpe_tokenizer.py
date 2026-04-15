"""
BPETokenizerSimple extracted from:
  doc/source_code/LLMs-from-scratch/ch02/05_bpe-from-scratch/bpe-from-scratch-simple.ipynb

Additions:
- train_step_by_step(): generator that yields merge info after each BPE step
"""
from __future__ import annotations
import re
from collections import defaultdict


class BPETokenizerSimple:
    def __init__(self):
        # Maps token_id -> bytes
        self.vocab: dict[int, bytes] = {}
        # Maps (token_id, token_id) -> merged_token_id
        self.bpe_merges: dict[tuple[int, int], int] = {}

    def train(self, text: str, vocab_size: int, allowed_special: set[str] | None = None):
        """Train BPE from text up to vocab_size."""
        if allowed_special is None:
            allowed_special = {"<|endoftext|>"}
        for step in self.train_step_by_step(text, vocab_size, allowed_special):
            pass  # consume the generator

    def train_step_by_step(self, text: str, vocab_size: int, allowed_special: set[str] | None = None):
        """
        Generator: trains BPE and yields after each merge step.

        Yields dict with:
          step, merged_pair, new_token_id, new_token (str), vocab_size, total_tokens
        """
        if allowed_special is None:
            allowed_special = {"<|endoftext|>"}

        # Step 1: Initialize byte-level vocab (0-255)
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.bpe_merges = {}

        # Step 2: Handle special tokens
        special_tokens = {}
        for token in allowed_special:
            token_id = len(self.vocab)
            self.vocab[token_id] = token.encode("utf-8")
            special_tokens[token] = token_id

        # Step 3: Pre-tokenize text (split on special tokens, keep them)
        if special_tokens:
            pattern = "(" + "|".join(re.escape(t) for t in special_tokens) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]

        # Build initial token IDs list (byte-level)
        token_ids: list[int] = []
        for part in parts:
            if part in special_tokens:
                token_ids.append(special_tokens[part])
            else:
                token_ids.extend(list(part.encode("utf-8")))

        initial_count = len(token_ids)

        # Yield initial state (Step 0)
        yield {
            "step": 0,
            "merged_pair": None,
            "merged_pair_str": None,
            "new_token_id": None,
            "new_token": None,
            "current_vocab_size": len(self.vocab),
            "total_tokens": len(token_ids),
            "compression_ratio": 1.0,
            "tokens_sample": [
                {"id": tid, "s": self._decode_token(tid)}
                for tid in token_ids[:100]
            ]
        }

        # Step 4: Merge loop
        num_merges = vocab_size - len(self.vocab)
        for step_idx in range(num_merges):
            pair = self._find_freq_pair(token_ids)
            if pair is None:
                break

            new_id = len(self.vocab)
            self.bpe_merges[pair] = new_id
            merged_bytes = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.vocab[new_id] = merged_bytes
            token_ids = self._replace_pair(token_ids, pair, new_id)

            yield {
                "step": step_idx + 1,
                "merged_pair": pair,
                "merged_pair_str": (
                    self._decode_token(pair[0]),
                    self._decode_token(pair[1]),
                ),
                "new_token_id": new_id,
                "new_token": self._decode_token(new_id),
                "current_vocab_size": len(self.vocab),
                "total_tokens": len(token_ids),
                "compression_ratio": round(initial_count / max(len(token_ids), 1), 3),
                "tokens_sample": [
                    {"id": tid, "s": self._decode_token(tid)}
                    for tid in token_ids[:100]
                ]
            }

    def _find_freq_pair(self, token_ids: list[int]) -> tuple[int, int] | None:
        """Find the most frequent adjacent pair."""
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for a, b in zip(token_ids, token_ids[1:]):
            counts[(a, b)] += 1
        if not counts:
            return None
        return max(counts, key=lambda p: counts[p])

    def _replace_pair(self, token_ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace all occurrences of pair with new_id."""
        result: list[int] = []
        i = 0
        while i < len(token_ids):
            if i < len(token_ids) - 1 and token_ids[i] == pair[0] and token_ids[i + 1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(token_ids[i])
                i += 1
        return result

    def _decode_token(self, token_id: int) -> str:
        try:
            return self.vocab[token_id].decode("utf-8", errors="replace")
        except Exception:
            return f"<{token_id}>"

    def encode(self, text: str) -> list[int]:
        """Encode text using trained BPE."""
        token_ids = list(text.encode("utf-8"))
        # Apply merges in order
        for pair, new_id in self.bpe_merges.items():
            token_ids = self._replace_pair(token_ids, pair, new_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        byte_string = b"".join(self.vocab.get(t, b"") for t in token_ids)
        return byte_string.decode("utf-8", errors="replace")

    def get_vocab_list(self, limit: int = 100) -> list[dict]:
        """Return a list of vocab entries for display."""
        result = []
        for token_id, token_bytes in list(self.vocab.items())[:limit]:
            result.append({
                "id": token_id,
                "token": token_bytes.decode("utf-8", errors="replace"),
                "bytes": list(token_bytes),
            })
        return result
