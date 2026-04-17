"""
Attention weight capture: subclass of MultiHeadAttention that stores weights.
"""
from __future__ import annotations
import sys
import torch
import torch.nn as nn
from config import PKG_DIR

if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from llms_from_scratch.ch03 import MultiHeadAttention


class MultiHeadAttentionWithWeights(MultiHeadAttention):
    """MultiHeadAttention that saves intermediate states after forward pass."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_attn_weights = None
        self._last_q = None
        self._last_k = None
        self._last_raw_scores = None
        self._last_scaled_scores = None
        self._last_masked_scores = None

    def forward(self, x: torch.Tensor, mode: str = "random") -> torch.Tensor:
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Implementation of toy modes for teaching
        if mode == "toy_previous":
            # Head 0: attends to previous token
            # Head 1: attends to first token
            # Others: random
            queries = torch.zeros_like(queries)
            keys = torch.zeros_like(keys)
            for i in range(num_tokens):
                queries[0, 0, i, 0] = 1.0
                if i > 0:
                    keys[0, 0, i - 1, 0] = 1.0
                if self.num_heads > 1:
                    queries[0, 1, i, 0] = 1.0
                    keys[0, 1, 0, 0] = 1.0

        attn_scores = queries @ keys.transpose(2, 3)
        self._last_raw_scores = attn_scores.detach().cpu()

        scaled_scores = attn_scores / keys.shape[-1] ** 0.5
        self._last_scaled_scores = scaled_scores.detach().cpu()

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # Use a large negative number instead of -inf for JSON compliance in last_masked_scores
        masked_scores = scaled_scores.masked_fill(mask_bool, -10000.0)
        self._last_masked_scores = masked_scores.detach().cpu()

        # For the actual softmax, we can still use -inf for precision
        attn_weights = torch.softmax(scaled_scores.masked_fill(mask_bool, -torch.inf), dim=-1)
        self._last_attn_weights = attn_weights.detach().cpu()
        self._last_q = queries.detach().cpu()
        self._last_k = keys.detach().cpu()

        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


def swap_attention_modules(model: nn.Module) -> nn.Module:
    """Replace all MultiHeadAttention modules with MultiHeadAttentionWithWeights."""
    for name, module in model.named_children():
        if type(module).__name__ == "MultiHeadAttention":
            # Create replacement with same config
            new_module = MultiHeadAttentionWithWeights(
                d_in=module.d_in if hasattr(module, "d_in") else module.W_query.in_features,
                d_out=module.d_out,
                context_length=module.mask.shape[0],
                dropout=0.0,  # No dropout during visualization
                num_heads=module.num_heads,
                qkv_bias=module.W_query.bias is not None,
            )
            new_module.load_state_dict(module.state_dict())
            new_module.eval()
            setattr(model, name, new_module)
        else:
            swap_attention_modules(module)
    return model


def extract_attention_weights(model: nn.Module) -> list[dict]:
    """Extract stored attention weights from all MultiHeadAttentionWithWeights modules."""
    results = []
    for name, module in model.named_modules():
        if isinstance(module, MultiHeadAttentionWithWeights) and module._last_attn_weights is not None:
            weights = module._last_attn_weights  # shape: [batch, heads, seq, seq]
            results.append({
                "layer_name": name,
                "num_heads": weights.shape[1],
                "seq_len": weights.shape[2],
                # Return weights as nested list [heads, seq, seq]
                "weights": weights[0].tolist(),
            })
    return results


def run_simple_attention(tokens: list[str], d_in: int = 16, d_out: int = 16, mode: str = "random") -> dict:
    """Run SelfAttention_v1 on tokens, return attention weights for visualization."""
    from llms_from_scratch.ch03 import SelfAttention_v1
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    token_ids = [enc.encode(t)[0] if t else 0 for t in tokens]
    seq_len = len(token_ids)

    # Simple embedding lookup
    vocab_size = 50257
    emb = nn.Embedding(vocab_size, d_in)
    x = emb(torch.tensor(token_ids)).unsqueeze(0)  # [1, seq, d_in]

    attn = SelfAttention_v1(d_in, d_out)
    attn.eval()

    with torch.no_grad():
        keys = x.squeeze(0) @ attn.W_key
        queries = x.squeeze(0) @ attn.W_query

        if mode == "toy_previous":
            queries = torch.zeros_like(queries)
            keys = torch.zeros_like(keys)
            for i in range(seq_len):
                queries[i, 0] = 1.0
                if i > 0:
                    keys[i - 1, 0] = 1.0

        raw_scores = queries @ keys.T
        scaled_scores = raw_scores / keys.shape[-1] ** 0.5
        attn_weights = torch.softmax(scaled_scores, dim=-1)

    return {
        "tokens": tokens,
        "q": queries.detach().tolist(),
        "k": keys.detach().tolist(),
        "raw_scores": raw_scores.detach().tolist(),
        "scaled_scores": scaled_scores.detach().tolist(),
        "weights": attn_weights.detach().tolist(),
        "type": "self_attention_v1",
        "mode": mode,
    }


def run_causal_attention(tokens: list[str], d_in: int = 16, d_out: int = 16, mode: str = "random") -> dict:
    """Run CausalAttention (masked) and return weights."""
    from llms_from_scratch.ch03 import CausalAttention
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    token_ids = [enc.encode(t)[0] if t else 0 for t in tokens]
    seq_len = len(token_ids)

    emb = nn.Embedding(50257, d_in)
    x = emb(torch.tensor(token_ids)).unsqueeze(0)

    attn = CausalAttention(d_in, d_out, context_length=seq_len, dropout=0.0)
    attn.eval()

    with torch.no_grad():
        keys = attn.W_key(x)
        queries = attn.W_query(x)

        if mode == "toy_previous":
            queries = torch.zeros_like(queries)
            keys = torch.zeros_like(keys)
            for i in range(seq_len):
                queries[0, i, 0] = 1.0
                if i > 0:
                    keys[0, i - 1, 0] = 1.0

        raw_scores = queries @ keys.transpose(1, 2)
        scaled_scores = raw_scores / keys.shape[-1] ** 0.5
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        # Use -10000.0 for JSON display and float("-inf") for actual calculation
        masked_scores = scaled_scores.masked_fill(mask, -10000.0)
        attn_weights = torch.softmax(scaled_scores.masked_fill(mask, float("-inf")), dim=-1)

    return {
        "tokens": tokens,
        "q": queries[0].detach().tolist(),
        "k": keys[0].detach().tolist(),
        "raw_scores": raw_scores[0].detach().tolist(),
        "scaled_scores": scaled_scores[0].detach().tolist(),
        "masked_scores": masked_scores[0].detach().tolist(),
        "weights": attn_weights[0].detach().tolist(),
        "type": "causal_attention",
        "is_masked": True,
        "mode": mode,
    }


def run_multihead_attention(tokens: list[str], d_model: int = 64, n_heads: int = 4, mode: str = "random") -> dict:
    """Run MultiHeadAttention with weight capture."""
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    token_ids = [enc.encode(t)[0] if t else 0 for t in tokens]
    seq_len = len(token_ids)

    emb = nn.Embedding(50257, d_model)
    x = emb(torch.tensor(token_ids)).unsqueeze(0)

    attn = MultiHeadAttentionWithWeights(
        d_in=d_model, d_out=d_model, context_length=seq_len,
        dropout=0.0, num_heads=n_heads, qkv_bias=False
    )
    attn.eval()

    with torch.no_grad():
        attn(x, mode=mode)

    return {
        "tokens": tokens,
        "q": attn._last_q[0].tolist(),
        "k": attn._last_k[0].tolist(),
        "raw_scores": attn._last_raw_scores[0].tolist(),
        "scaled_scores": attn._last_scaled_scores[0].tolist(),
        "masked_scores": attn._last_masked_scores[0].tolist(),
        "weights": attn._last_attn_weights[0].tolist(),
        "num_heads": n_heads,
        "type": "multihead_attention",
        "mode": mode,
        "is_masked": True,
    }
