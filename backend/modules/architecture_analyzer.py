"""
Analyzes GPT model architecture: parameter counts, memory estimates, structure.
"""
from __future__ import annotations
import sys
from config import PKG_DIR

if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))


def count_params(cfg: dict) -> dict:
    """Count parameters for each component of a GPT model."""
    V = cfg["vocab_size"]
    C = cfg["context_length"]
    E = cfg["emb_dim"]
    H = cfg["n_heads"]
    L = cfg["n_layers"]
    head_dim = E // H

    # Embeddings
    token_emb = V * E
    pos_emb = C * E

    # Per transformer block
    # Attention: Q, K, V projections + output + optional bias
    qkv_bias = cfg.get("qkv_bias", False)
    attn_qkv = 3 * (E * E + (E if qkv_bias else 0))
    attn_out = E * E + E
    attn_total = attn_qkv + attn_out

    # LayerNorm (scale + shift)
    ln_params = 2 * E

    # FeedForward: linear1 + gelu + linear2
    ff_params = E * (4 * E) + (4 * E) + (4 * E) * E + E

    # Residual connections: no params
    block_params = attn_total + ff_params + 2 * ln_params

    # Final LayerNorm
    final_ln = 2 * E

    # Output head (tied with token embedding typically, but as standalone)
    out_head = E * V

    total_blocks = block_params * L
    total = token_emb + pos_emb + total_blocks + final_ln + out_head

    return {
        "token_embedding": token_emb,
        "position_embedding": pos_emb,
        "per_block": {
            "attention_qkv": attn_qkv,
            "attention_out": attn_out,
            "feedforward": ff_params,
            "layer_norm_1": ln_params,
            "layer_norm_2": ln_params,
            "subtotal": block_params,
        },
        "num_blocks": L,
        "all_blocks": total_blocks,
        "final_layer_norm": final_ln,
        "output_head": out_head,
        "total": total,
        "total_M": round(total / 1e6, 2),
    }


def estimate_memory_mb(cfg: dict, dtype_bytes: int = 4) -> dict:
    """Estimate GPU memory in MB for training (model + gradients + optimizer states)."""
    params = count_params(cfg)["total"]

    # Parameters (fp32 = 4 bytes)
    param_mem = params * dtype_bytes

    # Gradients (same size as params)
    grad_mem = params * dtype_bytes

    # Adam optimizer states: m + v (2 * params, fp32)
    optim_mem = 2 * params * 4

    # Activations (rough estimate: batch * seq * emb * layers * 4)
    # Using batch=1 for minimum estimate
    batch = 1
    act_mem = batch * cfg["context_length"] * cfg["emb_dim"] * cfg["n_layers"] * 12 * dtype_bytes

    total = param_mem + grad_mem + optim_mem + act_mem

    return {
        "parameters_mb": round(param_mem / 1e6, 1),
        "gradients_mb": round(grad_mem / 1e6, 1),
        "optimizer_states_mb": round(optim_mem / 1e6, 1),
        "activations_mb": round(act_mem / 1e6, 1),
        "total_training_mb": round(total / 1e6, 1),
        "inference_only_mb": round((param_mem + act_mem) / 1e6, 1),
    }


def get_architecture_layers(cfg: dict) -> list[dict]:
    """Return a structured description of all layers for visualization."""
    E = cfg["emb_dim"]
    H = cfg["n_heads"]
    L = cfg["n_layers"]
    V = cfg["vocab_size"]
    C = cfg["context_length"]

    layers = [
        {"name": "TokenEmbedding", "type": "embedding", "shape": f"[{V}, {E}]", "level": 0},
        {"name": "PositionEmbedding", "type": "embedding", "shape": f"[{C}, {E}]", "level": 0},
        {"name": "Dropout", "type": "dropout", "shape": f"[*, {C}, {E}]", "level": 0},
    ]

    for i in range(L):
        block = {
            "name": f"TransformerBlock[{i}]",
            "type": "block",
            "level": 1,
            "children": [
                {"name": "LayerNorm1", "type": "norm", "shape": f"[*, {C}, {E}]"},
                {"name": f"MultiHeadAttention (H={H})", "type": "attention", "shape": f"[*, {C}, {E}]",
                 "children": [
                     {"name": f"W_query [{E}→{E}]", "type": "linear"},
                     {"name": f"W_key   [{E}→{E}]", "type": "linear"},
                     {"name": f"W_value [{E}→{E}]", "type": "linear"},
                     {"name": f"out_proj [{E}→{E}]", "type": "linear"},
                 ]},
                {"name": "Residual +", "type": "residual"},
                {"name": "LayerNorm2", "type": "norm", "shape": f"[*, {C}, {E}]"},
                {"name": f"FeedForward [{E}→{4*E}→{E}]", "type": "ff",
                 "children": [
                     {"name": f"Linear [{E}→{4*E}]", "type": "linear"},
                     {"name": "GELU", "type": "activation"},
                     {"name": f"Linear [{4*E}→{E}]", "type": "linear"},
                 ]},
                {"name": "Residual +", "type": "residual"},
            ],
        }
        layers.append(block)

    layers += [
        {"name": "FinalLayerNorm", "type": "norm", "shape": f"[*, {C}, {E}]", "level": 0},
        {"name": f"OutputHead [{E}→{V}]", "type": "linear", "shape": f"[*, {C}, {V}]", "level": 0},
    ]

    return layers


def validate_config(cfg: dict) -> list[str]:
    """Validate model config, return list of error messages (empty = valid)."""
    errors = []
    E = cfg.get("emb_dim", 0)
    H = cfg.get("n_heads", 0)
    if H > 0 and E % H != 0:
        errors.append(f"emb_dim ({E}) must be divisible by n_heads ({H})")
    if cfg.get("n_layers", 0) < 1:
        errors.append("n_layers must be at least 1")
    if cfg.get("context_length", 0) < 1:
        errors.append("context_length must be at least 1")
    if cfg.get("vocab_size", 0) < 1:
        errors.append("vocab_size must be at least 1")
    return errors
