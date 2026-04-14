from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from modules.attention_viz import (
    run_simple_attention, run_causal_attention, run_multihead_attention
)

router = APIRouter(prefix="/api/attention", tags=["attention"])


class AttentionRequest(BaseModel):
    tokens: list[str]
    d_model: int = 64
    n_heads: int = 4


@router.post("/simple")
def simple_attention(req: AttentionRequest):
    if len(req.tokens) < 2:
        raise HTTPException(400, "Need at least 2 tokens")
    if len(req.tokens) > 32:
        raise HTTPException(400, "Maximum 32 tokens")
    return run_simple_attention(req.tokens, d_in=req.d_model, d_out=req.d_model)


@router.post("/causal")
def causal_attention(req: AttentionRequest):
    if len(req.tokens) < 2:
        raise HTTPException(400, "Need at least 2 tokens")
    if len(req.tokens) > 32:
        raise HTTPException(400, "Maximum 32 tokens")
    return run_causal_attention(req.tokens, d_in=req.d_model, d_out=req.d_model)


@router.post("/multihead")
def multihead_attention(req: AttentionRequest):
    if len(req.tokens) < 2:
        raise HTTPException(400, "Need at least 2 tokens")
    if len(req.tokens) > 32:
        raise HTTPException(400, "Maximum 32 tokens")
    if req.d_model % req.n_heads != 0:
        raise HTTPException(400, f"d_model ({req.d_model}) must be divisible by n_heads ({req.n_heads})")
    return run_multihead_attention(req.tokens, d_model=req.d_model, n_heads=req.n_heads)


@router.get("/configs")
def attention_configs():
    return {
        "presets": [
            {"name": "Small", "d_model": 32, "n_heads": 2},
            {"name": "Medium", "d_model": 64, "n_heads": 4},
            {"name": "Large", "d_model": 128, "n_heads": 8},
        ]
    }
