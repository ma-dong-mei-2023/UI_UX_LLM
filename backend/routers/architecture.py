from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from config import GPT_PRESETS
from modules.architecture_analyzer import (
    count_params, estimate_memory_mb, get_architecture_layers, validate_config
)

router = APIRouter(prefix="/api/architecture", tags=["architecture"])


class ConfigRequest(BaseModel):
    vocab_size: int = 50257
    context_length: int = 256
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = True


@router.post("/build")
def build_model_info(cfg: ConfigRequest):
    config = cfg.model_dump()
    errors = validate_config(config)
    if errors:
        raise HTTPException(400, "; ".join(errors))

    return {
        "config": config,
        "params": count_params(config),
        "memory": estimate_memory_mb(config),
        "layers": get_architecture_layers(config),
        "errors": [],
    }


@router.get("/presets")
def list_presets():
    result = {}
    for name, cfg in GPT_PRESETS.items():
        params = count_params(cfg)
        result[name] = {
            "config": cfg,
            "total_params_M": params["total_M"],
        }
    return result


@router.post("/analyze")
def analyze_config(cfg: ConfigRequest):
    config = cfg.model_dump()
    errors = validate_config(config)
    params = count_params(config)
    memory = estimate_memory_mb(config)
    return {
        "config": config,
        "params": params,
        "memory": memory,
        "errors": errors,
    }


@router.post("/validate")
def validate(cfg: ConfigRequest):
    errors = validate_config(cfg.model_dump())
    return {"valid": len(errors) == 0, "errors": errors}
