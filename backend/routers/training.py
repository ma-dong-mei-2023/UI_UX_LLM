from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from pathlib import Path
from config import GPT_PRESETS, MODELS_DIR, UPLOADS_DIR
from modules.run_manager import create_run, register_run, get_run_status, cancel_run, get_active_run
from modules.training_runner import start_training_thread
import database
import shutil

router = APIRouter(prefix="/api/training", tags=["training"])


class TrainingConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_preset: str = "tiny"
    custom_model_config: dict | None = None  # Override preset if provided
    learning_rate: float = 5e-4
    num_epochs: int = 5
    batch_size: int = 2
    eval_freq: int = 5
    eval_iter: int = 1
    use_warmup: bool = False
    grad_clip: float = 1.0
    start_context: str = "Every effort moves"
    text_path: str | None = None  # None = use default verdict text


class SaveConfigRequest(BaseModel):
    name: str
    config: dict


@router.post("/start")
def start_training(cfg: TrainingConfig):
    active = get_active_run()
    if active and active.status == "running":
        raise HTTPException(409, f"Training run {active.run_id} is already active. Cancel it first.")

    # Resolve model config
    if cfg.custom_model_config:
        model_cfg = cfg.custom_model_config
    elif cfg.model_preset in GPT_PRESETS:
        model_cfg = GPT_PRESETS[cfg.model_preset]
    else:
        raise HTTPException(400, f"Unknown preset '{cfg.model_preset}'. Valid: {list(GPT_PRESETS.keys())}")

    training_cfg = {
        "learning_rate": cfg.learning_rate,
        "num_epochs": cfg.num_epochs,
        "batch_size": cfg.batch_size,
        "eval_freq": cfg.eval_freq,
        "eval_iter": cfg.eval_iter,
        "use_warmup": cfg.use_warmup,
        "grad_clip": cfg.grad_clip,
        "start_context": cfg.start_context,
        "text_path": cfg.text_path,
    }

    config = {"model": model_cfg, "training": training_cfg}
    run = create_run("training", config)

    if not register_run(run):
        raise HTTPException(409, "Could not register run")

    database.save_run(run.run_id, "training", config, "pending")
    start_training_thread(run)
    database.update_run_status(run.run_id, "running")

    return {"run_id": run.run_id, "status": "started", "model_config": model_cfg}


@router.post("/cancel/{run_id}")
def cancel_training(run_id: str):
    if cancel_run(run_id):
        return {"cancelled": True}
    raise HTTPException(404, "Run not found or not active")


@router.get("/status/{run_id}")
def get_status(run_id: str):
    status = get_run_status(run_id)
    if status is None:
        raise HTTPException(404, "Run not found")
    return status


@router.get("/presets")
def list_presets():
    return {"presets": list(GPT_PRESETS.keys()), "configs": GPT_PRESETS}


@router.get("/history")
def list_history():
    return database.list_runs("training")


@router.post("/configs/save")
def save_config(req: SaveConfigRequest):
    config_id = database.save_config(req.name, "training", req.config)
    return {"id": config_id, "saved": True}


@router.get("/configs")
def list_configs():
    return database.list_configs("training")


@router.delete("/configs/{config_id}")
def delete_config(config_id: int):
    deleted = database.delete_config(config_id)
    if not deleted:
        raise HTTPException(404, "Config not found")
    return {"deleted": True}


@router.get("/model/{run_id}/download")
def download_model(run_id: str):
    model_path = MODELS_DIR / f"gpt_{run_id}.pth"
    if not model_path.exists():
        raise HTTPException(404, "Model file not found")
    return FileResponse(str(model_path), filename=f"gpt_{run_id}.pth")
