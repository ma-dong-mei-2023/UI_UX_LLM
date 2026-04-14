"""
Single training run manager: enforces only one active training/fine-tuning run at a time.
"""
from __future__ import annotations
import threading
import queue
import uuid
from dataclasses import dataclass, field
from typing import Optional
import torch.nn as nn
import torch


@dataclass
class TrainingRun:
    run_id: str
    module: str  # "training" | "classification" | "chat_finetune"
    config: dict
    status: str = "pending"  # pending | running | completed | cancelled | error
    cancel_flag: threading.Event = field(default_factory=threading.Event)
    metrics_queue: queue.Queue = field(default_factory=queue.Queue)
    thread: Optional[threading.Thread] = None
    accumulated_metrics: list = field(default_factory=list)
    model: Optional[nn.Module] = None
    error_message: Optional[str] = None


# Global active run registry
_lock = threading.Lock()
_active_run: Optional[TrainingRun] = None


def create_run(module: str, config: dict) -> TrainingRun:
    run = TrainingRun(
        run_id=str(uuid.uuid4())[:8],
        module=module,
        config=config,
    )
    return run


def register_run(run: TrainingRun) -> bool:
    """Register run as active. Returns False if another run is active."""
    global _active_run
    with _lock:
        if _active_run is not None and _active_run.status == "running":
            return False
        _active_run = run
        return True


def get_active_run() -> Optional[TrainingRun]:
    return _active_run


def get_run_status(run_id: str) -> Optional[dict]:
    global _active_run
    if _active_run and _active_run.run_id == run_id:
        return {
            "run_id": run_id,
            "status": _active_run.status,
            "module": _active_run.module,
            "metrics": list(_active_run.accumulated_metrics),
            "error": _active_run.error_message,
        }
    return None


def cancel_run(run_id: str) -> bool:
    global _active_run
    if _active_run and _active_run.run_id == run_id:
        _active_run.cancel_flag.set()
        return True
    return False
