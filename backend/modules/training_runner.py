"""
Callback-based training runner for GPT pretraining (Chapter 5).
Reimplements train_model_simple loop with metrics pushed to a queue.
"""
from __future__ import annotations
import sys
import threading
import torch
import torch.nn as nn
from pathlib import Path
from config import PKG_DIR, SOURCE_DIR, MODELS_DIR

if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from llms_from_scratch.ch04 import GPTModel
from llms_from_scratch.ch05 import (
    generate, text_to_token_ids, token_ids_to_text,
    calc_loss_batch, calc_loss_loader, evaluate_model,
)
from llms_from_scratch.ch02 import create_dataloader_v1
from modules.run_manager import TrainingRun


def train_gpt(run: TrainingRun, device: torch.device):
    """Main training function, runs in a background thread."""
    try:
        run.status = "running"
        cfg = run.config
        model_cfg = cfg["model"]
        train_cfg = cfg["training"]

        # Build model
        model = GPTModel(model_cfg)
        model.to(device)
        run.model = model

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.get("learning_rate", 5e-4),
            weight_decay=train_cfg.get("weight_decay", 0.1),
        )

        # Load training text
        text_path = train_cfg.get("text_path", str(SOURCE_DIR / "ch02" / "01_main-chapter-code" / "the-verdict.txt"))
        with open(text_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Split 90/10
        split = int(len(raw_text) * 0.9)
        train_text = raw_text[:split]
        val_text = raw_text[split:]

        context_length = model_cfg["context_length"]
        batch_size = train_cfg.get("batch_size", 2)
        stride = context_length // 2

        train_loader = create_dataloader_v1(
            train_text, batch_size=batch_size, max_length=context_length, stride=stride, shuffle=True
        )
        val_loader = create_dataloader_v1(
            val_text, batch_size=batch_size, max_length=context_length, stride=stride, shuffle=False
        )

        import tiktoken
        tokenizer = tiktoken.get_encoding("gpt2")
        start_context = train_cfg.get("start_context", "Every effort moves")

        num_epochs = train_cfg.get("num_epochs", 5)
        eval_freq = train_cfg.get("eval_freq", 5)
        eval_iter = train_cfg.get("eval_iter", 1)
        use_warmup = train_cfg.get("use_warmup", False)
        grad_clip = train_cfg.get("grad_clip", 1.0)

        # LR warmup setup
        warmup_steps = 0
        if use_warmup:
            warmup_steps = int(len(train_loader) * num_epochs * 0.1)
            peak_lr = train_cfg.get("learning_rate", 5e-4)
            min_lr = peak_lr * 0.1
            total_steps = len(train_loader) * num_epochs

        tokens_seen = 0
        global_step = -1

        for epoch in range(num_epochs):
            model.train()
            for input_batch, target_batch in train_loader:
                if run.cancel_flag.is_set():
                    run.status = "cancelled"
                    run.metrics_queue.put({"type": "cancelled"})
                    return

                optimizer.zero_grad()

                # Learning rate schedule
                if use_warmup and global_step >= 0:
                    if global_step < warmup_steps:
                        lr = peak_lr * global_step / warmup_steps
                    else:
                        progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
                        import math
                        lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr

                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)

                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()

                if grad_clip > 0 and global_step >= 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()
                tokens_seen += input_batch.numel()
                global_step += 1

                if global_step % eval_freq == 0:
                    model.eval()
                    with torch.no_grad():
                        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
                        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
                    model.train()

                    metric = {
                        "type": "eval",
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "train_loss": round(train_loss, 4),
                        "val_loss": round(val_loss, 4),
                        "tokens_seen": tokens_seen,
                        "learning_rate": round(optimizer.param_groups[0]["lr"], 6),
                    }
                    run.accumulated_metrics.append(metric)
                    run.metrics_queue.put(metric)

            # Generate sample at end of epoch
            model.eval()
            with torch.no_grad():
                sample_ids = generate(
                    model=model,
                    idx=text_to_token_ids(start_context, tokenizer).to(device),
                    max_new_tokens=50,
                    context_size=context_length,
                    temperature=1.0,
                    top_k=25,
                )
            sample_text = token_ids_to_text(sample_ids, tokenizer)

            epoch_metric = {
                "type": "epoch_end",
                "epoch": epoch + 1,
                "sample_text": sample_text,
            }
            run.accumulated_metrics.append(epoch_metric)
            run.metrics_queue.put(epoch_metric)

        # Save model
        save_path = MODELS_DIR / f"gpt_{run.run_id}.pth"
        torch.save(model.state_dict(), str(save_path))

        run.status = "completed"
        run.metrics_queue.put({"type": "completed", "model_path": str(save_path)})

    except Exception as e:
        run.status = "error"
        run.error_message = str(e)
        run.metrics_queue.put({"type": "error", "message": str(e)})


def start_training_thread(run: TrainingRun) -> threading.Thread:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = threading.Thread(target=train_gpt, args=(run, device), daemon=True)
    run.thread = t
    t.start()
    return t
