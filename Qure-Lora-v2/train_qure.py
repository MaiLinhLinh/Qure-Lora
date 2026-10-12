"""Train the QuRe objective with the shared BLIP2 Q-Former and two LoRA adapters."""

import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import create_dataloaders
from evaluate_qure.eval_utils import eval_cirr, eval_fiq
from lora.checkpoint import load_checkpoint, save_checkpoint
from models import create_qure_models
from options import get_experiment_config
from set_up import setup_experiment
from transforms import image_transform_factory


def _use_wandb(configs):
    return bool(configs.get("use_wandb", configs.get("experiment_description") != "debug"))


def _atomic_save_checkpoint(path, **save_kwargs):
    """Save one checkpoint without destroying the previous valid copy."""
    path = Path(path)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        save_checkpoint(temporary_path, **save_kwargs)
        os.replace(temporary_path, path)
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _device_from_config(configs):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    device_idx = str(configs.get("device_idx", "0")).split(",")[0].strip()
    return torch.device(f"cuda:{device_idx}")


def _make_epoch_scheduler(optimizer, init_lr, min_lr, max_epochs):
    if init_lr <= 0:
        raise ValueError("init_lr must be positive")
    min_ratio = min_lr / init_lr

    def cosine_factor(epoch):
        progress = min(max(epoch, 0) / max(max_epochs, 1), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_factor)


def _make_warmup_cosine_scheduler(
    optimizer, total_steps, warmup_steps, min_lr, reference_lr
):
    """Create a per-update linear-warmup + cosine-decay scheduler.

    ``min_lr`` is interpreted relative to the QuRe/task learning rate.  The
    same ratio is applied to every optimizer group, so a LoRA group starting
    at 5e-4 decays to 5e-6 when the task group starts at 5e-6 and
    ``min_lr`` is 5e-8.
    """
    total_steps = int(total_steps)
    warmup_steps = int(warmup_steps)
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if reference_lr <= 0:
        raise ValueError("reference_lr must be positive")
    if min_lr < 0 or min_lr > reference_lr:
        raise ValueError("min_lr must be in the range [0, reference_lr]")

    warmup_steps = min(warmup_steps, total_steps)
    min_ratio = min_lr / reference_lr

    def warmup_cosine_factor(step):
        # LambdaLR evaluates step=0 during construction. Returning 1/N here
        # makes the first optimizer update use the first warmup learning rate.
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / warmup_steps

        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min(
            max(step - warmup_steps, 0) / decay_steps,
            1.0,
        )
        return min_ratio + (1.0 - min_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[warmup_cosine_factor] * len(optimizer.param_groups),
    )


def _build_optimizer(model, configs):
    """Build the baseline optimizer or the two-group LoRA optimizer."""
    trainable_named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_named_parameters:
        raise RuntimeError("No trainable parameters were found")

    if not configs.get("use_lora", False):
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in trainable_named_parameters],
            lr=configs["init_lr"],
            weight_decay=configs["weight_decay"],
        )
        return optimizer, trainable_named_parameters

    lora_parameters = [
        parameter
        for name, parameter in trainable_named_parameters
        if ".lora_A." in name or ".lora_B." in name
    ]
    task_parameters = [
        parameter
        for name, parameter in trainable_named_parameters
        if ".lora_A." not in name and ".lora_B." not in name
    ]
    if not lora_parameters:
        raise RuntimeError(
            "LoRA is enabled, but no trainable LoRA parameters were found"
        )

    lora_lr = float(configs.get("lora_lr", 5e-4))
    task_lr = float(configs["init_lr"])
    if lora_lr <= 0 or task_lr <= 0:
        raise ValueError("lora_lr and init_lr must be positive")

    parameter_groups = [
        {
            "params": lora_parameters,
            "lr": lora_lr,
            "group_name": "lora",
        }
    ]
    if task_parameters:
        parameter_groups.append(
            {
                "params": task_parameters,
                "lr": task_lr,
                "group_name": "task",
            }
        )

    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=configs["weight_decay"],
    )
    lora_parameter_count = sum(parameter.numel() for parameter in lora_parameters)
    task_parameter_count = sum(parameter.numel() for parameter in task_parameters)
    print(
        f"Optimizer groups: LoRA={lora_parameter_count:,} parameters "
        f"(lr={lora_lr:.2e}), task={task_parameter_count:,} parameters "
        f"(lr={task_lr:.2e})"
    )
    return optimizer, trainable_named_parameters


def _current_learning_rates(optimizer):
    return {
        group.get("group_name", f"group_{index}"): group["lr"]
        for index, group in enumerate(optimizer.param_groups)
    }


def _autocast_context(device, enabled):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def _save_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str)


def main():
    run_started = time.time()
    configs = get_experiment_config()
    export_root, configs = setup_experiment(configs)
    export_root = Path(export_root)
    device = _device_from_config(configs)
    use_amp = bool(configs.get("use_amp", True)) and device.type == "cuda"

    print(f"Training device: {device}; AMP: {use_amp}")
    model, txt_processors = create_qure_models(configs, device)

    image_transform = image_transform_factory(config=configs)
    train_dataloader, test_dataloaders = create_dataloaders(
        image_transform, None, configs
    )
    if train_dataloader is None:
        raise RuntimeError("Training dataloader was not created")

    optimizer, trainable_named_parameters = _build_optimizer(model, configs)
    trainable_parameters = [
        parameter for _, parameter in trainable_named_parameters
    ]
    max_epochs = int(configs["epoch"])
    is_lora_run = bool(configs.get("use_lora", False))
    if is_lora_run:
        total_steps = max_epochs * len(train_dataloader)
        warmup_steps = int(configs.get("warmup_steps", 500))
        scheduler = _make_warmup_cosine_scheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr=float(configs.get("min_lr", configs["init_lr"] / 100)),
            reference_lr=float(configs["init_lr"]),
        )
        print(
            f"Scheduler: linear warmup for "
            f"{min(warmup_steps, total_steps)} steps, "
            f"then cosine decay over {total_steps} total steps"
        )
    else:
        scheduler = _make_epoch_scheduler(
            optimizer,
            configs["init_lr"],
            configs.get("min_lr", configs["init_lr"] / 100),
            max_epochs,
        )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    gpu_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU"
    )
    print(
        f"Trainable parameters: {sum(p.numel() for p in trainable_parameters):,}; "
        f"GPU: {gpu_name}"
    )

    resume_path = configs.get("resume_path") or configs.get("checkpoint_path")
    start_epoch = 0
    global_step = 0
    best_metric = None
    best_epoch = None
    history = []
    if resume_path:
        previous_history = Path(resume_path).parent / "history.json"
        if previous_history.exists():
            try:
                history = json.loads(previous_history.read_text(encoding="utf-8"))
                if not isinstance(history, list):
                    history = []
            except (OSError, json.JSONDecodeError):
                history = []
    if resume_path:
        checkpoint = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            restore_rng=True,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_metric = checkpoint.get("best_metric")
        best_epoch = checkpoint.get("best_epoch")
        if best_epoch is None and history:
            scored_history = [
                record for record in history
                if isinstance(record, dict) and "average_recall" in record
            ]
            if scored_history:
                best_record = max(
                    scored_history,
                    key=lambda record: float(record["average_recall"]),
                )
                best_epoch = best_record.get("epoch")
        print(f"Resumed from {resume_path}; next epoch={start_epoch + 1}")

    if start_epoch >= max_epochs:
        raise ValueError(
            f"Resume checkpoint is already at epoch {start_epoch}, but epoch={max_epochs}."
        )

    negative_period = max(1, max_epochs // int(configs["negative_definition_epoch_num"]))
    eval_frequency = max(1, int(configs.get("eval_frequency", 1)))
    model.train()

    for epoch in range(start_epoch, max_epochs):
        model.train()
        if configs.get("use_rerank", False) and epoch % negative_period == 0:
            if not configs.get("rerank_warmup", False) or epoch != 0:
                train_dataloader.dataset.rerank_score(
                    model, device, txt_processors, configs, epoch
                )

        epoch_running_loss = 0.0
        current_loss = None
        progress = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{max_epochs}")
        for batch_idx, (ref_images, tar_images, negative_images, sentences) in enumerate(progress):
            optimizer.zero_grad(set_to_none=True)
            ref_images = ref_images.to(device, non_blocking=True)
            tar_images = tar_images.to(device, non_blocking=True)
            negative_images = negative_images.to(device, non_blocking=True)
            sentences = [txt_processors["eval"](caption) for caption in sentences]

            with _autocast_context(device, use_amp):
                scores = model(
                    ref_images,
                    tar_images,
                    sentences,
                    negative_images,
                    configs["use_temp"],
                )
                log_softmax_scores = F.log_softmax(scores, dim=1)
                target_probs = torch.zeros_like(log_softmax_scores)
                target_probs[:, 0] = 1.0
                loss = F.kl_div(
                    log_softmax_scores, target_probs, reduction="batchmean"
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            if is_lora_run:
                scheduler.step()
            epoch_running_loss += loss.item()
            current_loss = epoch_running_loss / (batch_idx + 1)
            progress.set_postfix(
                loss=f"{current_loss:.5f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        if current_loss is None:
            raise RuntimeError("Training dataloader produced no batches")

        if not is_lora_run:
            scheduler.step()
        learning_rates = _current_learning_rates(optimizer)
        train_results = {
            "loss": current_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "lr_lora": learning_rates.get("lora"),
            "lr_task": learning_rates.get("task"),
            "epoch": epoch + 1,
            "global_step": global_step,
        }
        print(
            f"[EPOCH {epoch + 1}/{max_epochs}] "
            f"loss={current_loss:.6f} "
            f"lr_lora={learning_rates.get('lora', train_results['lr']):.3e} "
            f"lr_task={learning_rates.get('task', train_results['lr']):.3e}"
        )

        if _use_wandb(configs):
            import wandb

            wandb.log(train_results)
            if hasattr(model, "temp"):
                wandb.log({"temp": model.temp.item(), "epoch": epoch + 1})

        metrics = dict(train_results)
        if (epoch + 1) % eval_frequency == 0:
            if configs["dataset"] == "fashionIQ":
                metrics.update(
                    eval_fiq(model, txt_processors, test_dataloaders, epoch + 1, configs)
                )
            elif configs["dataset"] == "cirr":
                metrics.update(
                    eval_cirr(model, txt_processors, test_dataloaders, epoch + 1, configs)
                )
            model.train()

        selection_metric = metrics.get("average_recall", -metrics["loss"])
        if best_metric is None or selection_metric > best_metric:
            best_metric = selection_metric
            best_epoch = epoch + 1
            is_best = True
        else:
            is_best = False

        save_kwargs = dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            configs=configs,
            epoch=epoch,
            global_step=global_step,
            metrics=metrics,
            best_metric=best_metric,
            best_epoch=best_epoch,
        )
        # Keep only the latest resumable checkpoint and the best checkpoint.
        # The per-epoch full checkpoints are too large for the shared server.
        _atomic_save_checkpoint(export_root / "last.pt", **save_kwargs)
        if is_best:
            _atomic_save_checkpoint(export_root / "best.pt", **save_kwargs)
        _save_json(export_root / "latest_metrics.json", metrics)

        history.append(metrics)
        _save_json(export_root / "history.json", history)
        _save_json(
            export_root / "run_summary.json",
            {
                "run_id": export_root.name,
                "experiment_dir": str(export_root),
                "config": configs,
                "seed": configs.get("random_seed"),
                "trainable_parameters": sum(
                    parameter.numel() for parameter in trainable_parameters
                ),
                "optimizer": "AdamW",
                "scheduler": (
                    "linear_warmup_cosine_per_step"
                    if is_lora_run
                    else "cosine_per_epoch"
                ),
                "initial_lr": configs.get("init_lr"),
                "lora_lr": configs.get("lora_lr"),
                "warmup_steps": configs.get("warmup_steps"),
                "min_lr": configs.get("min_lr"),
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "best_checkpoint": str(export_root / "best.pt"),
                "last_epoch": epoch + 1,
                "global_step": global_step,
                "gpu": gpu_name,
                "elapsed_seconds": time.time() - run_started,
                "history_file": str(export_root / "history.json"),
            },
        )

    print(f"Training complete. Checkpoints are in: {export_root}")


if __name__ == "__main__":
    main()
