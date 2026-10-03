"""Train the QuRe objective with the shared BLIP2 Q-Former and two LoRA adapters."""

import json
import math
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


def _device_from_config(configs):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    device_idx = str(configs.get("device_idx", "0")).split(",")[0].strip()
    return torch.device(f"cuda:{device_idx}")


def _make_scheduler(optimizer, init_lr, min_lr, max_epochs):
    if init_lr <= 0:
        raise ValueError("init_lr must be positive")
    min_ratio = min_lr / init_lr

    def cosine_factor(epoch):
        progress = min(max(epoch, 0) / max(max_epochs, 1), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_factor)


def _autocast_context(device, enabled):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def _save_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str)


def main():
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

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters were found")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=configs["init_lr"],
        weight_decay=configs["weight_decay"],
    )
    max_epochs = int(configs["epoch"])
    scheduler = _make_scheduler(
        optimizer,
        configs["init_lr"],
        configs.get("min_lr", configs["init_lr"] / 100),
        max_epochs,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    resume_path = configs.get("resume_path") or configs.get("checkpoint_path")
    start_epoch = 0
    global_step = 0
    best_metric = None
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

            epoch_running_loss += loss.item()
            current_loss = epoch_running_loss / (batch_idx + 1)
            progress.set_postfix(
                loss=f"{current_loss:.5f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        scheduler.step()
        train_results = {
            "loss": current_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch": epoch + 1,
            "global_step": global_step,
        }
        print(
            f"[EPOCH {epoch + 1}/{max_epochs}] "
            f"loss={current_loss:.6f} lr={train_results['lr']:.3e}"
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
        )
        save_checkpoint(export_root / "last.pt", **save_kwargs)
        save_checkpoint(export_root / f"epoch_{epoch + 1:03d}.pt", **save_kwargs)
        if is_best:
            save_checkpoint(export_root / "best.pt", **save_kwargs)
        _save_json(export_root / "latest_metrics.json", metrics)

    print(f"Training complete. Checkpoints are in: {export_root}")


if __name__ == "__main__":
    main()
