"""Checkpoint helpers for QuRe-LoRA training and deployment.

The BLIP2 base checkpoint is deliberately not copied into a deployment
checkpoint.  A deployment checkpoint contains the adapter tensors and the
small set of explicitly trainable non-LoRA tensors; the configured BLIP2
pretrained file remains the single source of W0 and the other frozen weights.
"""

import random
from pathlib import Path

import numpy as np
import torch


CHECKPOINT_FORMAT_VERSION = 1


def _rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        # load_checkpoint may use map_location=cuda; torch.set_rng_state
        # requires the CPU ByteTensor RNG state.
        torch.set_rng_state(state["torch"].detach().cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(
            [item.detach().cpu() for item in state["cuda"]]
        )


def make_model_state(model):
    if getattr(model, "_lora_configured", False):
        return {
            "lora_state_dict": model.get_lora_state_dict(),
            "trainable_state_dict": model.get_trainable_state_dict(),
        }
    return {"model_state_dict": model.state_dict()}


def make_checkpoint(model, optimizer=None, scheduler=None, scaler=None,
                    configs=None, epoch=-1, global_step=0, metrics=None,
                    best_metric=None, best_epoch=None):
    configs = dict(configs or {})
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_type": "qure_lora" if getattr(model, "_lora_configured", False) else "qure_full",
        "epoch": int(epoch),
        "global_step": int(global_step),
        "configs": configs,
        "metrics": dict(metrics or {}),
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        **make_model_state(model),
        "rng_state": _rng_state(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    return payload


def save_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None,
                    configs=None, epoch=-1, global_step=0, metrics=None,
                    best_metric=None, best_epoch=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = make_checkpoint(
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
    torch.save(payload, path)
    return payload


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None,
                    restore_rng=True, map_location="cpu"):
    # The checkpoint intentionally stores Python/NumPy RNG state.  PyTorch
    # 2.6+ defaults to weights_only=True, which cannot deserialize that state.
    # These files are produced by this trusted training code, so request the
    # complete checkpoint payload explicitly, with compatibility for older
    # PyTorch versions that do not expose weights_only.
    try:
        checkpoint = torch.load(
            path, map_location=map_location, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)

    if getattr(model, "_lora_configured", False):
        if "lora_state_dict" not in checkpoint:
            raise ValueError(
                "The model is configured for LoRA but the checkpoint has no "
                "lora_state_dict."
            )
        model.load_lora_checkpoint(checkpoint)
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if restore_rng:
        _restore_rng_state(checkpoint.get("rng_state"))

    return checkpoint
