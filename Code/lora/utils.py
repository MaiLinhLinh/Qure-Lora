from contextlib import contextmanager

import torch
from torch import nn

from .multi_adapter import MultiAdapterLinear


DEFAULT_TARGET_MODULES = (
    "attention.self.query",
    "attention.self.value",
    "crossattention.self.query",
    "crossattention.self.value",
)


def _config_get(config, key, default):
    if config is None:
        return default
    value = config.get(key, default)
    return default if value is None else value


def _target_matches(module_name, target_modules):
    return any(
        module_name == target or module_name.endswith("." + target)
        for target in target_modules
    )


def _get_parent_module(root, module_name):
    parent_name, _, child_name = module_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    return parent, child_name


def replace_linear_modules(module, config=None):
    """Replace only configured Linear suffixes and return their full names."""
    target_modules = tuple(
        _config_get(config, "target_modules", DEFAULT_TARGET_MODULES)
    )
    rank = int(_config_get(config, "rank", 8))
    alpha = float(_config_get(config, "alpha", 16))
    dropout = float(_config_get(config, "dropout", 0.05))

    replaced_names = []
    for module_name, child in list(module.named_modules()):
        if not module_name or not isinstance(child, nn.Linear):
            continue
        if isinstance(child, MultiAdapterLinear):
            continue
        if not _target_matches(module_name, target_modules):
            continue

        wrapped = MultiAdapterLinear.from_linear(
            child,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        parent, child_name = _get_parent_module(module, module_name)
        setattr(parent, child_name, wrapped)
        replaced_names.append(module_name)

    return replaced_names


def _adapter_modules(module):
    return [
        child
        for child in module.modules()
        if isinstance(child, MultiAdapterLinear)
    ]


def set_active_adapter(module, adapter_name):
    for child in _adapter_modules(module):
        child.set_active_adapter(adapter_name)


@contextmanager
def adapter_context(module, adapter_name):
    adapter_layers = _adapter_modules(module)
    previous = [layer.active_adapter for layer in adapter_layers]
    try:
        for layer in adapter_layers:
            layer.set_active_adapter(adapter_name)
        yield
    finally:
        for layer, previous_adapter in zip(adapter_layers, previous):
            layer.set_active_adapter(previous_adapter)


def mark_lora_and_task_params_trainable(model):
    """Freeze the base model, then enable only planned trainable parameters."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for module in model.modules():
        if isinstance(module, MultiAdapterLinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)

    for name in ("query_tokens", "temp"):
        parameter = getattr(model, name, None)
        if parameter is not None:
            parameter.requires_grad_(True)

    for name in ("ln_vision", "text_proj", "vision_proj"):
        module = getattr(model, name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad_(True)


def get_lora_state_dict(model):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }


def get_non_lora_trainable_state_dict(model):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and ".lora_A." not in name
        and ".lora_B." not in name
    }


def count_trainable_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def validate_lora_model(model):
    layers = _adapter_modules(model)
    if not layers:
        raise AssertionError("no MultiAdapterLinear modules found")
    for layer in layers:
        if set(layer.lora_A.keys()) != {"reference", "candidate"}:
            raise AssertionError("unexpected adapter names in lora_A")
        if set(layer.lora_B.keys()) != {"reference", "candidate"}:
            raise AssertionError("unexpected adapter names in lora_B")
        if layer.weight.requires_grad:
            raise AssertionError("base weight must be frozen")
        if layer.bias is not None and layer.bias.requires_grad:
            raise AssertionError("base bias must be frozen")
    return True
