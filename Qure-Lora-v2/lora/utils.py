from contextlib import contextmanager
from fnmatch import fnmatchcase

from torch import nn

from .multi_adapter import MultiAdapterLinear


DEFAULT_TARGET_MODULES = (
    "attention.self.query",
    "attention.self.value",
    "crossattention.self.query",
    "crossattention.self.value",
    # Q-Former feed-forward networks.  The query stream has separate FFN
    # modules in this implementation, so both variants are targeted.
    "encoder.layer.*.intermediate.dense",
    "encoder.layer.*.output.dense",
    "encoder.layer.*.intermediate_query.dense",
    "encoder.layer.*.output_query.dense",
)
SELF_TARGET_MODULES = (
    "attention.self.query",
    "attention.self.value",
    "encoder.layer.*.intermediate.dense",
    "encoder.layer.*.output.dense",
    "encoder.layer.*.intermediate_query.dense",
    "encoder.layer.*.output_query.dense",
)
CROSS_TARGET_MODULES = (
    "crossattention.self.query",
    "crossattention.self.value",
)
# Keep attention LoRA on both the self- and cross-attention modules, but adapt
# only the Q-Former query-token FFN.  The regular FFN is used for text tokens.
ATTENTION_QUERY_FFN_TARGET_MODULES = (
    "attention.self.query",
    "attention.self.value",
    "crossattention.self.query",
    "crossattention.self.value",
    "encoder.layer.*.intermediate_query.dense",
    "encoder.layer.*.output_query.dense",
)
VALID_LORA_MODES = ("separate", "shared", "query_only", "candidate_only")
VALID_TARGET_SCOPES = ("self", "cross", "both", "attention_query_ffn")


def target_modules_for_scope(target_scope):
    if target_scope not in VALID_TARGET_SCOPES:
        raise ValueError(
            f"unknown target_scope {target_scope!r}; "
            f"expected one of {VALID_TARGET_SCOPES}"
        )
    if target_scope == "self":
        return SELF_TARGET_MODULES
    if target_scope == "cross":
        return CROSS_TARGET_MODULES
    if target_scope == "attention_query_ffn":
        return ATTENTION_QUERY_FFN_TARGET_MODULES
    return DEFAULT_TARGET_MODULES


def _config_get(config, key, default):
    if config is None:
        return default
    value = config.get(key, default)
    return default if value is None else value


def _target_matches(module_name, target_modules):
    return any(
        fnmatchcase(module_name, target)
        or module_name == target
        or ("*" not in target and module_name.endswith("." + target))
        for target in target_modules
    )


def _get_parent_module(root, module_name):
    parent_name, _, child_name = module_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    return parent, child_name


def _resolve_lora_config(config):
    config = config or {}
    lora_mode = _config_get(config, "lora_mode", "separate")
    target_scope = _config_get(config, "target_scope", None)
    if lora_mode not in VALID_LORA_MODES:
        raise ValueError(
            f"unknown lora_mode {lora_mode!r}; "
            f"expected one of {VALID_LORA_MODES}"
        )

    if target_scope is not None:
        target_modules = target_modules_for_scope(target_scope)
    else:
        target_modules = tuple(
            _config_get(config, "target_modules", DEFAULT_TARGET_MODULES)
        )

    if not target_modules:
        raise ValueError("target_modules must not be empty")

    return {
        "target_modules": tuple(target_modules),
        "rank": int(_config_get(config, "rank", 8)),
        "alpha": float(_config_get(config, "alpha", 16)),
        "dropout": float(_config_get(config, "dropout", 0.05)),
        "lora_mode": lora_mode,
        "lora_lambda": float(_config_get(config, "lora_lambda", 1.0)),
        "target_scope": target_scope,
    }


def replace_linear_modules(module, config=None):
    """Replace configured Q-Former Linear suffixes with LoRA modules."""
    resolved = _resolve_lora_config(config)
    replaced_names = []
    adapter_mode = (
        resolved["lora_mode"]
        if resolved["lora_mode"] in MultiAdapterLinear.VALID_MODES
        else "separate"
    )

    for module_name, child in list(module.named_modules()):
        if not module_name or not isinstance(child, nn.Linear):
            continue
        if isinstance(child, MultiAdapterLinear):
            continue
        if not _target_matches(module_name, resolved["target_modules"]):
            continue

        wrapped = MultiAdapterLinear.from_linear(
            child,
            rank=resolved["rank"],
            alpha=resolved["alpha"],
            dropout=resolved["dropout"],
            lora_mode=adapter_mode,
            lora_lambda=resolved["lora_lambda"],
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


def mark_lora_and_task_params_trainable(model, lora_mode="separate"):
    """Freeze the base model and enable only the planned trainable tensors."""
    if lora_mode not in VALID_LORA_MODES:
        raise ValueError(f"unsupported lora_mode: {lora_mode}")

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for module in model.modules():
        if not isinstance(module, MultiAdapterLinear):
            continue
        if lora_mode == "query_only":
            module.lora_A["reference"].requires_grad_(True)
            module.lora_B["reference"].requires_grad_(True)
        elif lora_mode == "candidate_only":
            module.lora_A["candidate"].requires_grad_(True)
            module.lora_B["candidate"].requires_grad_(True)
        else:
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


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


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
        if layer.lora_mode == "shared":
            if layer.lora_A["reference"] is not layer.lora_A["candidate"]:
                raise AssertionError("shared mode must share lora_A")
            if layer.lora_B["reference"] is not layer.lora_B["candidate"]:
                raise AssertionError("shared mode must share lora_B")
    return True
