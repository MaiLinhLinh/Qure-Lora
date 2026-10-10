from options.command_line import load_config_from_command
from options.config_file import load_config_from_file


def _merge_configs(configs_ordered_by_increasing_priority):
    merged_config = {}
    for config in configs_ordered_by_increasing_priority:
        for k, v in config.items():
            merged_config[k] = v
    return merged_config


def _check_mandatory_config(config_from_config_file, user_defined_configs,
                            exception_keys=('experiment_description', 'device_idx')):
    exception_keys = [] if exception_keys is None else exception_keys
    trigger = False
    undefined_configs = []
    for key, val in config_from_config_file.items():
        if val == "":
            if key not in user_defined_configs and key not in exception_keys:
                trigger = True
                undefined_configs.append(key)
                print("Must define {} setting from command".format(key))
    if trigger:
        raise Exception('Mandatory configs not defined:', undefined_configs)


def _generate_experiment_description(configs, config_from_command):
    experiment_description = configs['experiment_description']

    if experiment_description == "":
        remove_keys = ['dataset', 'trainer', 'config_path', 'device_idx']
        for key in remove_keys:
            if key in config_from_command:
                config_from_command.pop(key)

        descriptors = []
        for key, val in config_from_command.items():
            descriptors.append(key + str(val))
        experiment_description = "_".join([configs['dataset'], configs['trainer'], *descriptors])
    return experiment_description


def _validate_ablation_config(config):
    """Validate the small set of variables allowed to differ in ablations."""
    valid_modes = {"separate", "shared", "query_only", "candidate_only"}
    valid_scopes = {"self", "cross", "both"}
    lora = dict(config.get("lora", {}))

    mode = config.get("lora_mode", lora.get("lora_mode", "separate"))
    scope = config.get("target_scope", lora.get("target_scope", "both"))
    lam = config.get("lora_lambda", lora.get("lora_lambda", 1.0))
    ensemble_members = int(
        config.get("ensemble_members", lora.get("ensemble_members", 1))
    )

    if mode not in valid_modes:
        raise ValueError(f"lora_mode must be one of {sorted(valid_modes)}, got {mode!r}")
    if scope not in valid_scopes:
        raise ValueError(f"target_scope must be one of {sorted(valid_scopes)}, got {scope!r}")
    if float(lam) < 0:
        raise ValueError("lora_lambda must be non-negative")
    if ensemble_members <= 0:
        raise ValueError("ensemble_members must be positive")
    if ensemble_members > 1 and mode != "separate":
        raise ValueError("ensemble_members > 1 requires lora_mode='separate'")

    config["lora_mode"] = mode
    config["target_scope"] = scope
    config["lora_lambda"] = float(lam)
    config["ensemble_members"] = ensemble_members
    lora.update(
        {
            "lora_mode": mode,
            "target_scope": scope,
            "lora_lambda": float(lam),
            "ensemble_members": ensemble_members,
        }
    )
    config["lora"] = lora

    if int(config.get("batch_size", 0)) <= 0:
        raise ValueError("batch_size must be positive")
    if int(config.get("rerank_batch_size", 16)) <= 0:
        raise ValueError("rerank_batch_size must be positive")
    if int(config.get("rerank_gallery_chunk_size", 1024)) <= 0:
        raise ValueError("rerank_gallery_chunk_size must be positive")
    if int(config.get("epoch", 0)) <= 0:
        raise ValueError("epoch must be positive")
    if config.get("use_lora", False):
        if float(config.get("lora_lr", 5e-4)) <= 0:
            raise ValueError("lora_lr must be positive")
        if int(config.get("warmup_steps", 500)) < 0:
            raise ValueError("warmup_steps must be non-negative")
    return config


def get_experiment_config():
    config_from_command, user_defined_configs = load_config_from_command()
    config_from_config_file = load_config_from_file(config_from_command['config_path'])
    _check_mandatory_config(config_from_config_file, user_defined_configs)
    merged_configs = _merge_configs([config_from_command, config_from_config_file, user_defined_configs])
    experiment_description = _generate_experiment_description(merged_configs, user_defined_configs)
    merged_configs['experiment_description'] = experiment_description
    return _validate_ablation_config(merged_configs)
