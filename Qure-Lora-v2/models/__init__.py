import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(parent_dir)

from lavis.models import load_model_and_preprocess
from lora.utils import count_parameters, count_trainable_parameters

def create_qure_models(configs, device):
    print(f"{configs['model']} {configs['backbone']} training")

    model_name = f"{configs['model']}_qure"
    backbone = configs["backbone"]
    model, _, txt_processors = load_model_and_preprocess(name=model_name, model_type=backbone, is_eval=False,
                                                         device=device)

    if configs.get("use_lora", False):
        lora_config = dict(configs.get("lora", {}))
        for key in ("lora_mode", "lora_lambda", "target_scope"):
            if key in configs:
                lora_config[key] = configs[key]
        model.configure_lora(lora_config)
        print(
            "LoRA enabled: "
            f"mode={lora_config.get('lora_mode', 'separate')}, "
            f"scope={lora_config.get('target_scope', 'both')}, "
            f"lambda={lora_config.get('lora_lambda', 1.0)}, "
            f"target_modules={len(getattr(model, '_lora_target_names', ()))}, "
            f"trainable={count_trainable_parameters(model):,}, "
            f"total={count_parameters(model):,}"
        )

    return model, txt_processors
