"""Create the eight reproducible FashionIQ ablation config files.

Run from the repository root, for example:

    python configs/fashionIQ/make_ablation_configs.py --output-dir configs/fashionIQ/ablation

Change ``--selected-mode`` only after Exp 2--4 have been compared.  The
default is ``separate`` so the generated files are immediately usable for a
smoke test, but the final Exp 5--7 runs should use the selected strategy.
"""

import argparse
import copy
import json
from pathlib import Path


EXPERIMENTS = {
    0: {"use_lora": False, "description": "exp_00_qure_baseline"},
    1: {"use_lora": True, "lora_mode": "separate", "target_scope": "both", "description": "exp_01_lora_separate"},
    2: {"use_lora": True, "lora_mode": "shared", "target_scope": "both", "description": "exp_02_lora_shared"},
    3: {"use_lora": True, "lora_mode": "query_only", "target_scope": "both", "description": "exp_03_lora_query_only"},
    4: {"use_lora": True, "lora_mode": "candidate_only", "target_scope": "both", "description": "exp_04_lora_candidate_only"},
    5: {"use_lora": True, "target_scope": "self", "description": "exp_05_lora_self"},
    6: {"use_lora": True, "target_scope": "cross", "description": "exp_06_lora_cross"},
    7: {"use_lora": True, "target_scope": "both", "description": "exp_07_lora_self_cross"},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path(__file__).with_name("train.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("ablation"),
    )
    parser.add_argument(
        "--selected-mode",
        choices=("separate", "shared", "query_only", "candidate_only"),
        default="separate",
    )
    args = parser.parse_args()

    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for exp_id, override in EXPERIMENTS.items():
        config = copy.deepcopy(base)
        config.update(
            {
                "experiment_description": override["description"],
                "experiment_dir": "runs",
                "batch_size": 8,
                "rerank_batch_size": 16,
                "rerank_gallery_chunk_size": 1024,
                "fashioniq_root": "../fashing_iq",
                "use_lora": override["use_lora"],
                "lora_mode": override.get("lora_mode", args.selected_mode),
                "lora_lambda": 1.0,
                "target_scope": override.get("target_scope", "both"),
            }
        )
        path = args.output_dir / f"exp_{exp_id:02d}.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
