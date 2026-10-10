# Running the QuRe-LoRA ablations

The ablation branch is self-contained. Run commands from `Code-ablation` so
that the relative `../fashing_iq`, `runs/`, and `logs/` paths resolve correctly.

## Generate the eight configs

```bash
python configs/fashionIQ/make_ablation_configs.py
```

The generated configs already use `fashioniq_root="../fashing_iq"`, matching
the supplied folder layout. Edit only machine-specific paths if your server
layout differs. Keep `batch_size=8`, `rank=8`, `alpha=16`, and `dropout=0.05`
fixed for the comparison.

Exp 5--7 are generated with `separate` as a temporary default. After Exp 2--4
are compared, regenerate them with the selected strategy, for example:

```bash
python configs/fashionIQ/make_ablation_configs.py --selected-mode shared
```

## Run one experiment and preserve the log

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 nohup python train_qure.py \
  --config_path configs/fashionIQ/ablation/exp_00.json \
  --device_idx 0 \
  > logs/exp_00.log 2>&1 &
```

Replace `exp_00` with `exp_01` through `exp_07`. Do not reuse a checkpoint
between different experiments. Each run writes `config.json`,
`history.json`, `run_summary.json`, `last.pt`, and `best.pt` into its own run
directory. `last.pt` is overwritten after every completed epoch, while
`best.pt` is overwritten only when the validation metric improves. Per-epoch
metrics remain in `history.json` and `latest_metrics.json`.

Validation metrics are printed as `[VALIDATION EPOCH N]` and are also included
in `history.json` and `latest_metrics.json`.
