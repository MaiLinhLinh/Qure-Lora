# QuRe-LoRA Ablation Implementation Plan

This directory is the ablation branch copied from `D:\NCKH\Code`.
The original `Code` directory must remain unchanged and is the reference for
the existing v1 implementation.

## 1. Experimental invariants

Every run must use the same:

- BLIP2 pretrained checkpoint and BERT checkpoint;
- FashionIQ split, preprocessing, reranking and validation protocol;
- random seed (`1000` for the first comparison series);
- optimizer, scheduler, learning rate and weight decay;
- batch size (`8`, matching the existing QuRe-LoRA runs);
- LoRA rank (`8`), alpha (`16`) and dropout (`0.05`);
- total training epochs (`30`), unless a run is explicitly resumed;
- checkpoint and metric-saving policy.

Only the ablation variables may change between Exp 0--7.

## 2. Configurable variables

Add these fields to each experiment config:

```json
{
  "lora_mode": "separate",
  "lora_lambda": 1.0,
  "target_scope": "both"
}
```

Allowed values:

- `lora_mode`: `separate`, `shared`, `query_only`, `candidate_only`;
- `target_scope`: `self`, `cross`, `both`;
- `lora_lambda`: positive float, normally `1.0`.

The existing LoRA values remain fixed:

```json
{
  "rank": 8,
  "alpha": 16,
  "dropout": 0.05
}
```

`target_scope` must generate the target list instead of requiring manual code
edits:

- `self`: `attention.self.query/value`, 24 modules;
- `cross`: `crossattention.self.query/value`, 12 modules;
- `both`: both groups, 36 modules.

## 3. Code changes, in implementation order

### Phase A -- configuration and validation

1. Extend the config validation/defaults.
2. Reject unknown `lora_mode`, `target_scope` and invalid `lora_lambda`.
3. Remove the current hard-coded check that accepts only the four default
   target-module names.
4. Keep a structural report in the startup log:
   target scope, target count, adapter mode and trainable parameter count.

### Phase B -- LoRA layer behavior

Update `lora/multi_adapter.py` and `lora/utils.py`:

- preserve frozen `W0` and bias;
- retain independent `reference` and `candidate` parameters for `separate`;
- make `shared` use exactly one A/B parameter set for both branches;
- make `lora_lambda` scale only the LoRA delta;
- support a base-only path for branches that do not use LoRA;
- keep old v1 checkpoints loadable when `lora_mode` is `separate`.

The forward equation must be:

```text
y = W0(x) + lora_lambda * (alpha / rank) * B(A(dropout(x)))
```

### Phase C -- branch routing

Update `lavis/models/blip2_models/blip2_qure.py`:

- `separate`: reference uses reference adapter; positive/negative/gallery use
  candidate adapter;
- `shared`: both branches use the shared A/B parameters;
- `query_only`: reference uses LoRA; candidate uses frozen `W0` only;
- `candidate_only`: reference uses frozen `W0` only; candidate uses LoRA;
- positive and negative must remain two independent candidate forward passes;
- inference must retain reference adapter for the query and candidate adapter
  for gallery images when those adapters are enabled.

### Phase D -- logging and checkpoint history

Update `train_qure.py` and `evaluate_qure/eval_utils.py`:

- print validation metrics after every evaluated epoch;
- save `history.json` with one record per epoch;
- save `run_summary.json` containing run ID, seed, trainable parameters,
  optimizer, scheduler, learning rate, best epoch, best checkpoint, elapsed
  time and GPU name;
- retain `config.json`, `last.pt`, `best.pt` and `latest_metrics.json`;
- overwrite `last.pt` after each completed epoch and overwrite `best.pt` only
  when the validation metric improves; per-epoch metrics remain in
  `history.json`;
- keep validation metrics identical to the existing FashionIQ evaluation;
- do not enable W&B as a requirement for local logging.

### Phase E -- smoke checks

Before any full run, verify:

- `use_lora=false` creates the full-finetune baseline without LoRA modules;
- `separate` creates independent reference/candidate tensors;
- `shared` does not create duplicated trainable A/B tensors;
- `query_only` and `candidate_only` bypass the correct branch;
- target counts are 24, 12 and 36 for self, cross and both;
- `W0` and bias remain frozen;
- all trainable task parameters are included in the optimizer;
- one forward/backward step succeeds on GPU;
- one validation pass prints and saves metrics.

## 4. Experiment matrix

Each experiment gets its own config and output directory. No experiment may
resume from another experiment's checkpoint.

| Experiment | use_lora | lora_mode | target_scope | Purpose |
|---|---:|---|---|---|
| Exp 0 | false | ignored | ignored | QuRe baseline in the current environment |
| Exp 1 | true | separate | both | Existing separate reference/candidate LoRA |
| Exp 2 | true | shared | both | Shared LoRA parameters |
| Exp 3 | true | query_only | both | LoRA only on reference/query branch |
| Exp 4 | true | candidate_only | both | LoRA only on candidate/gallery branch |
| Exp 5 | true | selected best mode | self | Self-attention Q/V only |
| Exp 6 | true | selected best mode | cross | Cross-attention Q/V only |
| Exp 7 | true | selected best mode | both | Self + cross comparison |

The mode for Exp 5--7 is selected after Exp 2--4 as specified in the
experiment guide. If no selection is made yet, use `separate` only as a
temporary smoke-test mode, not as the final scientific claim.

## 5. Output layout

Use a separate directory for every run:

```text
runs/
  exp_00_qure_baseline/
  exp_01_lora_separate/
  exp_02_lora_shared/
  exp_03_lora_query_only/
  exp_04_lora_candidate_only/
  exp_05_lora_self/
  exp_06_lora_cross/
  exp_07_lora_self_cross/
```

Each run must contain at least:

```text
config.json
history.json
run_summary.json
last.pt
best.pt
```

Terminal output should also be captured as `logs/exp_XX.log` on the remote
server.

## 6. Execution order

1. Implement Phases A--D once in this directory.
2. Run Phase E smoke checks.
3. Run Exp 0 and Exp 1.
4. Run Exp 2, Exp 3 and Exp 4.
5. Select the strategy for Exp 5--7.
6. Run Exp 5, Exp 6 and Exp 7.
7. Compare only runs with matching seed, batch size, optimizer, schedule,
   reranking and validation protocol.

The original `Code` directory must not be edited during this process.

## 10. Exp 9: four-member LoRA ensemble

Exp 9 extends the Exp 1 architecture without changing the existing
`MultiAdapterLinear` implementation.  It uses a separate
`EnsembleMultiAdapterLinear` module with four independent member-specific
LoRA pairs:

```text
member 0: reference A/B + candidate A/B
member 1: reference A/B + candidate A/B
member 2: reference A/B + candidate A/B
member 3: reference A/B + candidate A/B
```

The BLIP2/Q-Former base weights and QuRe task parameters remain shared.  Each
member is initialized before training with its own random `A` factors and
zero-initialized `B` factors.  The training loss is the mean of the four
member losses, followed by one optimizer and scheduler step per batch.  Exp 9
keeps Exp 1's optimizer, learning rates, warmup, schedule, seed, data and
evaluation protocol.  It must start from the pretrained BLIP2 checkpoint and
must not resume from the Exp 1 checkpoint.

During evaluation and hard-negative reranking, features are extracted for all
four members and their retrieval score matrices are averaged before ranking.
The Exp 9 checkpoint is stored separately and is tagged as
`qure_lora_ensemble`; the existing Exp 0--7 checkpoint format remains
unchanged.
