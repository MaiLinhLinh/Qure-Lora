# Kaggle: QuRe-LoRA FashionIQ, 30 epochs

Tài liệu này dành cho phiên bản đã sửa trong thư mục `Code`. Không chạy full
training trên CPU; Kaggle phải bật GPU.

## 1. Chuẩn bị Kaggle

Tạo một Kaggle Dataset chứa toàn bộ thư mục `D:\NCKH\Code`, bao gồm cả:

```text
Code/
  checkpoints/
    blip2_pretrained_vitL.pth
    clip_vit_L.pth
    hf_cache/hub/models--bert-base-uncased/...
  configs/
  data/
  lavis/
  lora/
  models/
  train_qure.py
```

Trong Notebook, chọn `Settings -> Accelerator -> GPU`, sau đó `Add Data`:

1. Dataset code của project.
2. Dataset FashionIQ.

Không cần CIRR/CIRCO cho lần chạy này.

## 2. Cell cài package và tìm đường dẫn

Chạy các cell sau trong Kaggle Notebook. Thay hai slug ở đầu cell bằng tên
thật đang hiện trong `/kaggle/input`.

```python
import json
import os
import subprocess
import sys
from pathlib import Path

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
print("Input datasets:")
for path in sorted(Path("/kaggle/input").iterdir()):
    print(" -", path)
```

```python
CODE_INPUT = Path("/kaggle/input/qure-code")       # đổi slug nếu cần
FASHIONIQ_INPUT = Path("/kaggle/input/fashioniq")   # đổi slug nếu cần

if (CODE_INPUT / "Code").is_dir():
    CODE_ROOT = CODE_INPUT / "Code"
else:
    CODE_ROOT = CODE_INPUT

def find_fashioniq_root(parent: Path) -> Path:
    candidates = [parent]
    candidates.extend(path.parent for path in parent.rglob("captions"))
    for candidate in candidates:
        if (
            (candidate / "captions").is_dir()
            and (candidate / "image_splits").is_dir()
            and (candidate / "images").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        "Không tìm thấy root FashionIQ chứa captions/, image_splits/, images/."
    )

FASHIONIQ_ROOT = find_fashioniq_root(FASHIONIQ_INPUT)
BLIP2_PATH = CODE_ROOT / "checkpoints" / "blip2_pretrained_vitL.pth"
CLIP_PATH = CODE_ROOT / "checkpoints" / "clip_vit_L.pth"

print("CODE_ROOT      =", CODE_ROOT)
print("FASHIONIQ_ROOT =", FASHIONIQ_ROOT)
print("BLIP2 bytes    =", BLIP2_PATH.stat().st_size)
print("CLIP bytes     =", CLIP_PATH.stat().st_size)
assert BLIP2_PATH.stat().st_size > 100_000_000
assert CLIP_PATH.stat().st_size > 100_000_000
```

Cài các package còn thiếu. PyTorch/torchvision của Kaggle được giữ nguyên để
dùng CUDA của Kaggle:

```python
%pip install -q \
    omegaconf==2.3.0 \
    fairscale==0.4.4 \
    timm==0.4.12 \
    iopath==0.1.10 \
    ftfy==6.2.0 \
    transformers==4.26.1 \
    wandb==0.16.3
```

Nếu Kaggle yêu cầu restart kernel sau khi cài package, chọn restart rồi chạy
lại các cell từ đầu.

## 3. Tạo config chạy thật

Không sửa đường dẫn Windows trong source. Tạo một bản config ở
`/kaggle/working`; checkpoint BLIP2 được trỏ tự động bằng biến
`QURe_CHECKPOINT_ROOT`.

```python
os.environ["QURe_CHECKPOINT_ROOT"] = str(CODE_ROOT / "checkpoints")
os.environ["BLIP2_BERT_PATH"] = str(
    CODE_ROOT / "checkpoints" / "hf_cache" / "hub" /
    "models--bert-base-uncased" / "snapshots" /
    next(
        p.name for p in (
            CODE_ROOT / "checkpoints" / "hf_cache" / "hub" /
            "models--bert-base-uncased" / "snapshots"
        ).iterdir() if p.is_dir()
    )
)
sys.path.insert(0, str(CODE_ROOT))
```

```python
source_config = CODE_ROOT / "configs" / "fashionIQ" / "train.json"
run_config = Path("/kaggle/working/qure_lora_fashioniq.json")
config = json.loads(source_config.read_text(encoding="utf-8"))
config["fashioniq_root"] = str(FASHIONIQ_ROOT)
config["experiment_dir"] = "/kaggle/working/qure_lora_runs"
config["use_lora"] = True
config["epoch"] = 30
config["use_wandb"] = False
config["device_idx"] = "0"
config["num_workers"] = 4
run_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
print(run_config.read_text())
```

Config này giữ đúng thiết kế đã chốt:

```text
BLIP2 W0                  frozen
Vision Encoder            frozen
Q-Former query/value W0   frozen
reference LoRA            trainable
candidate LoRA            trainable
query_tokens              trainable
ln_vision                 trainable
text_proj/vision_proj     trainable
temp                      trainable
```

`positive` và `negative` đều route qua `candidate` adapter. Gallery inference
cũng route qua `candidate`; query inference route qua `reference`.

## 4. Smoke check trên GPU trước khi chạy 30 epoch

Cell này chỉ load model và kiểm tra route/parameter count, không training:

```python
import torch
from models import create_qure_models

assert torch.cuda.is_available(), "Kaggle chưa bật GPU"
device = torch.device("cuda:0")
model, txt_processors = create_qure_models(config, device)
print("device:", device)
print("lora configured:", model._lora_configured)
print("target Linear count:", len(model._lora_target_names))
print("trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
assert model._lora_configured
assert len(model._lora_target_names) == 36
assert model.query_tokens.requires_grad
```

Nếu cell này pass, giải phóng model trước khi chạy script:

```python
del model, txt_processors
torch.cuda.empty_cache()
```

## 5. Chạy full 30 epoch

```python
%cd {CODE_ROOT}
!python train_qure.py --config_path /kaggle/working/qure_lora_fashioniq.json
```

Script sẽ:

- load `blip2_pretrained_vitL.pth` làm base;
- inject một Q-Former duy nhất với hai adapter;
- train đúng 30 epoch trên FashionIQ;
- dùng hard-negative reranking theo config;
- evaluate FashionIQ sau mỗi epoch;
- lưu `last.pt`, `best.pt` và `epoch_001.pt`, ...;
- không lưu lại toàn bộ BLIP2 base trong adapter checkpoint.

Đầu ra nằm trong một thư mục dạng:

```text
/kaggle/working/qure_lora_runs/
  qure_lora_fashioniq_YYYY-MM-DD_0/
    config.json
    last.pt
    best.pt
    epoch_001.pt
    latest_metrics.json
```

## 6. Tiếp tục nếu Kaggle bị ngắt

Tìm checkpoint cuối:

```python
from pathlib import Path
last_checkpoints = sorted(
    Path("/kaggle/working/qure_lora_runs").glob("**/last.pt")
)
print(last_checkpoints)
```

Chạy lại với `--resume_path`; không dùng `--checkpoint_path` cho resume nếu
muốn giữ rõ nghĩa:

```python
RESUME = str(last_checkpoints[-1])
!python train_qure.py \
    --config_path /kaggle/working/qure_lora_fashioniq.json \
    --resume_path "$RESUME"
```

Resume khôi phục adapter, query tokens, các head trainable, optimizer,
scheduler, AMP scaler, epoch/global step và RNG.

## 7. Đánh giá checkpoint tốt nhất

Sau training, tạo config eval từ config train:

```python
best_path = sorted(
    Path("/kaggle/working/qure_lora_runs").glob("**/best.pt")
)[-1]
eval_config = dict(config)
eval_config.update({
    "mode": "eval",
    "experiment_description": "qure_lora_fashioniq_eval",
    "experiment_dir": "/kaggle/working/qure_lora_eval",
    "pretrained_path": str(best_path),
    "use_rerank": False,
})
eval_json = Path("/kaggle/working/qure_lora_fashioniq_eval.json")
eval_json.write_text(json.dumps(eval_config, indent=2), encoding="utf-8")
print("best checkpoint:", best_path)
```

```python
%cd {CODE_ROOT}
!python evaluate_qure/evaluate_fiq.py \
    --config_path /kaggle/working/qure_lora_fashioniq_eval.json
```

## 8. So sánh với full fine-tuning của QuRe

Sau khi hoàn tất LoRA, có thể chạy thêm baseline:

```python
baseline_source = CODE_ROOT / "configs" / "fashionIQ" / "train_baseline.json"
baseline_config = json.loads(baseline_source.read_text(encoding="utf-8"))
baseline_config["fashioniq_root"] = str(FASHIONIQ_ROOT)
baseline_config["experiment_dir"] = "/kaggle/working/qure_baseline_runs"
baseline_config["use_wandb"] = False
Path("/kaggle/working/qure_baseline_fashioniq.json").write_text(
    json.dumps(baseline_config, indent=2), encoding="utf-8"
)
```

```python
!python train_qure.py \
    --config_path /kaggle/working/qure_baseline_fashioniq.json
```

Khi báo cáo, ghi riêng `best.pt` của hai lần chạy, `latest_metrics.json`,
seed, batch size, learning rate và số trainable parameters. CIRCO được bỏ qua
theo phạm vi hiện tại.
