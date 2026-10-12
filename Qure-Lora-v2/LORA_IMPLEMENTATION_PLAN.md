# Kế hoạch triển khai LoRA hai adapter cho QuRe trên BLIP2 pretrained

> **Qure-Lora-v2 note:** Đây là bản kế hoạch nền được copy từ v1. Trong v2,
> ngoài `query/value` của self/cross-attention, LoRA còn được inject vào
> `intermediate.dense`, `output.dense`, `intermediate_query.dense` và
> `output_query.dense` của Q-Former. Vì vậy các đoạn audit của v1 ghi
> `36 wrappers` hoặc `FFN không có LoRA` không áp dụng cho v2; v2 kỳ vọng
> `84 MultiAdapterLinear` với cấu hình `pretrain_vitL` hiện tại.

## 0. Mục tiêu và phạm vi

Mục tiêu là sửa bản sao mã nguồn trong D:\NCKH\Code để QuRe sử dụng:

~~~text
Một BLIP2 pretrained
└── một Vision Encoder duy nhất
└── một Q-Former duy nhất
      ├── một bộ trọng số Q-Former đã pretrained trong BLIP2, W0 dùng chung và frozen
      ├── một bộ query_tokens dùng chung và trainable
      ├── LoRA_reference
      └── LoRA_candidate
~~~

Hai adapter được dùng theo vai trò dữ liệu, không theo nhãn positive/negative:

~~~text
reference image + caption → LoRA_reference
positive image             → LoRA_candidate
negative image             → LoRA_candidate
gallery image              → LoRA_candidate
~~~

Không tạo ba Q-Former, không tạo ba bộ query_tokens và không tạo ba bộ trọng số
BLIP2. Các module Linear mục tiêu chỉ được bọc bằng wrapper LoRA nhưng vẫn dùng
chung đúng một W0 frozen cho hai adapter.

Loss, dataset, hard-negative sampling và công thức score của QuRe được giữ
nguyên trừ những thay đổi cần thiết để routing adapter đúng trong training và
inference.

---

## 1. Các bất biến bắt buộc của kiến trúc

### 1.1. Một Q-Former và một bộ query tokens

Model chỉ có một object:

~~~python
model.Qformer
~~~

Trong đó chỉ có một:

~~~python
model.query_tokens
~~~

query_tokens có dạng gần như:

~~~text
[1, num_query_tokens, hidden_size]
~~~

Khi chạy batch, nó được expand theo batch nhưng không được nhân bản thành
query_tokens_reference, query_tokens_positive hoặc query_tokens_negative.

query_tokens là tham số học được. Nó nhận gradient từ cả ba lần gọi Q-Former:

~~~text
reference branch + positive branch + negative branch
~~~

Logic trạng thái đầu `H0` phải giữ đúng Q-Former của QuRe:

~~~text
positive/negative/gallery:
    H0 = query_tokens

reference có caption:
    H0 = [query_tokens ; text_embeddings]
    # query_tokens nằm ở phần đầu, caption nằm ở phần sau
~~~

`query_tokens` là phần `H0` dùng để truy vấn ảnh; trong nhánh reference, toàn
bộ chuỗi `H0` còn chứa thêm các embedding của caption. Không được thay
`query_tokens` bằng một ma trận Wq hoặc tạo H0 riêng cho từng nhánh.

### 1.2. Trọng số base dùng chung

Với mỗi Linear mục tiêu bên trong Q-Former, chỉ có một:

~~~python
weight       # W0, frozen
bias         # frozen
~~~

và hai cặp LoRA:

~~~python
lora_A["reference"]
lora_B["reference"]

lora_A["candidate"]
lora_B["candidate"]
~~~

Với adapter đang active:

\[
W_a(x) = W_0x + \frac{\alpha}{r}B_aA_ax
\]

Trong đó a là reference hoặc candidate. Chỉ các projection `query` và `value`
trong self-attention/cross-attention là Linear mục tiêu. Các Linear khác của
Q-Former không có LoRA và vẫn frozen.

Vì vậy, với attention projection, cách hiểu đúng là:

~~~text
Wq_reference = Wq0 + Bq_reference @ Aq_reference
Wq_candidate = Wq0 + Bq_candidate @ Aq_candidate

Wv_reference = Wv0 + Bv_reference @ Av_reference
Wv_candidate = Wv0 + Bv_candidate @ Av_candidate
~~~

`Wq0` và `Wv0` dùng chung, frozen; chỉ các ma trận `A/B` tương ứng được train.
Không được gọi `requires_grad_(False)` trên toàn bộ wrapper sau khi inject nếu
điều đó làm freeze nhầm cả `lora_A` và `lora_B`.

### 1.3. Trạng thái train/freeze cuối cùng

| Thành phần | Trạng thái | Ghi chú |
|---|---:|---|
| BLIP2 Vision Encoder | Frozen | Giữ không gian đặc trưng ảnh pretrained |
| ln_vision | Trainable | Giữ nguyên đúng code QuRe gốc; đây là LayerNorm riêng sau Vision Encoder |
| BLIP2-pretrained Q-Former W0 (`Wq0`, `Wk0`, `Wv0`, `Wo0`, FFN, LayerNorm, Embedding) | Frozen | W0 được lấy từ checkpoint BLIP2; LoRA `A/B` trên `Wq0` và `Wv0` vẫn trainable |
| query_tokens | Trainable | Tham số truy vấn dùng chung cho cả ba nhánh |
| LoRA reference A/B | Trainable | Chỉ nhận gradient từ reference branch |
| LoRA candidate A/B | Trainable | Nhận gradient từ positive và negative branch |
| text_proj | Trainable | Head của QuRe, không phải LoRA trong Q-Former |
| vision_proj | Trainable | Dùng chung cho positive, negative và gallery |
| temp | Trainable | Giữ cách huấn luyện QuRe gốc |
| itm_head | Unused / excluded | QuRe không gọi trong forward hoặc các hàm extract; loại khỏi optimizer |

Lưu ý: code QuRe gốc chỉ duyệt `self.visual_encoder.named_parameters()` để
freeze Vision Encoder. `ln_vision` là module riêng được tạo trong
`init_vision_encoder()` và không nằm bên trong `visual_encoder`, nên nó vẫn
trainable. Vì mục tiêu là giữ nguyên kiến trúc và trạng thái train của QuRe
gốc, kế hoạch này cũng giữ `ln_vision` trainable.

---

## 2. Luồng tính toán phải giữ nguyên

### 2.1. Training

#### Reference branch

~~~text
reference image
        ↓
BLIP2 Vision Encoder frozen
        ↓
Q-Former + LoRA_reference
        ↓
query_tokens representation
        ↓
text_proj
        ↓
query vector q
~~~

Reference branch sử dụng caption thông qua lời gọi Q-Former hiện tại của
QuRe: `H0 = [query_tokens ; text_embeddings]`. Không được chuyển caption sang
candidate adapter.

#### Positive branch

~~~text
positive image
        ↓
BLIP2 Vision Encoder frozen
        ↓
Q-Former + LoRA_candidate
        ↓
32 query-token features
        ↓
vision_proj
        ↓
positive features
~~~

#### Negative branch

~~~text
negative image
        ↓
BLIP2 Vision Encoder frozen
        ↓
Q-Former + LoRA_candidate
        ↓
32 query-token features
        ↓
vision_proj
        ↓
negative features
~~~

Positive và negative phải chạy trong cùng context candidate nhưng vẫn là hai
lần forward độc lập trên hai ảnh khác nhau.

### 2.2. Inference

~~~text
reference image + caption
        ↓
Q-Former + LoRA_reference
        ↓
query vector q, kích thước 256
~~~

~~~text
gallery image
        ↓
Q-Former + LoRA_candidate
        ↓
32 gallery vectors, mỗi vector kích thước 256
~~~

Điểm ảnh giữ nguyên cách tính QuRe:

\[
s(q,I_j)=\max_l q\cdot f_{j,l}
\]

Trong inference không có adapter positive và adapter negative riêng. Mọi ảnh
gallery đều là candidate chưa biết trước nhãn và đi qua LoRA_candidate.

---

## 3. Thứ tự load BLIP2 pretrained và inject LoRA

Đây là thứ tự bắt buộc:

~~~text
1. Khởi tạo Blip2QuRe
2. Load BLIP2 pretrained checkpoint
3. Xác nhận Q-Former và query_tokens đã ở đúng trạng thái pretrained
4. Freeze các module cần freeze
5. Thay các nn.Linear mục tiêu bằng MultiAdapterLinear
6. Copy weight/bias gốc vào layer mới
7. Khởi tạo A/B cho reference và candidate
8. Bật requires_grad cho LoRA, query_tokens, ln_vision và QuRe heads
9. Nếu có, load adapter checkpoint của QuRe-LoRA
10. Train hoặc eval
~~~

Không inject adapter trước khi checkpoint BLIP2 được load. Cách làm đó dễ gây
missing key/unexpected key và khó phân biệt lỗi của checkpoint base với lỗi của
checkpoint adapter.

BLIP2 pretrained checkpoint chứa Q-Former, query_tokens, ln_vision và các
projection liên quan; đây là nguồn tạo W0 frozen cho LoRA. Vision Encoder không
nằm trong file này: code BLIP2 của source khởi tạo CLIP-L riêng từ checkpoint
`clip_vit_L.pth`. Không phải mỗi lần train LoRA lại lưu một bản sao của toàn bộ
BLIP2.

---

## 4. Các file cần tạo và sửa

### 4.1. Tạo module LoRA nội bộ

Tạo:

~~~text
D:\NCKH\Code\lora\
├── __init__.py
├── multi_adapter.py
└── utils.py
~~~

Không import trực tiếp bằng đường dẫn cứng sang D:\NCKH\LoRa\LoRA. Code phải tự
chạy độc lập trong folder Code.

#### lora/multi_adapter.py

Tạo class:

~~~python
class MultiAdapterLinear(nn.Linear):
    ...
~~~

Yêu cầu:

- Giữ weight và bias với đúng tên state-dict như nn.Linear.
- Có lora_A và lora_B dưới dạng nn.ParameterDict.
- Có đúng hai adapter: reference, candidate.
- Có active_adapter không phải parameter.
- Có rank, alpha, scaling, dropout.
- Dùng cùng một weight cho cả hai adapter.
- weight.requires_grad=False và bias.requires_grad=False.
- merge_weights=False cố định.
- Tương thích device và dtype của Linear gốc.

Forward logic:

~~~python
base_output = F.linear(x, self.weight, self.bias)

A = self.lora_A[self.active_adapter]
B = self.lora_B[self.active_adapter]

delta = F.dropout(x, p=self.lora_dropout, training=self.training)
delta = delta @ A.transpose(-1, -2)
delta = delta @ B.transpose(-1, -2)
delta = delta * self.scaling

return base_output + delta
~~~

Không dùng cơ chế merge khi gọi model.eval(), vì trong cùng một model sẽ có
lúc dùng reference, có lúc dùng candidate.

#### lora/utils.py

Cần có các tiện ích:

~~~python
replace_linear_modules(module, config)
set_active_adapter(module, adapter_name)
adapter_context(module, adapter_name)
mark_lora_and_task_params_trainable(model)
get_lora_state_dict(model)
get_non_lora_trainable_state_dict(model)
count_trainable_parameters(model)
validate_lora_model(model)
~~~

adapter_context phải khôi phục adapter trước đó sau khi thoát context, kể cả
khi forward phát sinh exception.

### 4.2. Sửa lavis/models/blip2_models/blip2_qure.py

Thêm các thành phần:

~~~python
def configure_lora(self, lora_config):
    ...

def set_active_adapter(self, adapter_name):
    ...

def get_lora_state_dict(self):
    ...

def get_trainable_state_dict(self):
    ...

def load_lora_checkpoint(self, checkpoint):
    ...
~~~

configure_lora() phải:

1. Kiểm tra model chưa được inject LoRA trước đó.
2. Duyệt self.Qformer.bert đệ quy.
3. Tìm các nn.Linear nằm trong danh sách target.
4. Tạo MultiAdapterLinear cùng in_features, out_features, bias, device và dtype.
5. Copy weight và bias của Linear cũ.
6. Khởi tạo A/B cho hai adapter.
7. Thay module cũ bằng module mới.
8. Freeze base parameters.
9. Giữ query_tokens, ln_vision, text_proj, vision_proj và temp trainable; loại
    itm_head khỏi optimizer vì QuRe không dùng nó trong forward.
10. Đặt adapter mặc định là candidate hoặc trạng thái được chỉ định rõ; mọi
    hàm encode vẫn phải dùng context tường minh, không được phụ thuộc vào
    default này.

Tạo helper để tránh lặp code:

~~~python
def _encode_reference(...):
    with self.adapter_context("reference"):
        ...

def _encode_candidate(...):
    with self.adapter_context("candidate"):
        ...
~~~

Các hàm phải được cập nhật routing:

~~~text
forward
extract_target_features
extract_query_features_fiq
extract_query_features_cirr
extract_query_features_cirr_val
extract_target_image_features_fiq
~~~

Không thay đổi cách lấy token, pooling, normalize, bmm, max hoặc temperature
nếu không cần thiết cho routing.

### 4.3. Các Linear Q-Former được inject

Trong self.Qformer.bert, bản triển khai chính chỉ inject `Wq` và `Wv` của mọi
module attention tương ứng:

~~~text
encoder.layer.*.attention.self.query
encoder.layer.*.attention.self.value

encoder.layer.*.crossattention.self.query
encoder.layer.*.crossattention.self.value
~~~

`crossattention` chỉ tồn tại ở các layer được cấu hình
`cross_attention_freq`. Không inject LoRA vào các Linear còn lại:

~~~text
encoder.layer.*.attention.self.key
encoder.layer.*.attention.output.dense
encoder.layer.*.crossattention.self.key
encoder.layer.*.crossattention.output.dense
encoder.layer.*.intermediate.dense
encoder.layer.*.output.dense
encoder.layer.*.intermediate_query.dense
encoder.layer.*.output_query.dense
Qformer.cls
LM head
itm_head
visual_encoder
vision_proj
text_proj
~~~

Các Linear bên trong Q-Former không nằm trong danh sách target vẫn chạy
forward với trọng số pretrained frozen. `query_tokens`, `ln_vision`,
`text_proj`, `vision_proj` và `temp` train trực tiếp theo bảng trạng thái ở
mục 1.3; chúng không phải là LoRA target.

Không sửa Qformer.py trong phiên bản đầu. Việc thay Linear diễn ra sau khi
Q-Former được tạo và load checkpoint.

### 4.4. Sửa models/__init__.py

Sau khi LAVIS tạo và load xong model pretrained, gọi:

~~~python
if isinstance(model, Blip2QuRe) and cfg.get("use_lora", False):
    model.configure_lora(cfg["lora"])
~~~

Không gọi cấu hình Q-Former LoRA cho BlipQuRe, vì model đó sử dụng BLIP
encoder khác và không có Qformer.bert.

### 4.5. Sửa config train/eval

Thêm cấu hình tương đương:

~~~json
{
  "use_lora": true,
  "lora": {
    "rank": 8,
    "alpha": 16,
    "dropout": 0.05,
    "target_modules": [
      "attention.self.query",
      "attention.self.value",
      "crossattention.self.query",
      "crossattention.self.value"
    ],
    "bias": "none",
    "train_query_tokens": true,
    "train_qure_heads": true,
    "freeze_vision_encoder": true,
    "freeze_vision_norm": false,
    "merge_weights": false
  }
}
~~~

Hai adapter không lấy từ config tên tùy ý; tên cố định là:

~~~text
reference
candidate
~~~

Không thêm positive_adapter hoặc negative_adapter.

Các config FashionIQ và CIRR phải dùng model BLIP2. Các config hiện đang dùng:

~~~json
"model": "blip"
~~~

chỉ được giữ nguyên nếu cố ý chạy model BLIP cũ và không dùng Q-Former LoRA.
CIRCO nằm ngoài phạm vi triển khai hiện tại và không được đưa vào các bước
train/evaluation của kế hoạch này.

### 4.6. Sửa train_qure.py

Giữ nguyên:

- batch reference/target/negative/sentence;
- positive score và negative score;
- log_softmax;
- KL-divergence loss;
- hard-negative sampling.

Sửa các phần:

~~~python
trainable_params = [
    p for p in model.parameters()
    if p.requires_grad
]
optimizer = AdamW(trainable_params, ...)
~~~

Thêm audit trước epoch đầu:

~~~text
base Q-Former weight requires_grad=False
reference LoRA requires_grad=True
candidate LoRA requires_grad=True
query_tokens requires_grad=True
ln_vision requires_grad=True
text_proj requires_grad=True
vision_proj requires_grad=True
temp requires_grad=True
itm_head không có gradient và không nằm trong optimizer
~~~

Các lỗi độc lập đang tồn tại trong script cũng phải được xử lý trước khi train
thật:

- biến topk phải được định nghĩa trước khi log;
- saving_path không được là chuỗi rỗng;
- đường dẫn export phải được tạo từ export_root hoặc config hợp lệ;
- cách dùng min_lr và init_lr phải nhất quán.

### 4.7. Sửa các script evaluation

Kiểm tra và cập nhật:

~~~text
evaluate_qure/evaluate_fiq.py
evaluate_qure/evaluate_cirr.py
HP_FashionIQ/evaluate.py
~~~

Thứ tự bắt buộc:

~~~text
load BLIP2 pretrained
→ inject reference/candidate adapters
→ load LoRA checkpoint
→ model.eval()
→ query extraction bằng reference
→ gallery extraction bằng candidate
→ score
~~~

Hard-negative reranking cũng phải dùng đúng routing này. Không được để
model.eval() làm merge adapter.

### 4.8. Dataset và loss

Không cần sửa logic trong:

~~~text
data/fashionIQ.py
data/cirr.py
~~~

Dataset FashionIQ và CIRR vẫn cung cấp reference, target/positive, negative và
caption như cũ. CIRCO được bỏ qua trong phiên bản triển khai này.

---

## 5. Định dạng checkpoint triển khai

Có hai loại checkpoint: checkpoint resume để tiếp tục train và checkpoint
deployment chỉ phục vụ inference.

### 5.0. Tình trạng BLIP2 checkpoint trong source hiện tại

Sau khi đọc đường đi khởi tạo model trong source, kết luận là:

```text
configs/fashionIQ/train.json
configs/fashionIQ/eval.json
configs/cirr/train.json
configs/cirr/eval.json
        ↓ backbone = pretrain_vitL
Blip2QuRe.PRETRAINED_MODEL_CONFIG_DICT
        ↓
lavis/configs/models/blip2/blip2_pretrain_vitL.yaml
        ↓
pretrained = D:/NCKH/Code/checkpoints/blip2_pretrained_vitL.pth
```

Trong `Blip2QuRe.from_config()`, model gọi
`load_checkpoint_from_config(cfg)`. Vì config có `load_finetuned: False` và
`pretrained` là đường dẫn local, `BaseModel` gọi `load_from_pretrained()` và
nạp state dict BLIP2 vào Q-Former, LayerNorm và query_tokens. Vision Encoder
được khởi tạo riêng trong `create_clip_vit_L()` từ
`D:/NCKH/Code/checkpoints/clip_vit_L.pth`.

Hiện hàm load dùng `strict=False`, nên sau lần tải đầu tiên bắt buộc phải in và
kiểm tra `missing_keys`/`unexpected_keys`. Với source hiện tại, checkpoint
BLIP2 có `Qformer` và `query_tokens` nhưng không chứa các key của
`visual_encoder`; đây là bình thường vì CLIP-L được load từ checkpoint riêng.
Các thành phần QuRe tạo riêng như `text_proj`, `vision_proj`, `itm_head` và
`temp` cũng không phải là mục tiêu của checkpoint BLIP2.

Trong workspace hiện tại, hai file local đã có và đã được kiểm tra có nội dung:

~~~text
D:/NCKH/Code/checkpoints/blip2_pretrained_vitL.pth
D:/NCKH/Code/checkpoints/clip_vit_L.pth
~~~

Checkpoint BLIP2 có top-level key `model`, 346 state keys, có `Qformer` và
`query_tokens`; checkpoint CLIP-L có kích thước khác và được dùng riêng cho
Vision Encoder.

Điều này **không có nghĩa phải chạy lại BLIP2 pretraining**. BLIP2 đã được
pretrain bởi tác giả và source đang trỏ đúng đến checkpoint công khai. Chỉ cần:

1. Giữ hai checkpoint local ở thư mục `D:/NCKH/Code/checkpoints/`.
2. Khởi tạo model để xác nhận BLIP2 và CLIP-L đều load được.
3. Chạy QuRe training trên FashionIQ/CIRR; phiên bản LoRA sẽ inject adapter sau
   khi checkpoint BLIP2 đã được load.

Cần phân biệt hai loại checkpoint:

```text
BLIP2 checkpoint:
    nền pretrained, dùng để khởi tạo model

QuRe checkpoint:
    kết quả train downstream trên FashionIQ/CIRR
```

`pretrained_path` trong `configs/fashionIQ/eval.json` và
`configs/cirr/eval.json` là đường dẫn tới QuRe checkpoint downstream có
`model.pth`, không phải đường dẫn BLIP2 checkpoint. Hiện các file eval vẫn để
placeholder `[Path to the pretrained model]`, nên chưa thể evaluation nếu
chưa có checkpoint QuRe tương ứng.

README của source có liên kết checkpoint QuRe FashionIQ/CIRR do tác giả cung
cấp. Nếu mục tiêu là train LoRA mới thì không cần lấy QuRe checkpoint đó làm
trọng số khởi đầu; chỉ cần BLIP2 base, sau đó train LoRA và các tham số
trainable của QuRe. Nếu mục tiêu là so sánh với QuRe gốc, có thể tải thêm QuRe
checkpoint downstream để chạy baseline.

### 5.1. Base model bắt buộc

LoRA checkpoint không chứa lại toàn bộ BLIP2. Khi deploy phải có:

~~~text
BLIP2 pretrained checkpoint
BLIP2 model type/config
QuRe config
LoRA checkpoint
~~~

LoRA checkpoint chỉ hợp lệ nếu được tạo trên cùng:

~~~text
base model type
pretrained checkpoint
Q-Former hidden size
num_query_tokens
LoRA rank
LoRA alpha
target module list
~~~

### 5.2. Checkpoint deployment tối thiểu

Tên đề xuất:

~~~text
qure_blip2_lora_best.pt
~~~

Cấu trúc logic:

~~~python
{
    "format_version": 1,
    "model_name": "blip2_qure",
    "base_model_type": "<blip2 model type>",
    "base_checkpoint": "<pretrained checkpoint identifier>",
    "num_query_tokens": 32,
    "embed_dim": 256,
    "lora_config": {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "target_modules": [
            "attention.self.query",
            "attention.self.value",
            "crossattention.self.query",
            "crossattention.self.value"
        ],
        "adapters": ["reference", "candidate"],
        "merge_weights": False
    },
    "lora_state_dict": {
        "...lora_A.reference": "...",
        "...lora_B.reference": "...",
        "...lora_A.candidate": "...",
        "...lora_B.candidate": "..."
    },
    "trainable_state_dict": {
        "query_tokens": "...",
        "ln_vision.weight": "...",
        "ln_vision.bias": "...",
        "text_proj.weight": "...",
        "text_proj.bias": "...",
        "vision_proj.weight": "...",
        "vision_proj.bias": "...",
        "temp": "..."
    },
    "metrics": {
        "epoch": 0,
        "step": 0,
        "best_metric": 0.0
    }
}
~~~

Trong file thực tế, các tensor được lưu bằng torch.save; phần trên mô tả key
logic, không phải JSON.

### 5.3. Checkpoint resume đầy đủ

Tên đề xuất:

~~~text
qure_blip2_lora_last_resume.pt
~~~

Ngoài toàn bộ trường của deployment checkpoint, thêm:

~~~python
{
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "scaler_state_dict": ...,       # nếu dùng AMP
    "epoch": ...,
    "global_step": ...,
    "best_metric": ...,
    "rng_state": ...,
    "cuda_rng_state": ...,
    "sampler_state": ...             # nếu sampler có state
}
~~~

Không dùng checkpoint resume làm checkpoint deployment nếu không cần thiết,
vì optimizer và RNG làm file lớn mà inference không sử dụng.

### 5.4. Hàm lưu

Trong Blip2QuRe hoặc utility chung cần có logic tương đương:

~~~python
def get_lora_state_dict(self):
    # chỉ lấy key chứa lora_A hoặc lora_B

def get_trainable_state_dict(self):
    # lấy query_tokens, ln_vision, text_proj, vision_proj, temp
    # không lặp lại lora_state_dict
~~~

Khi lưu, phải chuyển tensor về CPU nếu cần để checkpoint không giữ tham chiếu
CUDA không cần thiết.

Không dùng nguyên model.state_dict() làm checkpoint deployment, vì như vậy sẽ
lưu lại toàn bộ W0 frozen và làm mất lợi ích của adapter-only checkpoint.

### 5.5. Quy trình load checkpoint deployment

~~~python
model = create_blip2_qure_model(config)
model.load_checkpoint_from_config(config)   # BLIP2 pretrained
model.configure_lora(config["lora"])

checkpoint = torch.load(path, map_location="cpu")
validate_checkpoint_metadata(model, checkpoint)

model.load_state_dict(
    checkpoint["lora_state_dict"],
    strict=False
)
model.load_state_dict(
    checkpoint["trainable_state_dict"],
    strict=False
)

model.eval()
~~~

Sau khi load phải kiểm tra:

~~~text
không thiếu LoRA key
không có unexpected LoRA key
W0 vẫn frozen
query_tokens đúng checkpoint
reference adapter load đúng
candidate adapter load đúng
~~~

Không load adapter state trước khi gọi configure_lora().

---

## 6. Các mốc kiểm tra triển khai

### Checkpoint triển khai 0: kiểm tra source tree

Mục tiêu:

- Code tự import được mà không cần import từ D:\NCKH\QuRe;
- model route đang dùng là Blip2QuRe;
- không có đường dẫn cứng tới folder LoRA bên ngoài;
- config BLIP2 pretrained tồn tại.

Điều kiện đạt:

~~~text
import model thành công
load config thành công
load BLIP2 pretrained thành công
~~~

Trạng thái đã đạt ngày 2026-09-26:

~~~text
[ĐẠT] import lavis và đăng ký Blip2QuRe trong Conda VAIC
[ĐẠT] route model dùng blip2_qure / pretrain_vitL
[ĐẠT] config BLIP2 trỏ tới checkpoint local
[ĐẠT] BLIP2 và CLIP-L checkpoint đã kiểm tra không rỗng
[ĐẠT] load model thành công trên CPU
[ĐẠT] query_tokens có shape [1, 32, 768]
[ĐẠT] Vision Encoder có num_features = 1024
~~~

Các file đã thay đổi trong checkpoint này:

~~~text
lavis/models/__init__.py
lavis/models/clip_vit.py
lavis/configs/models/blip2/blip2_pretrain_vitL.yaml
~~~

Chưa inject LoRA và chưa chạy Checkpoint 1.

### Checkpoint triển khai 1: kiểm tra BLIP2 pretrained gốc

Chưa inject LoRA. Chạy một batch nhỏ với model QuRe gốc.

Kiểm tra:

~~~text
reference output có shape đúng
positive output có shape đúng
negative output có shape đúng
query vector có kích thước 256
candidate features có shape [B, 32, 256]
score chạy được
~~~

Mục tiêu là bảo đảm lỗi không đến từ BLIP2 hoặc dữ liệu trước khi thêm LoRA.

Trạng thái đã đạt ngày 2026-09-30:

~~~text
[ĐẠT] load model QuRe gốc trong Conda VAIC trên CPU
[ĐẠT] xác nhận chưa có lora_A/lora_B trước khi inject
[ĐẠT] reference Q-Former tokens có shape [1, 32, 768]
[ĐẠT] query vector có shape [1, 256]
[ĐẠT] positive features có shape [1, 32, 256]
[ĐẠT] negative features có shape [1, 32, 256]
[ĐẠT] scores có shape [1, 2]
[ĐẠT] score tái dựng theo forward gốc có sai số tuyệt đối lớn nhất 0.0
~~~

Kiểm tra dùng batch nhân tạo kích thước 1 để xác nhận luồng tính toán và shape;
đây không phải đánh giá chất lượng trên FashionIQ/CIRR. Không có source code
được sửa trong checkpoint này; script kiểm tra tạm thời đã được xóa.

### Checkpoint triển khai 2: kiểm tra cấu trúc adapter

Sau configure_lora():

~~~text
chỉ có một model.Qformer
chỉ query/value của self-attention và cross-attention là target
mỗi target Linear có một W0
mỗi target Linear có 2 bộ A/B
Wk, Wo và toàn bộ FFN không có lora_A/lora_B
có đúng tên adapter reference/candidate
merge_weights=False
~~~

Kiểm tra không có layer nào bị inject hai lần.

Trạng thái đã đạt ngày 2026-09-30:

~~~text
[ĐẠT] model chỉ có một Q-Former và một query_tokens
[ĐẠT] đã inject 36 MultiAdapterLinear vào Q-Former
[ĐẠT] self-attention: 12 query + 12 value
[ĐẠT] cross-attention: 6 query + 6 value
[ĐẠT] mỗi target Linear có một W0 và hai cặp A/B
[ĐẠT] adapter names đúng là reference và candidate
[ĐẠT] Wk, Wo và FFN không bị inject
[ĐẠT] merge_weights=False
[ĐẠT] gọi configure_lora() lần thứ hai bị chặn
~~~

Các file đã tạo hoặc sửa trong checkpoint này:

~~~text
lora/__init__.py
lora/multi_adapter.py
lora/utils.py
lavis/models/blip2_models/blip2_qure.py
~~~

Checkpoint này mới xác nhận cấu trúc injection; việc audit toàn bộ
`requires_grad`, gradient sau backward và trainability là nội dung của
Checkpoint 3 và chưa được kết luận ở đây.

### Checkpoint triển khai 3: kiểm tra freeze/trainable

Assert bắt buộc:

~~~python
assert not base_weight.requires_grad
assert not base_bias.requires_grad
assert not qformer_layernorm.requires_grad
assert model.query_tokens.requires_grad
assert model.ln_vision.weight.requires_grad
assert model.ln_vision.bias.requires_grad
assert reference_A.requires_grad
assert reference_B.requires_grad
assert candidate_A.requires_grad
assert candidate_B.requires_grad
~~~

Đồng thời kiểm tra Vision Encoder đã frozen và `ln_vision` vẫn trainable theo
đúng code QuRe gốc.

Trạng thái đã đạt ngày 2026-09-30:

~~~text
[ĐẠT] 333 tham số tensor nền của Q-Former bị frozen
[ĐẠT] 144 tensor LoRA A/B được kiểm tra là trainable
[ĐẠT] query_tokens trainable
[ĐẠT] ln_vision.weight và ln_vision.bias trainable
[ĐẠT] text_proj, vision_proj và temp trainable
[ĐẠT] Vision Encoder frozen
[ĐẠT] Q-Former W0, bias, LayerNorm và FFN frozen
[ĐẠT] itm_head frozen/không nằm trong nhóm trainable
[ĐẠT] tổng số tensor trainable được audit: 152
[ĐẠT] không có lỗi trong danh sách trainable ngoài LoRA
~~~

Checkpoint này chỉ kiểm tra cờ `requires_grad`; chưa gọi `loss.backward()` và
chưa xác nhận gradient thực tế. Hai nội dung đó thuộc Checkpoint 6.

### Checkpoint triển khai 4: kiểm tra output trước train

Vì tất cả B được khởi tạo zero:

~~~text
output với reference adapter ≈ output Q-Former gốc
output với candidate adapter ≈ output Q-Former gốc
~~~

Sai số chỉ được phép ở mức sai số số học nhỏ. Nếu output khác lớn ngay từ đầu,
cần kiểm tra copy weight, bias, dtype, device hoặc dropout.

Trạng thái đã đạt ngày 2026-09-30:

~~~text
[ĐẠT] 36 wrapper được kiểm tra
[ĐẠT] tất cả B_reference và B_candidate khởi tạo bằng zero
[ĐẠT] sai số copy W0 lớn nhất: 0.0
[ĐẠT] sai số output với candidate adapter lớn nhất: 0.0
[ĐẠT] sai số output với reference adapter lớn nhất: 0.0
~~~

Kiểm tra dùng cùng một model, cùng một input và ở chế độ eval. Vì vậy kết quả
xác nhận việc bọc Linear không làm thay đổi output khi delta LoRA bằng zero.
Chưa kiểm tra routing giữa các nhánh; nội dung đó thuộc Checkpoint 5.

### Checkpoint triển khai 5: kiểm tra routing

Dùng hook hoặc log adapter active để xác nhận:

~~~text
_encode_reference() → reference
positive encoding   → candidate
negative encoding   → candidate
extract query       → reference
extract gallery     → candidate
~~~

Đặc biệt phải kiểm tra cả các đường extract_query_features_* và
extract_target_features, không chỉ forward().

Trạng thái đã đạt ngày 2026-09-30:

~~~text
[ĐẠT] forward reference dùng adapter reference
[ĐẠT] forward positive dùng adapter candidate
[ĐẠT] forward negative dùng adapter candidate
[ĐẠT] gallery extraction dùng adapter candidate
[ĐẠT] FashionIQ query extraction dùng adapter reference
[ĐẠT] CIRR query extraction dùng adapter reference
[ĐẠT] CIRR validation query extraction dùng adapter reference
[ĐẠT] FashionIQ target-image extraction dùng adapter candidate
[ĐẠT] adapter context khôi phục về candidate sau mỗi lần gọi
[ĐẠT] forward score vẫn có shape [1, 2]
~~~

`extract_query_features_circo()` không nằm trong kiểm tra này vì CIRCO đã được
đặt ngoài phạm vi triển khai. Các đường FashionIQ/CIRR và gallery trong phạm vi
kế hoạch đã được hook kiểm tra trực tiếp.

### Checkpoint triển khai 6: kiểm tra gradient

Chạy một batch và gọi loss.backward().

Kỳ vọng:

~~~text
query_tokens             có gradient
reference A/B             có gradient
candidate A/B             có gradient
ln_vision                 có gradient
text_proj/vision_proj     có gradient
temp                     có gradient
Q-Former W0               không có gradient
Q-Former bias/LayerNorm   không có gradient
Vision Encoder            không có gradient
~~~

Cần xác nhận cả positive và negative thực sự đóng góp gradient cho
LoRA_candidate.

Lưu ý về khởi tạo LoRA: vì `B_reference` và `B_candidate` được khởi tạo bằng
zero để giữ output ban đầu giống Q-Former gốc, backward đầu tiên có thể cho
gradient bằng zero ở các ma trận A. Đây là hành vi toán học bình thường của
LoRA; các ma trận B nhận gradient trước, sau bước optimizer đầu tiên thì A và B
đều phải nhận gradient.

Trạng thái đã đạt ngày 2026-09-30:

~~~text
[ĐẠT] loss KL của QuRe chạy được và backward thành công
[ĐẠT] query_tokens có gradient
[ĐẠT] ln_vision, text_proj, vision_proj và temp có gradient
[ĐẠT] B_reference và B_candidate có gradient ở backward đầu tiên
[ĐẠT] LoRA_candidate nhận gradient riêng từ positive score
[ĐẠT] LoRA_candidate nhận gradient riêng từ negative score
[ĐẠT] sau một bước optimizer, A_reference/B_reference đều có gradient
[ĐẠT] sau một bước optimizer, A_candidate/B_candidate đều có gradient
[ĐẠT] Q-Former W0, bias, LayerNorm và FFN không có gradient
[ĐẠT] Vision Encoder không có gradient
~~~

Checkpoint này xác nhận đường truyền gradient; chưa đánh giá loss giảm qua
nhiều bước hoặc khả năng overfit, là nội dung của Checkpoint 7.

### Checkpoint triển khai 7: overfit một batch nhỏ

Chạy vài chục đến vài trăm bước trên một batch cố định.

Kỳ vọng:

~~~text
loss giảm
positive score tăng tương đối
negative score giảm tương đối
adapter parameters thay đổi
base W0 không thay đổi
query_tokens thay đổi
~~~

Nếu loss không giảm, trước tiên kiểm tra routing và gradient, chưa thay đổi
rank hoặc learning rate ngay.

Trạng thái smoke test đã đạt ngày 2026-09-30:

~~~text
[ĐẠT-SMOKE] loss giảm trên batch cố định
[ĐẠT-SMOKE] positive score tăng
[ĐẠT-SMOKE] negative score giảm
[ĐẠT-SMOKE] adapter parameters thay đổi
[ĐẠT-SMOKE] base W0 không thay đổi
[ĐẠT-SMOKE] query_tokens thay đổi
~~~

Smoke test dùng output đã cache của Vision Encoder frozen để tránh chạy
backward nặng trên CPU; phần trainable Q-Former, hai adapter, query_tokens và
các head vẫn chạy thật. Full training/evaluation trên toàn bộ dataset được
defer sang Kaggle GPU và chưa được coi là hoàn thành ở checkpoint này.

### Checkpoint triển khai 8: save/load round-trip

Quy trình:

~~~text
model A train vài bước
→ lưu deployment checkpoint
→ tạo model B từ BLIP2 pretrained
→ inject LoRA
→ load checkpoint
→ chạy cùng một input
~~~

Kỳ vọng:

~~~text
query output A ≈ query output B
gallery output A ≈ gallery output B
score A ≈ score B
~~~

Trạng thái đã đạt ngày 2026-09-30:

~~~text
[ĐẠT] deployment checkpoint được tạo và có kích thước 5,372,513 bytes
[ĐẠT] checkpoint chứa 144 tensor LoRA
[ĐẠT] checkpoint chứa 8 tensor trainable ngoài LoRA
[ĐẠT] model B load checkpoint không có unexpected key
[ĐẠT] LoRA state sau load khớp model A với sai số lớn nhất 0.0
[ĐẠT] trainable state sau load khớp model A với sai số lớn nhất 0.0
[ĐẠT] query output A/B khớp với sai số 0.0
[ĐẠT] gallery output A/B khớp với sai số 0.0
[ĐẠT] score A/B khớp với sai số 0.0
~~~

Để không phải train trên CPU, model A được gán state LoRA/trainable khác zero
theo giá trị cố định trước khi lưu; đây là kiểm tra serialization/inference,
không phải checkpoint đã train. Checkpoint huấn luyện thật sẽ được tạo trên
Kaggle GPU.

Đây là checkpoint bắt buộc trước khi chạy evaluation lớn.

### Checkpoint triển khai 9: resume training

Kiểm tra:

~~~text
load model adapter
load optimizer
load scheduler
load AMP scaler nếu có
load epoch/global_step
load RNG nếu cần tái lập
~~~

Sau resume, learning rate và loss phải tiếp tục từ trạng thái cũ, không quay về
epoch 0 hoặc khởi tạo optimizer mới.

Trạng thái đã đạt ngày 2026-09-30:

~~~text
[ĐẠT] save/load lora_state_dict và trainable_state_dict
[ĐẠT] load optimizer state
[ĐẠT] load scheduler state
[ĐẠT] load AMP scaler state
[ĐẠT] load epoch=4 và global_step=123 trong probe
[ĐẠT] khôi phục RNG state
[ĐẠT] sai số state sau resume: 0.0
~~~

Probe CP9 chỉ kiểm tra serialization/resume trên CPU, không chạy training.
Full 30 epoch vẫn dành cho Kaggle GPU.

### Checkpoint triển khai 10: evaluation thật

Chạy lần lượt:

~~~text
FashionIQ
CIRR
~~~

Lưu lại:

~~~text
config sử dụng
base checkpoint identifier
LoRA checkpoint
metric
bản source/commit
~~~

Trạng thái chuẩn bị ngày 2026-09-30:

~~~text
[SẴN SÀNG] train_qure.py tạo last.pt, best.pt và epoch_NNN.pt
[SẴN SÀNG] evaluate_fiq.py đọc checkpoint LoRA bằng load_checkpoint()
[SẴN SÀNG] FashionIQ root được truyền qua fashioniq_root
[CHƯA CHẠY] evaluation thật; cần checkpoint 30 epoch và dữ liệu FashionIQ trên Kaggle
~~~

---

## 7. Các lỗi cần tránh

### Không tạo adapter theo positive/negative

Sai:

~~~text
reference adapter
positive adapter
negative adapter
~~~

Đúng:

~~~text
reference adapter
candidate adapter
~~~

### Không dùng loralib.Linear nguyên bản cho toàn bộ bài toán

loralib.Linear thông thường chỉ có một cặp A/B. Nó không đủ để chứa đồng thời
hai bộ A/B trong cùng một Linear. Có thể dùng làm tài liệu tham khảo, nhưng phải
tạo MultiAdapterLinear nội bộ.

### Không merge adapter trong eval

Nếu merge LoRA vào W0 khi model.eval() thì sau đó chuyển sang adapter còn lại
sẽ làm sai base weight dùng chung. merge_weights=False là bắt buộc.

### Không dùng default adapter ngầm định trong encode

Mỗi nhánh phải có context rõ ràng. Không được dựa vào việc adapter cuối cùng
được set là reference hay candidate.

### Không chỉ sửa forward()

Nếu chỉ sửa training forward mà không sửa các hàm extract_*, training và
inference sẽ dùng hai routing khác nhau.

### Không lưu full pretrained model trong adapter checkpoint

Deployment checkpoint chỉ nên chứa LoRA và các tham số trainable ngoài LoRA.
Base BLIP2 phải được nạp từ pretrained checkpoint đã xác định.

---

## 8. Tiêu chí hoàn thành

Triển khai được coi là hoàn thành khi đồng thời đạt các điều kiện:

1. Có đúng một Qformer và một query_tokens.
2. Chỉ self/cross-attention `query` và `value` là target LoRA.
3. Tất cả target Linear có chung W0 frozen.
4. Mỗi target Linear có đúng hai adapter reference và candidate.
5. Wk, Wo, FFN, LayerNorm và Embedding của Q-Former không có LoRA.
6. query_tokens nhận gradient từ reference, positive và negative.
7. LoRA_reference chỉ nhận gradient từ reference.
8. LoRA_candidate nhận gradient từ positive và negative.
9. Gallery inference dùng LoRA_candidate.
10. Query inference dùng LoRA_reference.
11. BLIP2 pretrained được load trước khi inject LoRA.
12. Save/load checkpoint cho output round-trip giống nhau.
13. Base W0 không thay đổi trong quá trình train.
14. Loss QuRe và công thức score không bị thay đổi ngoài routing adapter.
15. Training và evaluation không phụ thuộc trực tiếp vào D:\NCKH\LoRa\LoRA
    hoặc D:\NCKH\QuRe.

---

## 9. Thứ tự triển khai thực tế

Thứ tự thực hiện được chốt là:

~~~text
Bước 1  tạo lora/multi_adapter.py và lora/utils.py
Bước 2  viết test cho MultiAdapterLinear độc lập
Bước 3  tích hợp configure_lora() vào Blip2QuRe
Bước 4  cập nhật routing forward/extract
Bước 5  cập nhật model builder và config
Bước 6  chạy Checkpoint 0–4
Bước 7  sửa train_qure.py và cơ chế save checkpoint
Bước 8  chạy Checkpoint 5–8
Bước 9  cập nhật evaluation
Bước 10 chạy Checkpoint 9–10
~~~

Chỉ sau khi Checkpoint 8 về save/load round-trip đạt mới chạy training toàn bộ
dataset. Đây là điều kiện quan trọng để tránh trường hợp training thành công
nhưng evaluation lại dùng nhầm adapter hoặc không load được checkpoint.
