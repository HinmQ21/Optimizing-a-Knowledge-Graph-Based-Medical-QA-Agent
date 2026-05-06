# Medical QA Agent — Knowledge-Graph-Augmented Reinforcement Learning

> **Tối ưu hóa Tác tử Hỏi đáp Y tế Dựa trên Đồ thị Tri thức bằng Học Tăng cường**
>
> Mô hình nền: **Qwen2.5-3B-Instruct** · KG nguồn: **PrimeKG (Harvard)** · Thuật toán RL: **GRPO** · Tăng tốc: **TRL + vLLM colocate**

Pipeline 3 giai đoạn huấn luyện một LLM 3B trở thành "tác tử y tế" biết **chủ động gọi công cụ** truy vấn Hyper Knowledge Graph để trả lời câu hỏi MCQ kiểu USMLE.

---

## Mục lục

- [1. Tổng quan](#1-tổng-quan)
- [2. Kiến trúc pipeline](#2-kiến-trúc-pipeline)
- [3. Hyper Knowledge Graph](#3-hyper-knowledge-graph)
- [4. Stage 1 — SFT Reasoning](#4-stage-1--sft-reasoning)
- [5. Stage 1.5 — Tool-Calling SFT](#5-stage-15--tool-calling-sft)
- [6. Stage 2 — GRPO RL](#6-stage-2--grpo-rl)
- [7. Tăng tốc huấn luyện với vLLM](#7-tăng-tốc-huấn-luyện-với-vllm)
- [8. Cơ chế Retrieval Tool](#8-cơ-chế-retrieval-tool)
- [9. Stack công nghệ](#9-stack-công-nghệ)

---

## 1. Tổng quan

### Vấn đề

Các LLM tổng quát khi áp dụng cho y tế gặp hai nhược điểm cốt lõi:

1. **Hallucination** — sinh ra fact y tế sai do kiến thức nội tại lỗi thời.
2. **Thiếu chứng cứ** — câu trả lời không có nguồn dẫn từ tri thức được kiểm chứng.

### Giải pháp

Pipeline 3 giai đoạn nối tiếp nhau, mỗi giai đoạn build trên checkpoint của giai đoạn trước:

| Stage | Phương pháp | Mục tiêu |
|-------|-------------|----------|
| Stage 1 | Full fine-tuning với MedReason | Học suy luận có cấu trúc `<think>...</think><answer>X</answer>` |
| Stage 1.5 | LoRA SFT với teacher traces gọi KG thật | Học **format gọi tool** trước khi RL |
| Stage 2 | LoRA GRPO với reward đa thành phần | Tối ưu **chính sách** gọi tool để tăng accuracy |

Khâu **Stage 1.5 đặc biệt quan trọng**: nếu vào thẳng GRPO mà không qua nó, mô hình bắt đầu với tool-call rate 0% và RL không có signal khác biệt giữa "có/không tool" để học.

### Bài toán cụ thể

- **Input:** Câu hỏi MCQ y tế + 4 đáp án (A/B/C/D)
- **Output:** Letter A/B/C/D + reasoning chain (có thể chèn tool call ở giữa)

---

## 2. Kiến trúc pipeline

```
PrimeKG (4.05M cạnh, 129K nút)
      │
      ▼
┌────────────────────┐
│  BƯỚC 0            │  Hyper KG Build (5 bước, không LLM)
│  Hyper KG          │  → medical_hg.json (54K entity, 81.8K hyperedge)
│                    │  → index_hyperedge.bin (FAISS, 320 MB)
│                    │  → index_entity.bin   (FAISS, 211 MB)
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  STAGE 1           │  Full FT — MedReason (~200K mẫu)
│  SFT Reasoning     │  Qwen2.5-3B-Instruct, lr=5e-6, 3 epochs, bf16
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  STAGE 1.5         │  LoRA SFT — Teacher traces gọi KG thật
│  Tool-Calling SFT  │  3K raw → 4,339 augmented
│                    │  r=32, α=32, lr=2e-5, 3 epochs
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  STAGE 2           │  LoRA GRPO — MedQA train
│  RL + Tool Calling │  r=32, α=64, lr=5e-5, β=0.1, G=4–8, temp=0.8
│                    │  Reward = 0.25·format + 0.50·answer + 0.25·tool
│                    │  Tăng tốc: TRL + vLLM colocate (~3–5× wall time)
└────────────────────┘
```

Nguyên tắc xuyên suốt:

- **Special tokens** `<think>`, `</think>`, `<answer>`, `</answer>` được add vào tokenizer ở mọi stage.
- **Loss masking:** chỉ train trên phần assistant; mask system prompt, user, tool response.
- **MedEmbed encoder** chạy CPU để giải phóng VRAM cho training GPU.
- **Singleton tool**: nạp FAISS index một lần, tái dùng trong SFT eval, GRPO rollout, reward computation.

---

## 3. Hyper Knowledge Graph

### 3.1 Nguồn: PrimeKG

PrimeKG (Harvard, 2022) — đồ thị tri thức y sinh học từ 20 cơ sở dữ liệu (DrugBank, OMIM, GO, Reactome, UniProt…), 10 loại thực thể (disease, drug, gene/protein, biological_process, pathway, phenotype, exposure, anatomy, …).

### 3.2 Pipeline 5 bước build KG

```
PrimeKG/kg.csv  (4.05M cạnh)
   ↓ filter.py        — giữ 18 quan hệ lâm sàng (3 tier)  →  ~830K cạnh
   ↓ aggregate.py     — gom thành siêu cạnh (3 chiến lược)
   ↓ verbalize.py     — template-based, không LLM
   ↓ store.py         — medical_hg.json
   ↓ embed.py         — MedEmbed-large (1024-dim) → FAISS IndexFlatIP
medical_hg.json + index_hyperedge.bin + index_entity.bin
```

Toàn bộ pipeline **không dùng LLM** ⇒ tính xác định, không tốn chi phí API, dễ rebuild.

### 3.3 Ba chiến lược gom siêu cạnh

| Chiến lược | Mô tả | Số lượng | Tỉ lệ |
|------------|-------|---------:|------:|
| `neighbor_agg` | Gom anchor + 3–9 hàng xóm cùng quan hệ | 46,887 | 57.3% |
| `path` | Chuỗi 2–3 hop theo 5 mẫu y học (`symptom→disease→drug`, `disease→protein→drug`, …) | 25,000 | 30.6% |
| `composite` | Kết hợp 2–3 quan hệ khác nhau của cùng anchor | 9,904 | 12.1% |
| **Tổng** | | **81,791** | 100% |

**Ví dụ siêu cạnh `neighbor_agg`:**
```
anchor: "Type 2 Diabetes"
relation: "disease_phenotype_positive"
entities: [Type 2 Diabetes, Polyuria, Polydipsia, Fatigue, Blurred vision, Weight loss]
description: "Type 2 Diabetes characteristically presents with Polyuria,
              Polydipsia, Fatigue, Blurred vision and Weight loss"
```

### 3.4 Verbalization template

~100 mẫu cho 18 loại quan hệ × 3–4 mẫu/quan hệ. Template engine sinh văn bản sạch (0 placeholder leak, 0 orphan, < 0.003% duplicate).

```python
TEMPLATES = {
  "disease_phenotype_positive": [
    "{anchor} characteristically presents with {nb}",
    "{anchor} is clinically associated with {nb}",
    "Patients with {anchor} typically exhibit {nb}",
  ],
  "indication": [
    "{anchor} is treated with {nb}",
    "{nb} is the first-line treatment for {anchor}",
  ],
  "contraindication": [
    "{anchor} is contraindicated in {nb}",
    "{nb} represents a contraindication for {anchor}",
  ],
  # ... 15 quan hệ khác
}
```

### 3.5 FAISS dual-index

| Index | Số vector | Dim | Nội dung |
|-------|---------:|----:|----------|
| `index_hyperedge.bin` | 81,791 | 1024 | Embedding mô tả siêu cạnh |
| `index_entity.bin` | 54,000 | 1024 | Embedding tên thực thể |

Cả hai đều `IndexFlatIP` (inner product trên vector chuẩn hoá L2 = cosine). Encoder: `abhinand/MedEmbed-large-v0.1` chạy CPU.

---

## 4. Stage 1 — SFT Reasoning

| Tham số | Giá trị |
|---------|---------|
| Base | `Qwen/Qwen2.5-3B-Instruct` |
| Dataset | `UCSC-VLAA/MedReason` (~200K mẫu) |
| Mode | **Full fine-tuning** (không LoRA) |
| Max seq len | 8,192 |
| Effective batch | 16 (1 × 16 grad acc) |
| LR | 5e-6, cosine, warmup 5% |
| Epochs | 3 |
| Optimizer | AdamW (bf16) |
| Output format | `<think>…</think><answer>X</answer>` |

Stage 1 chỉ dạy mô hình **suy luận có cấu trúc**, chưa biết tới tool — đây là tầng "kiến thức nền".

---

## 5. Stage 1.5 — Tool-Calling SFT

### 5.1 Vì sao cần?

GRPO không tự dạy được mô hình **gọi tool đúng format** chỉ qua reward — phải có cold-start. Nếu vào thẳng Stage 2:

- Tool-call rate ban đầu = 0% (mô hình chưa từng gọi tool)
- Không có rollout nào có tool → không có signal khác biệt
- GRPO collapse về policy "không gọi tool"

Stage 1.5 dạy **format**, GRPO dạy **chính sách** (khi nào nên / không nên gọi).

### 5.2 Pipeline dữ liệu

```
gen_data_groq.py     Teacher (GPT lớn qua Groq/Cerebras API) được cấp quyền gọi
                     KG THẬT (FAISS local). Mỗi câu hỏi → trace đầy đủ
                     (think + tool_call + tool_response + answer)
                     → 3,000 raw traces

prepare_sft_data.py  3 chiến lược augmentation:
                     - Split multi-call traces → các trace độc lập
                     - Verbose variants
                     - No-tool samples (tránh over-trigger tool)
                     → 4,339 mẫu, split 95/5 train/val

sft_train.py         LoRA SFT trên Stage 1 checkpoint
                     - Loss masking: chỉ assistant
                     - Special tokens add vào tokenizer

merge_peft_adapter   Merge LoRA → dense weights để Stage 2 load làm base
```

**Điểm cốt yếu:** Teacher gọi **KG thật**, không phải tool giả tạo → distribution của Stage 1.5 trùng khớp environment Stage 2 (zero distribution gap).

### 5.3 Hyperparameters

| Tham số | Giá trị |
|---------|---------|
| Adapter | LoRA r=32, α=32, dropout=0.05 |
| Target modules | q,k,v,o,up,down_proj |
| Loss masking | Assistant-only (mask system + user + tool_response) |
| LR | 2e-5, cosine, warmup 5% |
| Effective batch | 16 (4 device × 4 grad acc) |
| Epochs | 3 |

### 5.4 Định dạng SFT

```
[SYSTEM] You are a medical reasoning assistant with access to search_medical_knowledge tool.

[USER] Question: A 65-year-old patient with HbSS disease presents...
       Options: A. ... B. ... C. ... D. ...

[ASSISTANT]
<think>This question asks about sickle cell disease... let me search.</think>
<tool_call>{"name": "search_medical_knowledge",
            "arguments": {"query": "sickle cell disease HbSS hemoglobin electrophoresis"}}</tool_call>
<tool_response>- HbSS shows predominantly HbS with HbF on electrophoresis...
               - HbA is absent in homozygous sickle cell disease...</tool_response>
<think>Based on the KG results, HbSS shows HbS + HbF, no HbA...</think>
<answer>C</answer>
```

---

## 6. Stage 2 — GRPO RL

### 6.1 Group Relative Policy Optimization

```
PPO:   A = r − V(s)              cần value/critic network
GRPO:  A_i = (r_i − μ_r) / σ_r    so sánh tương đối trong nhóm G rollouts
```

Không cần critic ⇒ **tiết kiệm bộ nhớ đáng kể**. Objective:

```
L_GRPO = E[ A_i · log π_θ(o_i | q) ] − β · KL(π_θ ‖ π_ref)
```

với `π_ref` = Stage 1.5 (frozen), `β = 0.1`.

### 6.2 Cấu hình

| Tham số | Giá trị |
|---------|---------|
| Library | TRL `GRPOTrainer` v0.29.1 |
| Adapter | LoRA r=32, α=64 |
| KL β | 0.1 |
| PPO clip ε | 0.2 |
| Generations G | 4 (vanilla) hoặc 8–16 (vLLM colocate) |
| Max completion | 2,048 tokens |
| Temperature | 0.8 |
| Max tool iter | 3 |
| LR | 5e-5 |
| Effective batch | 16 (2 × 8 grad acc) |
| Warmup | 3% |
| Epochs | 3 |
| Attention | SDPA (PyTorch native) |

### 6.3 Hàm thưởng 3 thành phần

```python
r_total = 0.25 * r_format + 0.50 * r_answer + 0.25 * r_tool
```

Trọng số được hardcode trong cả `grpo_train.py` và `reward_fns.py` — phải sync khi đổi.

#### Format reward [0, 1] — trọng số 0.25
- `+0.25` nếu có cặp `<tool_call>` & `<tool_response>` hợp lệ
- `+0.25` nếu có `<think>...</think>`
- `+0.50` nếu có `<answer>...</answer>` (+0.25 bonus nếu answer ≤ 10 từ)

#### Answer reward [0, 1] — trọng số 0.50

Hierarchical matching:
```
exact match            → 1.0
letter match (A/B/C/D) → 1.0
letter-in-text         → 0.8
substring match        → 0.5
token-F1 overlap       → ×0.3
```

#### Enhanced Tool Quality reward [-0.30, +0.40] — trọng số 0.25

Phiên bản frequency-only ban đầu chỉ đếm số lần gọi tool, không phân biệt **query có relevant không** và **kết quả có chất lượng không**. Enhanced reward bổ sung 3 semantic signal:

| Thành phần | Range | Cơ chế |
|------------|------:|--------|
| **Base** (n_calls) | -0.30 → +0.10 | n=0 → -0.30 (collapse prevention); n∈{1,2} → +0.10; n>2 → +0.05 |
| **Signal 1 — Query relevance** | 0 → +0.10 | `cos(question, query)`; giảm dần khi cosine > 0.80 (chống copy-paste nguyên câu hỏi) |
| **Signal 2 — Retrieval quality** | 0 → +0.15 | `cos("Q + answer is GT", tool_response)` — đo trực tiếp xem KG có retrieve được fact liên quan đáp án đúng không |
| **Signal 3 — Grounding** | {0, +0.05} | Binary: post-tool `<think>` có ≥3 medical token trùng tool response (không phải stopword) |

**Tối ưu encode:** toàn bộ text cần nhúng (question, q+a anchor, queries, tool responses) trong cả batch rollout được gom lại và encode 1 lần với `batch_size=64`. Encoder dùng lại singleton `MedicalKnowledgeTool._instance` (MedEmbed-large CPU) — **không tốn thêm VRAM**.

### 6.4 Vòng lặp GRPO + Tool Calling

```
Mỗi prompt q  →  G rollouts song song
  Rollout i:
    1. <think>...</think>
    2. <tool_call>{"name": "search_medical_knowledge",
                    "arguments": {"query": "..."}}</tool_call>
    3. MedicalKnowledgeTool.retrieve()
       → <tool_response>top-5 hyperedge descriptions</tool_response>
    4. <think>...</think>     # đọc kết quả
    5. <answer>X</answer>
    6. r_i = 0.25 r_format + 0.50 r_answer + 0.25 r_tool

A_i = (r_i − μ_r) / (σ_r + 1e-8)
loss = − mean(A_i · log π_θ) + β · KL(π_θ ‖ π_ref)
θ ← θ − ∇loss
```

`max_tool_iter = 3` cho phép tối đa 3 vòng lặp tool/rollout.

---

## 7. Tăng tốc huấn luyện với vLLM

GRPO có **70–80% wall-time dành cho rollout generation**. HF `model.generate()` xử lý tuần tự từng token và không tận dụng được continuous batching ⇒ rất chậm khi G lớn. Giải pháp: dùng **TRL + vLLM colocate mode** (`grpo_train_vllm.py`).

### 7.1 Cơ chế colocate

```
Cùng một process Python:
┌─────────────────────────────────────────────────────┐
│  Training model (HF, bf16) + LoRA + optimizer states │  ~35 GB
│                          ↕  weight sync mỗi step     │
│  vLLM engine (base weights + KV cache)               │  ~66 GB
└─────────────────────────────────────────────────────┘
```

Trước **mỗi optimizer step:**

1. TRL **merge LoRA weights vào base** trong RAM
2. **Sync** sang vLLM engine (cùng process, không qua disk/network)
3. vLLM generate **G rollouts/prompt** trong **một batched call** — continuous batching, paged KV cache
4. Tool calling (`_tool_call_loop`) được giữ nguyên — vLLM colocate hỗ trợ tool, server mode thì không

Kỳ vọng: rollout phase nhanh **10–30×** ⇒ tổng wall-time GRPO giảm **~3–5×** (vài ngày → ~1–2 ngày).

### 7.2 Memory profile (GB10 120 GB unified, sleep mode OFF)

| Thành phần | VRAM |
|------------|-----:|
| Training model + LoRA + optimizer states | ~35 GB |
| vLLM engine (base weights + KV cache, util=0.5) | ~66 GB |
| **Peak đồng thời** | **~101 GB** (vừa với 120 GB) |

`enable_sleep_mode` (offload vLLM weights ra CPU/disk khi backward) trên GB10 **gây chậm hơn baseline** vì vLLM 0.20.0 reload từ safetensors mỗi `wake_up()` (~34s/step). Trên unified memory 120 GB, **sleep mode không cần thiết**.

### 7.3 Cờ điều khiển chính

| Flag | Ý nghĩa |
|------|---------|
| `--use-vllm` | Bật vLLM colocate (không bật → fallback HF generate, hành vi như `grpo_train.py`) |
| `--vllm-gpu-mem-util 0.5` | Tỷ lệ VRAM dành cho vLLM engine (KV cache + weights) |
| `--vllm-sleep-mode` | Offload vLLM khi backward (default OFF) |
| `--vllm-model-impl` | `transformers` (TRL ép) hoặc native vLLM |
| `--num-generations` | G rollouts/prompt (4 vanilla; vLLM cho phép 8–16) |

### 7.4 Lưu ý vận hành

- **Hai venv tách biệt:** `training_venv312/` cho HF-only training, `vllm_venv312/` cho mọi script có vLLM (training & eval). Không trộn — TRL + vLLM cần torch/CUDA build khớp.
- **TRL version warning:** TRL 0.29.1 list vLLM 0.10–0.12 nhưng nội bộ dùng V1 API có từ 0.6+; **vLLM 0.20.0 vẫn tương thích** — script đã filter warning.
- **Tool calling chỉ chạy được với `vllm_mode="colocate"`** — `vllm_mode="server"` raise `NotImplementedError` khi prompt có tools.
- **vLLM eval (cho benchmark) khác vLLM training:** eval nhanh hơn ~30× nhưng AnswerRate giảm ~6 pp do sampler khác — chỉ dùng cho dev iter, không dùng số final.

---

## 8. Cơ chế Retrieval Tool

### 8.1 Tool schema (Qwen ChatML JSON)

```json
{
  "type": "function",
  "function": {
    "name": "search_medical_knowledge",
    "description": "Search the medical knowledge graph for relevant clinical facts, drug interactions, disease mechanisms, symptoms, diagnoses, and treatments.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "A medical search query. Be specific and clinical."
        }
      },
      "required": ["query"]
    }
  }
}
```

### 8.2 Quy trình retrieval (dual FAISS + lexical re-rank)

```
query  →  MedEmbed-large (1024-dim, L2-normalized vector)
       ↓
   ┌───── FAISS A: index_hyperedge → top-40 ứng viên (cosine)
   │
   └───── FAISS B: index_entity → top-12 entity
                     → mỗi entity: top-8 hedge liên quan  (≤96 ứng viên)
       ↓
   Tổng hợp + dedup
       ↓
   score = α · semantic_cosine
         + β · lexical_overlap   (token, đã loại stopwords)
         + γ · entity_match      (query tokens vs hedge entities)
         + δ · anchor_bonus      (query entity = anchor của hedge)
       ↓
   Top-5 description trả về model
```

### 8.3 Singleton pattern

```python
class MedicalKnowledgeTool:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load_indices()      # nạp 1 lần
        return cls._instance

    def load_indices(self):
        self.hg            = load_hypergraph("data/medical_hg.json")
        self.hedge_index   = faiss.read_index("data/index_hyperedge.bin")
        self.entity_index  = faiss.read_index("data/index_entity.bin")
        self.encoder       = SentenceTransformer("abhinand/MedEmbed-large-v0.1",
                                                 device="cpu")
```

Index nạp **một lần duy nhất**, tái dùng cho:
- Stage 1.5 teacher trace generation
- GRPO rollout (`_tool_call_loop`)
- Reward function (Signal 1, 2)
- Eval scripts

⇒ không double-load FAISS, không nuốt thêm VRAM.

---

## 9. Stack công nghệ

### 9.1 Mô hình & framework

| Tầng | Công cụ |
|------|---------|
| Base LLM | `Qwen/Qwen2.5-3B-Instruct` (chạy thử thêm `Llama-3.2-3B-Instruct`) |
| Embedding | `abhinand/MedEmbed-large-v0.1` (1024-dim, biomedical-tuned) |
| Vector index | **FAISS** `IndexFlatIP` (cosine trên L2-normalized) |
| Fine-tuning | **HF Transformers** + **PEFT (LoRA)** + **bitsandbytes** (bf16) |
| RL | **TRL** `GRPOTrainer` v0.29.1 |
| Inference accel | **vLLM** v0.20.0 colocate (paged KV cache, continuous batching) |
| Distributed | Single-node (DGX-Spark GB10 unified memory) |
| Logging | **Weights & Biases** |

### 9.2 KG sources

| Tầng | Công cụ |
|------|---------|
| Source KG | **PrimeKG** (Harvard, 2022): 4.05M cạnh, 129K nút, 30 quan hệ, 10 loại thực thể |
| Aggregation | Custom Python (filter / aggregate / verbalize / store / embed) |
| Verbalization | Template-based — không LLM, fully deterministic |

### 9.3 Datasets

| Mục đích | Dataset |
|----------|---------|
| Stage 1 SFT (reasoning) | `UCSC-VLAA/MedReason` |
| Stage 1.5 SFT (tool format) | Teacher traces tự sinh từ MedQA + MedMCQA train (Groq / Cerebras API) |
| Stage 2 GRPO | MedQA train |
| Eval | MedQA test (USMLE), MedMCQA, PubMedQA, MedXpertQA, BioMed-R1-Eval |

### 9.4 Cấu trúc kho code

```
scripts/
├── build_kg/         5 bước xây Hyper KG (filter → aggregate → verbalize → store → embed)
├── finetune/         Stage 1 SFT — full fine-tune (medreason, huatuo) + merge_peft_adapter
├── stage1_5/         Stage 1.5 — gen_data_groq, prepare_sft_data, sft_train, eval_sft
├── train_rl/         Stage 2 — grpo_train (HF), grpo_train_vllm, reward_fns, data_prep
├── serve/            retrieval_tool (singleton) + retrieval_api (FastAPI optional)
├── benchmark/        baseline / sft_eval / grpo_eval / embed_eval / teacher
├── analysis/         Token & dataset profiling
├── utils/            model_adapter (Qwen / Llama family detection)
└── setup/            Bash scripts cài venv (training & vllm)
```

### 9.5 Hai môi trường Python

| Venv | Dùng cho |
|------|----------|
| `training_venv312/` | Stage 1 / 1.5 SFT, GRPO không vLLM, eval HF |
| `vllm_venv312/` | GRPO + vLLM colocate, eval vLLM |

Tách biệt vì TRL+vLLM cần khớp chặt phiên bản torch/CUDA — không trộn dependencies.

---

## Tóm tắt một dòng

> Pipeline 3-stage (**Full FT reasoning → LoRA tool-SFT → LoRA GRPO + KG tool**) trên **Qwen2.5-3B**, tích hợp Hyper KG xây từ PrimeKG (81.8K hyperedge, dual FAISS retrieval) và tăng tốc bằng **TRL + vLLM colocate** giúp giảm wall-time GRPO ~3–5×.
