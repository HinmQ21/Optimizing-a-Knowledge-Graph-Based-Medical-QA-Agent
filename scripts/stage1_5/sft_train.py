"""Stage 1.5 SFT: teach tool-calling behaviour before GRPO.

Trains LoRA adapters on the Stage 1 checkpoint using multi-turn teacher traces
that include search_medical_knowledge tool calls.  The resulting checkpoint
becomes the base (and reference) model for Stage 2 GRPO.

Key design decisions aligned with GRPO Stage 2:
  - Same model, tokenizer, special tokens as grpo_train.py
  - Same system prompt / chat template (Qwen2.5 ChatML)
  - Assistant-only loss masking (prompt + tool-response tokens masked)
  - LoRA r=32 to match GRPO adapter rank
  - Conservative alpha=32 (scaling=1.0) to avoid overfitting small dataset

Usage:
    cd /home/vcsai/minhlbq/baseline
    ./training_venv312/bin/python -m scripts.stage1_5.sft_train \
        --model-path outputs/medreason_stage1_qwen25_3b_full_trainer_think_tag/checkpoint-1000 \
        --data-path data/stage1_5_sft.jsonl \
        --output-dir outputs/stage1_5_tool_sft \
        --num-train-epochs 3 \
        --report-to wandb --run-name stage1.5-tool-sft
"""

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import json


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1.5 SFT: tool-calling behaviour training",
    )

    # Model
    p.add_argument(
        "--model-path",
        default="outputs/medreason_stage1_qwen25_3b_full_trainer_think_tag/checkpoint-1000",
    )

    # Data
    p.add_argument("--data-path", default="data/stage1_5_sft.jsonl")
    p.add_argument("--eval-split", type=float, default=0.05,
                   help="Fraction of data for eval split.")
    p.add_argument("--max-seq-len", type=int, default=2048)

    # Output
    p.add_argument("--output-dir", default="outputs/stage1_5_tool_sft")

    # LoRA — conservative to avoid overfitting small dataset
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32,
                   help="Scaling=alpha/r=1.0. Conservative for 4K sample SFT.")
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # Training
    p.add_argument("--num-train-epochs", type=float, default=3.0)
    p.add_argument("--per-device-train-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--lr-scheduler-type", default="cosine")
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)

    # Logging / saving
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=70)
    p.add_argument("--eval-steps", type=int, default=70)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--report-to", default="none", choices=["none", "wandb"])
    p.add_argument("--run-name", default="stage1.5-tool-sft")
    p.add_argument("--wandb-project", default="MedicalSFT-stage1.5")

    # Precision
    p.add_argument("--dtype", default=None, choices=[None, "float16", "bfloat16"])

    return p.parse_args()


# ---------------------------------------------------------------------------
# Precision helpers (same as grpo_train.py)
# ---------------------------------------------------------------------------

def bf16_supported() -> bool:
    return bool(
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )


def default_torch_dtype(dtype_name: str | None):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return None


def resolve_training_precision(dtype_name: str | None) -> tuple[bool, bool]:
    dtype = default_torch_dtype(dtype_name)
    if dtype == torch.float16:
        return False, torch.cuda.is_available()
    if dtype == torch.bfloat16:
        return torch.cuda.is_available(), False
    return bf16_supported(), torch.cuda.is_available() and not bf16_supported()


# ---------------------------------------------------------------------------
# Special tokens (same as grpo_train.py)
# ---------------------------------------------------------------------------

def normalize_special_tokens(tokenizer, model) -> None:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("tokenizer.chat_template is missing.")

    vocab = tokenizer.get_vocab()
    if tokenizer.eos_token in {None, "", "<EOS_TOKEN>"}:
        if "<|im_end|>" in vocab:
            tokenizer.eos_token = "<|im_end|>"
        else:
            raise ValueError(f"Cannot resolve eos_token: {tokenizer.eos_token!r}")

    if tokenizer.pad_token in {None, "", "<PAD_TOKEN>", "<EOS_TOKEN>"}:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"  # Right-pad for SFT (causal LM)
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id


# ---------------------------------------------------------------------------
# Dataset: JSONL → tokenized with assistant-only loss masking
# ---------------------------------------------------------------------------

def load_traces(path: str) -> list[dict]:
    traces = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def find_assistant_spans(token_ids: list[int], tokenizer) -> list[tuple[int, int]]:
    """Find (start, end) token index spans for assistant content.

    Qwen2.5 ChatML format:
      <|im_start|>assistant\n ... <|im_end|>

    We train on tokens between '<|im_start|>assistant\\n' and '<|im_end|>'
    for each assistant turn. System, user, and tool tokens are masked.
    """
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Tokenize "assistant\n" to get the suffix tokens after <|im_start|>
    assistant_marker_ids = tokenizer.encode("assistant\n", add_special_tokens=False)

    spans = []
    i = 0
    while i < len(token_ids):
        if token_ids[i] == im_start_id:
            # Check if this is an assistant turn
            marker_end = i + 1 + len(assistant_marker_ids)
            if (marker_end <= len(token_ids)
                    and token_ids[i + 1: marker_end] == assistant_marker_ids):
                # Content starts after the marker
                content_start = marker_end
                # Find the closing <|im_end|>
                content_end = content_start
                while content_end < len(token_ids) and token_ids[content_end] != im_end_id:
                    content_end += 1
                # Include <|im_end|> in the span (model should learn to stop)
                if content_end < len(token_ids):
                    content_end += 1
                spans.append((content_start, content_end))
                i = content_end
                continue
        i += 1

    return spans


def tokenize_trace(trace: dict, tokenizer, max_seq_len: int) -> dict | None:
    """Tokenize a multi-turn trace with assistant-only loss masking.

    All tokens are kept as input_ids, but labels are -100 for everything
    except assistant content tokens. This teaches the model to:
      - Generate tool calls at the right time
      - Produce <think>...</think> reasoning
      - Output <answer>...</answer> with correct letter
    while conditioning on system prompt, user query, and tool responses.
    """
    messages = trace["messages"]

    # Apply chat template to get full text
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )

    # Tokenize
    encoding = tokenizer(text, add_special_tokens=False, truncation=False)
    input_ids = encoding["input_ids"]

    # Truncate from the left (preserve reasoning end)
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[-max_seq_len:]

    # Build labels: -100 everywhere except assistant spans
    labels = [-100] * len(input_ids)
    spans = find_assistant_spans(input_ids, tokenizer)
    for start, end in spans:
        for j in range(start, min(end, len(input_ids))):
            labels[j] = input_ids[j]

    # Sanity: at least some labels should be non-masked
    if all(l == -100 for l in labels):
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def build_dataset(traces: list[dict], tokenizer, max_seq_len: int) -> Dataset:
    """Convert list of traces to a HuggingFace Dataset with tokenized fields."""
    records = []
    skipped = 0
    for trace in traces:
        result = tokenize_trace(trace, tokenizer, max_seq_len)
        if result is None:
            skipped += 1
            continue
        records.append(result)

    if skipped:
        print(f"Skipped {skipped} traces (fully masked after tokenization)")

    ds = Dataset.from_dict({
        "input_ids": [r["input_ids"] for r in records],
        "attention_mask": [r["attention_mask"] for r in records],
        "labels": [r["labels"] for r in records],
    })
    return ds


# ---------------------------------------------------------------------------
# Data collator with dynamic padding
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorForSFT:
    """Pad input_ids, attention_mask, labels to the longest in the batch."""
    tokenizer: Any
    max_seq_len: int = 2048

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = min(
            max(len(f["input_ids"]) for f in features),
            self.max_seq_len,
        )

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Eval metrics: token accuracy on assistant spans
# ---------------------------------------------------------------------------

def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Keep only the predicted token id (argmax) to save memory during eval."""
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred) -> dict[str, float]:
    """Compute token-level accuracy on assistant-only tokens (labels != -100).

    Metrics:
      - token_accuracy: fraction of assistant tokens predicted correctly
    """
    pred_ids, labels = eval_pred
    # Shift: model predicts token[i+1] from token[i]
    pred_ids = pred_ids[:, :-1]
    labels = labels[:, 1:]

    mask = labels != -100
    if mask.sum() == 0:
        return {"token_accuracy": 0.0}

    correct = (pred_ids == labels) & mask
    accuracy = correct.sum().item() / mask.sum().item()
    return {"token_accuracy": accuracy}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- WandB ---
    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # --- Model & tokenizer ---
    print(f"Loading model from {args.model_path} ...")
    torch_dtype = default_torch_dtype(args.dtype)
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    normalize_special_tokens(tokenizer, model)

    # --- LoRA ---
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- Dataset ---
    print(f"Loading traces from {args.data_path} ...")
    traces = load_traces(args.data_path)
    print(f"Loaded {len(traces)} traces")

    print("Tokenizing traces ...")
    full_ds = build_dataset(traces, tokenizer, args.max_seq_len)
    print(f"Tokenized dataset: {len(full_ds)} samples")

    # Token stats
    total_tokens = sum(len(ids) for ids in full_ds["input_ids"])
    train_tokens = sum(
        sum(1 for l in labels if l != -100)
        for labels in full_ds["labels"]
    )
    print(f"Total tokens: {total_tokens:,}")
    print(f"Trainable tokens (assistant only): {train_tokens:,} "
          f"({100*train_tokens/total_tokens:.1f}%)")

    # Train/eval split
    if args.eval_split > 0:
        split = full_ds.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_ds = split["train"]
        eval_ds = split["test"]
        print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    else:
        train_ds = full_ds
        eval_ds = None

    # --- Training args ---
    use_bf16, use_fp16 = resolve_training_precision(args.dtype)

    steps_per_epoch = len(train_ds) // (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )
    total_steps = max(1, steps_per_epoch) * int(args.num_train_epochs)
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    print(f"Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}, "
          f"Warmup: {warmup_steps}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=warmup_steps,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        seed=args.seed,

        # Eval — track token_accuracy, save best checkpoint by eval_loss
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        eval_on_start=eval_ds is not None,
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Precision & memory
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Logging / saving — save_steps = eval_steps so best model can be loaded
        logging_steps=args.logging_steps,
        save_strategy="steps" if eval_ds is not None else "steps",
        save_steps=args.eval_steps if eval_ds is not None else args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        run_name=args.run_name,

        # DataLoader
        dataloader_num_workers=4,
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSFT(tokenizer, args.max_seq_len),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # --- Train ---
    print("Starting Stage 1.5 SFT training ...")
    trainer.train()

    # --- Save ---
    final_dir = Path(args.output_dir) / "final"
    print(f"Saving model to {final_dir} ...")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print("Done.")


if __name__ == "__main__":
    main()
