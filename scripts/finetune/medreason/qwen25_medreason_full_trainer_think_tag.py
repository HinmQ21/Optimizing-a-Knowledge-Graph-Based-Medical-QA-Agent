#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MedReason-style SFT for Qwen2.5 — full-parameter training with Transformers Trainer. "
            "Uses <think>...</think> <answer>...</answer> tag format instead of markdown headers. "
            "Uses the tokenizer's built-in chat_template (Qwen2.5 ChatML). "
            "Assistant-only masked loss, token-accuracy logging via MedReasonTrainer."
        )
    )
    parser.add_argument(
        "--model-path",
        default="models/Qwen2.5-3B-Instruct",
        help="Local model path or HF model id.",
    )
    parser.add_argument(
        "--data-path",
        default="UCSC-VLAA/MedReason",
        help="HF dataset id, local load_from_disk path, or local json/jsonl file.",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Optional dataset config name when loading from Hugging Face.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default=None)
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.02,
        help="Fraction of train split used for eval when --eval-split is not provided. Set 0 to disable.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--dataset-num-proc", type=int, default=None)

    parser.add_argument(
        "--output-dir",
        default="outputs/medreason_qwen25_3b_think_tag",
    )
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument(
        "--eval-strategy",
        default="steps",
        choices=["no", "steps", "epoch"],
    )
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument(
        "--save-strategy",
        default="steps",
        choices=["steps", "epoch"],
    )
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2002)

    parser.add_argument("--report-to", default="none", choices=["none", "wandb"])
    parser.add_argument("--run-name", default="medreason-qwen25-3b-think-tag")
    parser.add_argument("--wandb-project", default="MedReason")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-dir", default=None)
    parser.add_argument(
        "--wandb-mode",
        default="offline",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("--wandb-tags", nargs="*", default=None)

    parser.add_argument(
        "--dtype",
        default=None,
        choices=[None, "float16", "bfloat16"],
        help="Override model dtype. Default: bf16 on supported CUDA, otherwise fp16.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing (strongly recommended for full-param training).",
    )
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--dataloader-num-workers", type=int, default=2)

    parser.add_argument(
        "--log-train-acc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log token-level train accuracy like the original MedReason SFT script.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def load_dataset_maybe_dict(data_path: str, dataset_config: str | None):
    local_path = Path(data_path)
    if local_path.exists():
        if local_path.is_file() and local_path.suffix in {".json", ".jsonl"}:
            return load_dataset("json", data_files=str(local_path))
        return load_from_disk(str(local_path))
    if dataset_config:
        return load_dataset(data_path, dataset_config)
    return load_dataset(data_path)


def get_split_or_first(loaded, split: str) -> Dataset:
    if isinstance(loaded, DatasetDict):
        if split in loaded:
            return loaded[split]
        first_split = next(iter(loaded.keys()))
        return loaded[first_split]
    return loaded


def maybe_select(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None:
        return dataset
    return dataset.select(range(min(max_samples, len(dataset))))


def build_train_eval_splits(args: argparse.Namespace) -> tuple[Dataset, Dataset | None]:
    loaded = load_dataset_maybe_dict(args.data_path, args.dataset_config)
    train_base = get_split_or_first(loaded, args.train_split)

    eval_base = None
    if args.eval_split:
        eval_base = get_split_or_first(loaded, args.eval_split)
    elif args.validation_ratio and args.validation_ratio > 0:
        split_dataset = train_base.train_test_split(
            test_size=args.validation_ratio,
            seed=args.seed,
        )
        train_base = split_dataset["train"]
        eval_base = split_dataset["test"]

    train_base = maybe_select(train_base, args.max_train_samples)
    if eval_base is not None:
        eval_base = maybe_select(eval_base, args.max_eval_samples)
    return train_base, eval_base


# ---------------------------------------------------------------------------
# MedReason field resolution
# ---------------------------------------------------------------------------

def resolve_field(example: dict[str, Any], *candidates: str) -> str:
    for name in candidates:
        if name in example and example[name] is not None:
            return str(example[name])
    raise KeyError(f"Missing expected field. Tried: {', '.join(candidates)}")


def resolve_question(example: dict[str, Any]) -> str:
    return resolve_field(example, "Question", "question", "prompt")


def resolve_reasoning(example: dict[str, Any]) -> str:
    return resolve_field(
        example,
        "Complex_CoT",
        "complex_cot",
        "complex_cot_en",
        "reasoning",
        "Reasoning",
    )


def resolve_response(example: dict[str, Any]) -> str:
    return resolve_field(
        example,
        "Response",
        "response",
        "final_response",
        "answer",
        "Answer",
    )


def format_completion(example: dict[str, Any], args: argparse.Namespace) -> str:
    reasoning = resolve_reasoning(example)
    response = resolve_response(example)
    return f"<think>\n{reasoning}\n</think>\n<answer>\n{response}\n</answer>"


# ---------------------------------------------------------------------------
# Dtype / precision helpers
# ---------------------------------------------------------------------------

def resolve_dtype(dtype_name: str | None):
    if dtype_name is None:
        return None
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]


def bf16_supported() -> bool:
    return bool(
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )


def default_torch_dtype(dtype_name: str | None):
    resolved = resolve_dtype(dtype_name)
    if resolved is not None:
        return resolved
    if bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return None


def resolve_training_precision(dtype_name: str | None) -> tuple[bool, bool]:
    resolved = resolve_dtype(dtype_name)
    if resolved == torch.float16:
        return False, torch.cuda.is_available()
    if resolved == torch.bfloat16:
        return torch.cuda.is_available(), False
    return bf16_supported(), torch.cuda.is_available() and not bf16_supported()


# ---------------------------------------------------------------------------
# Model & tokenizer
# ---------------------------------------------------------------------------

def normalize_special_tokens(tokenizer, model) -> None:
    """Setup EOS/PAD tokens. Uses the tokenizer's built-in chat_template (Qwen2.5 ChatML).
    Raises ValueError if chat_template is missing rather than silently using a wrong fallback."""
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "tokenizer.chat_template is missing. "
            "Qwen2.5-3B-Instruct should have a built-in ChatML template. "
            "Check the model path or tokenizer files."
        )

    vocab = tokenizer.get_vocab()

    if tokenizer.eos_token in {None, "", "<EOS_TOKEN>"}:
        recovered = None
        if tokenizer.eos_token_id is not None:
            try:
                recovered = tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id)
            except Exception:
                recovered = None

        if recovered and recovered not in {"", "<unk>", None, "<EOS_TOKEN>"}:
            tokenizer.eos_token = recovered
        elif "<|im_end|>" in vocab:
            tokenizer.eos_token = "<|im_end|>"
        else:
            raise ValueError(
                f"Could not resolve a valid eos_token. "
                f"eos_token={tokenizer.eos_token!r}, eos_token_id={tokenizer.eos_token_id!r}"
            )

    if tokenizer.pad_token in {None, "", "<PAD_TOKEN>", "<EOS_TOKEN>"}:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id


def build_model_and_tokenizer(args: argparse.Namespace):
    torch_dtype = default_torch_dtype(args.dtype)
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    normalize_special_tokens(tokenizer, model)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    print("DEBUG eos_token:", tokenizer.eos_token, tokenizer.eos_token_id)
    print("DEBUG pad_token:", tokenizer.pad_token, tokenizer.pad_token_id)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Chat rendering & tokenization
# ---------------------------------------------------------------------------

def render_chat(prompt: str, completion: str, tokenizer) -> tuple[str, str]:
    prompt_messages = [{"role": "user", "content": prompt}]
    full_messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return prompt_text, full_text


def convert_to_tokenized_chat_dataset(
    dataset: Dataset,
    tokenizer,
    args: argparse.Namespace,
) -> Dataset:
    remove_columns = list(dataset.column_names)

    def map_row(example: dict[str, Any]) -> dict[str, Any]:
        prompt = resolve_question(example)
        completion = format_completion(example, args)
        prompt_text, full_text = render_chat(prompt, completion, tokenizer)

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        # Left-truncate to preserve the end of long reasoning chains
        if len(full_ids) > args.max_seq_len:
            full_ids = full_ids[-args.max_seq_len:]
            labels = labels[-args.max_seq_len:]

        attention_mask = [1] * len(full_ids)

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    tokenized = dataset.map(
        map_row,
        remove_columns=remove_columns,
        num_proc=args.dataset_num_proc,
        desc="Tokenizing MedReason dataset with assistant-only masked labels",
    )

    tokenized = tokenized.filter(
        lambda ex: any(label != -100 for label in ex["labels"]),
        desc="Filtering fully-masked samples",
    )

    return tokenized


# ---------------------------------------------------------------------------
# Token accuracy tracking
# ---------------------------------------------------------------------------

@dataclass
class RunningTokenStats:
    steps: int = 0
    correct: int = 0
    total: int = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        with torch.no_grad():
            shift_preds = logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:]
            valid_mask = shift_labels.ne(-100)
            self.correct += ((shift_preds == shift_labels) & valid_mask).sum().item()
            self.total += valid_mask.sum().item()
            self.steps += 1

    def pop(self) -> dict[str, float]:
        if self.steps == 0:
            return {}
        accuracy = float(self.correct / self.total) if self.total else 0.0
        self.steps = 0
        self.correct = 0
        self.total = 0
        return {"train_acc": accuracy}


class MedReasonTrainer(Trainer):
    def __init__(self, *args, log_train_acc: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_train_acc = log_train_acc
        self._running_token_stats = RunningTokenStats()
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        if self._log_train_acc and model.training and "labels" in inputs:
            try:
                logits = outputs.logits.detach()
            except (AttributeError, NotImplementedError):
                logits = None

            if logits is not None:
                self._running_token_stats.update(logits, inputs["labels"].detach())

        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs, start_time=None):
        if self._log_train_acc and "loss" in logs:
            logs = dict(logs)
            logs.update(self._running_token_stats.pop())
        super().log(logs, start_time)


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def configure_wandb(args: argparse.Namespace, output_dir: Path, run_config: dict[str, Any]):
    if args.report_to != "wandb":
        return None

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: wandb. Install with `pip install wandb` or use `--report-to none`."
        ) from exc

    wandb_dir = args.wandb_dir or str(output_dir / "wandb")
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_NAME"] = args.run_name
    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ["WANDB_DIR"] = wandb_dir
    os.environ.setdefault("WANDB_WATCH", "false")

    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_tags:
        os.environ["WANDB_TAGS"] = ",".join(args.wandb_tags)

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        dir=wandb_dir,
        mode=args.wandb_mode,
        tags=args.wandb_tags,
        config=run_config,
    )


def build_run_config(
    args: argparse.Namespace,
    train_base: Dataset,
    eval_base: Dataset | None,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    output_dir: Path,
    tokenizer,
) -> dict[str, Any]:
    return {
        **vars(args),
        "output_dir": str(output_dir),
        "train_num_rows_raw": len(train_base),
        "eval_num_rows_raw": 0 if eval_base is None else len(eval_base),
        "train_num_rows_tokenized": len(train_dataset),
        "eval_num_rows_tokenized": 0 if eval_dataset is None else len(eval_dataset),
        "cuda_available": torch.cuda.is_available(),
        "bf16_enabled": bf16_supported(),
        "trainer_type": "transformers.Trainer + transformers.AutoModelForCausalLM",
        "finetuning_mode": "full",
        "dataset_format": "chat_with_think_answer_tags_assistant_only_masked_labels",
        "loss_behavior": "assistant_only_via_-100_labels",
        "truncation_behavior": "left_truncate_to_preserve_reasoning_end",
        "tokenizer_eos_token": tokenizer.eos_token,
        "tokenizer_pad_token": tokenizer.pad_token,
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_bf16, use_fp16 = resolve_training_precision(args.dtype)

    train_base, eval_base = build_train_eval_splits(args)
    model, tokenizer = build_model_and_tokenizer(args)

    train_dataset = convert_to_tokenized_chat_dataset(train_base, tokenizer, args)
    eval_dataset = None
    if eval_base is not None:
        eval_dataset = convert_to_tokenized_chat_dataset(eval_base, tokenizer, args)

    if eval_dataset is None:
        args.eval_strategy = "no"
    elif args.eval_strategy != "no" and args.save_strategy != args.eval_strategy:
        raise ValueError(
            "--save-strategy must match --eval-strategy when evaluation is enabled "
            "because load_best_model_at_end=True."
        )

    run_config = build_run_config(
        args=args,
        train_base=train_base,
        eval_base=eval_base,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        tokenizer=tokenizer,
    )
    save_json(run_config, output_dir / "run_config.json")
    wandb_run = configure_wandb(args, output_dir, run_config)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=bool(eval_dataset is not None and args.eval_strategy != "no"),
        metric_for_best_model="eval_loss" if eval_dataset is not None and args.eval_strategy != "no" else None,
        greater_is_better=False if eval_dataset is not None and args.eval_strategy != "no" else None,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=[] if args.report_to == "none" else ["wandb"],
        run_name=args.run_name,
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        optim=args.optim,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    trainer = MedReasonTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        log_train_acc=args.log_train_acc,
    )

    train_result = trainer.train()
    final_eval_metrics = trainer.evaluate() if eval_dataset is not None else {}

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    best_info = {
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "train_metrics": train_result.metrics,
        "final_eval_metrics": final_eval_metrics,
    }
    save_json(best_info, output_dir / "best_checkpoint.json")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
