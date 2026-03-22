#!/usr/bin/env python3
import argparse
import json
import os
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
        description="HuatuoGPT-o1 style stage-1 full SFT using Transformers Trainer with assistant-only masked loss."
    )
    parser.add_argument("--model-path", required=True, help="Local model path or HF model id.")
    parser.add_argument(
        "--data-path",
        default="FreedomIntelligence/medical-o1-reasoning-SFT",
        help="HF dataset id, local load_from_disk path, or local json/jsonl file.",
    )
    parser.add_argument(
        "--dataset-config",
        default="en",
        help="HF dataset config name. Use en/zh for the official medical-o1-reasoning-SFT dataset.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default=None)
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--dataset-num-proc", type=int, default=None)

    parser.add_argument("--output-dir", default="outputs/huatuo_stage1_qwen25_3b_full_trainer")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--report-to", default="none", choices=["none", "wandb"])
    parser.add_argument("--run-name", default="huatuo-stage1-qwen25-3b-full-trainer")
    parser.add_argument("--wandb-project", default="med-qwen2.5-3b-huatuo")
    parser.add_argument(
        "--wandb-entity",
        default="qminhlb-vietnam-national-university-hanoi",
    )
    parser.add_argument("--wandb-dir", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-tags", nargs="*", default=None)

    parser.add_argument(
        "--dtype",
        default=None,
        choices=[None, "float16", "bfloat16"],
        help="Override model dtype. Default uses bf16 on supported CUDA, otherwise fp16 on CUDA.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing for lower memory usage during full finetuning.",
    )
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    return parser.parse_args()


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


def build_train_eval_splits(args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    loaded = load_dataset_maybe_dict(args.data_path, args.dataset_config)
    train_base = get_split_or_first(loaded, args.train_split)

    if args.eval_split:
        eval_base = get_split_or_first(loaded, args.eval_split)
    else:
        split_dataset = train_base.train_test_split(
            test_size=args.validation_ratio,
            seed=args.seed,
        )
        train_base = split_dataset["train"]
        eval_base = split_dataset["test"]

    train_base = maybe_select(train_base, args.max_train_samples)
    eval_base = maybe_select(eval_base, args.max_eval_samples)
    return train_base, eval_base


def resolve_field(example: dict[str, Any], *candidates: str) -> str:
    for name in candidates:
        if name in example and example[name] is not None:
            return str(example[name])
    raise KeyError(f"Missing expected field. Tried: {', '.join(candidates)}")


def format_completion(example: dict[str, Any]) -> str:
    cot = resolve_field(example, "Complex_CoT", "complex_cot", "complex_cot_en")
    response = resolve_field(example, "Response", "response", "final_response")
    return f"## Thinking\n\n{cot}\n\n## Final Response\n\n{response}"


def resolve_dtype(dtype_name: str | None):
    if dtype_name is None:
        return None
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_name]


def bf16_supported() -> bool:
    return bool(
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )


def default_torch_dtype(dtype_name: str | None):
    resolved_dtype = resolve_dtype(dtype_name)
    if resolved_dtype is not None:
        return resolved_dtype
    if bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return None


def resolve_training_precision(dtype_name: str | None) -> tuple[bool, bool]:
    resolved_dtype = resolve_dtype(dtype_name)
    if resolved_dtype == torch.float16:
        return False, torch.cuda.is_available()
    if resolved_dtype == torch.bfloat16:
        return torch.cuda.is_available(), False
    return bf16_supported(), torch.cuda.is_available() and not bf16_supported()


def normalize_special_tokens(tokenizer, model) -> None:
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
                f"Could not resolve a valid eos_token. Current eos_token={tokenizer.eos_token!r}, "
                f"eos_token_id={tokenizer.eos_token_id!r}"
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
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    torch_dtype = default_torch_dtype(args.dtype)
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


def render_chat(prompt: str, completion: str, tokenizer) -> tuple[str, str]:
    prompt_messages = [
        {"role": "user", "content": prompt},
    ]
    full_messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]

    has_chat_template = getattr(tokenizer, "chat_template", None)

    if has_chat_template:
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
    else:
        prompt_text = f"### User:\n{prompt}\n\n### Assistant:\n"
        full_text = f"{prompt_text}{completion}"

    return prompt_text, full_text


def convert_to_tokenized_chat_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int,
    dataset_num_proc: int | None,
) -> Dataset:
    remove_columns = list(dataset.column_names)

    def map_row(example: dict[str, Any]) -> dict[str, Any]:
        question = resolve_field(example, "Question", "question", "prompt")
        answer = format_completion(example)

        prompt_text, full_text = render_chat(question, answer, tokenizer)

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        full_enc = tokenizer(
            full_text,
            add_special_tokens=False,
        )

        input_ids = full_enc["input_ids"]

        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        # Left-truncate to preserve the end of long reasoning chains
        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
            labels = labels[-max_length:]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    tokenized = dataset.map(
        map_row,
        remove_columns=remove_columns,
        num_proc=dataset_num_proc,
        desc="Tokenizing dataset with assistant-only masked labels",
    )

    tokenized = tokenized.filter(
        lambda ex: any(label != -100 for label in ex["labels"]),
        desc="Filtering fully-masked samples",
    )

    return tokenized


def build_run_config(
    args: argparse.Namespace,
    train_base: Dataset,
    eval_base: Dataset,
    output_dir: Path,
    tokenizer,
) -> dict[str, Any]:
    return {
        **vars(args),
        "output_dir": str(output_dir),
        "train_num_rows": len(train_base),
        "eval_num_rows": len(eval_base),
        "cuda_available": torch.cuda.is_available(),
        "bf16_enabled": bf16_supported(),
        "trainer_type": "transformers.Trainer + transformers.AutoModelForCausalLM",
        "finetuning_mode": "full",
        "dataset_format": "pretokenized_chat_with_masked_labels",
        "loss_behavior": "assistant_only_via_-100_labels",
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "tokenizer_eos_token": tokenizer.eos_token,
        "tokenizer_pad_token": tokenizer.pad_token,
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
    }


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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_bf16, use_fp16 = resolve_training_precision(args.dtype)

    train_base, eval_base = build_train_eval_splits(args)
    model, tokenizer = build_model_and_tokenizer(args)

    run_config = build_run_config(args, train_base, eval_base, output_dir, tokenizer)
    save_json(run_config, output_dir / "run_config.json")
    wandb_run = configure_wandb(args, output_dir, run_config)

    train_dataset = convert_to_tokenized_chat_dataset(
        train_base,
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
        dataset_num_proc=args.dataset_num_proc,
    )
    eval_dataset = convert_to_tokenized_chat_dataset(
        eval_base,
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
        dataset_num_proc=args.dataset_num_proc,
    )

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
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    final_metrics = trainer.evaluate()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    best_info = {
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "train_metrics": train_result.metrics,
        "final_eval_metrics": final_metrics,
    }
    save_json(best_info, output_dir / "best_checkpoint.json")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
