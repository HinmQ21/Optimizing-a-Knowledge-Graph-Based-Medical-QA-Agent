"""Step 7: GRPO training for medical reasoning with knowledge retrieval.

Trains Qwen2.5-3B SFT checkpoint using TRL GRPOTrainer. The model learns to:
  1. Call search_medical_knowledge tool for relevant KG facts
  2. Reason in <think>...</think> blocks
  3. Answer in <answer>...</answer> blocks (concise MCQ letter)

Usage:
    cd /home/vcsai/minhlbq/baseline
    ./training_venv312/bin/python -m scripts.train_rl.grpo_train \
        --model-path outputs/medreason_stage2_from_huatuo_qwen25_3b \
        --data-path dataset/MedQA/train \
        --data-dir data/ \
        --output-dir outputs/grpo_medical \
        --max-completion-length 2048 \
        --report-to wandb \
        --run-name grpo-medical-kg
"""

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from trl.chat_template_utils import qwen3_schema  # Qwen2.5 uses same <tool_call> format

from scripts.serve.retrieval_tool import MedicalKnowledgeTool, search_medical_knowledge
from scripts.train_rl.data_prep import load_medqa
from scripts.train_rl.reward_fns import answer_reward, format_reward, tool_quality_reward, enhanced_tool_quality_reward


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GRPO training for medical reasoning with knowledge-graph retrieval. "
            "Trains LoRA adapters on top of a Stage 1/2 SFT checkpoint. "
            "Uses TRL GRPOTrainer with in-process tool calling."
        )
    )

    # Model
    parser.add_argument(
        "--model-path",
        default="outputs/medreason_stage2_from_huatuo_qwen25_3b",
        help="Path to Stage 1/2 SFT checkpoint (local) or HF model id.",
    )

    # Data
    parser.add_argument(
        "--data-path",
        default="dataset/MedQA/train",
        help="load_from_disk path to training dataset (MedQA or MedMCQA).",
    )
    parser.add_argument(
        "--data-dir",
        default="data/",
        help="Directory containing FAISS indices and medical_hg.json for retrieval tool.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap number of training examples (for quick ablations).",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default="outputs/grpo_medical",
    )

    # QLoRA / LoRA
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load base weights in 4-bit NF4 via bitsandbytes (QLoRA).",
    )
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # GRPO algorithm
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL penalty coefficient (anchors policy to SFT reference).")
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="PPO clip ratio.")
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Number of rollouts (G) per prompt.")
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tool-calling-iterations", type=int, default=3)

    # Training
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lr-scheduler-type", default="cosine",
                        help="LR scheduler type (default: cosine).")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Override num_train_epochs. Use 1 for a dry run.")
    parser.add_argument("--seed", type=int, default=42)

    # Evaluation
    parser.add_argument(
        "--eval-data-path",
        default=None,
        help="load_from_disk path to eval dataset (e.g. dataset/MedQA/test).",
    )
    parser.add_argument("--eval-steps", type=int, default=20,
                        help="Run evaluation every N steps.")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--max-eval-samples", type=int, default=100,
                        help="Cap eval examples (GRPO eval is slow due to generation).")

    # Logging / saving
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument(
        "--report-to",
        default="none",
        choices=["none", "wandb", "tensorboard"],
    )
    parser.add_argument("--run-name", default="grpo-medical-kg")
    parser.add_argument("--wandb-project", default="MedGRPO")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-mode",
        default="offline",
        choices=["online", "offline", "disabled"],
    )

    # Precision
    parser.add_argument(
        "--dtype",
        default=None,
        choices=[None, "float16", "bfloat16"],
    )

    # Native generation speedups (no vLLM required)
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
        help=(
            "Attention implementation for model loading. "
            "'sdpa' (default) uses PyTorch's fused scaled_dot_product_attention — "
            "no extra install needed, works on GB10. "
            "'flash_attention_2' requires `pip install flash-attn` and sm>=8.0."
        ),
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=None,
        help=(
            "Batch size for rollout generation. "
            "Defaults to per_device_train_batch_size × num_generations. "
            "Increase to improve GPU utilization during generation "
            "(at the cost of more peak VRAM)."
        ),
    )
    parser.add_argument(
        "--torch-empty-cache-steps",
        type=int,
        default=None,
        help=(
            "Call torch.cuda.empty_cache() every N steps. "
            "Helps reclaim fragmented VRAM on long runs (e.g. --torch-empty-cache-steps 50)."
        ),
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes (default 4). Set 0 to disable.",
    )
    parser.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=2,
        help="Number of batches to prefetch per DataLoader worker (default 2).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dtype helpers (same pattern as medreason SFT scripts)
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
    """Returns (bf16, fp16) tuple for TrainingArguments."""
    dtype = default_torch_dtype(dtype_name)
    if dtype == torch.float16:
        return False, torch.cuda.is_available()
    if dtype == torch.bfloat16:
        return torch.cuda.is_available(), False
    return bf16_supported(), torch.cuda.is_available() and not bf16_supported()


# ---------------------------------------------------------------------------
# Special token helpers (from qwen25_medreason_full_trainer_think_tag.py)
# ---------------------------------------------------------------------------

def normalize_special_tokens(tokenizer, model) -> None:
    """Ensure EOS/PAD tokens are correctly set for Qwen2.5 ChatML."""
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "tokenizer.chat_template is missing. "
            "Check that the model path contains a valid Qwen2.5 tokenizer."
        )

    vocab = tokenizer.get_vocab()
    if tokenizer.eos_token in {None, "", "<EOS_TOKEN>"}:
        if "<|im_end|>" in vocab:
            tokenizer.eos_token = "<|im_end|>"
        else:
            raise ValueError(
                f"Cannot resolve eos_token. "
                f"eos_token={tokenizer.eos_token!r}, eos_token_id={tokenizer.eos_token_id!r}"
            )

    if tokenizer.pad_token in {None, "", "<PAD_TOKEN>", "<EOS_TOKEN>"}:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"   # Left-pad for decoder-only generation
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- WandB setup ---
    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
        os.environ.setdefault("WANDB_MODE", args.wandb_mode)

    # --- Pre-load retrieval tool with correct data_dir ---
    # This ensures MedEmbed-large + FAISS are loaded before training starts.
    print(f"Pre-loading retrieval tool from {args.data_dir} ...")
    MedicalKnowledgeTool.load(data_dir=args.data_dir)
    print("Retrieval tool ready.")

    # --- Load model and tokenizer ---
    print(f"Loading model from {args.model_path} (4-bit={args.load_in_4bit}) ...")
    torch_dtype = default_torch_dtype(args.dtype)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype or torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif torch_dtype is not None:
        model_kwargs["dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    normalize_special_tokens(tokenizer, model)

    # TRL 0.29 only auto-detects Qwen3 chat template for tool calling.
    # Qwen2.5 uses the same <tool_call>...</tool_call> format, so qwen3_schema applies.
    tokenizer.response_schema = qwen3_schema

    # --- Dataset ---
    print(f"Loading dataset from {args.data_path} ...")
    train_ds = load_medqa(args.data_path, max_samples=args.max_train_samples)
    print(f"Training examples: {len(train_ds)}")

    eval_ds = None
    if args.eval_data_path:
        print(f"Loading eval dataset from {args.eval_data_path} ...")
        eval_ds = load_medqa(args.eval_data_path, max_samples=args.max_eval_samples)
        print(f"Eval examples: {len(eval_ds)}")

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

    # --- Warmup steps: compute from actual dataset size ---
    # NOTE: TRL GRPOTrainer treats per_device_train_batch_size as the number
    # of *sequences* (prompts × num_generations).  The number of unique prompts
    # consumed per micro-batch is  bs // num_generations, so the real number of
    # optimizer steps per epoch is:
    #   len(dataset) // ((bs // G) * grad_accum)
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        prompts_per_micro_batch = max(
            1, args.per_device_train_batch_size // args.num_generations
        )
        steps_per_epoch = len(train_ds) // (
            prompts_per_micro_batch * args.gradient_accumulation_steps
        )
        total_steps = max(1, steps_per_epoch) * int(args.num_train_epochs)
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    print(f"Total training steps: {total_steps}, warmup steps: {warmup_steps}")

    # --- GRPOConfig ---
    use_bf16, use_fp16 = resolve_training_precision(args.dtype)

    # DataLoader prefetch: only meaningful when num_workers > 0
    prefetch = args.dataloader_prefetch_factor if args.dataloader_num_workers > 0 else None

    training_args = GRPOConfig(
        output_dir=args.output_dir,

        # Generation
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        # generation_batch_size: how many prompts × generations to run at once.
        # None → trainer uses per_device_train_batch_size × num_generations.
        # Larger values improve GPU utilization but need more VRAM.
        generation_batch_size=args.generation_batch_size,

        # Tool calling
        max_tool_calling_iterations=args.max_tool_calling_iterations,

        # GRPO algorithm
        loss_type="grpo",
        beta=args.beta,
        epsilon=args.epsilon,
        scale_rewards="group",
        num_iterations=1,

        # Reward — rebalanced to prevent tool-calling collapse.
        # Previous [0.15, 0.70, 0.15] let model shortcut by skipping tools.
        reward_weights=[0.25, 0.50, 0.25],

        # Training
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=warmup_steps,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,

        # Evaluation
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        per_device_eval_batch_size=args.per_device_eval_batch_size,

        # Optimizer: adamw_torch_fused uses CUDA-fused kernel → ~5-10% faster
        # than standard AdamW. Default in TRL 0.29.1 GRPOConfig.
        optim="adamw_torch_fused",

        # Memory
        gradient_checkpointing=True,
        # Recompute activations with fused attention to save VRAM during backward.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16,
        fp16=use_fp16,

        # VRAM management: empty CUDA cache periodically to reduce fragmentation
        torch_empty_cache_steps=args.torch_empty_cache_steps,

        # DataLoader: parallel workers + prefetch to keep GPU fed during rollouts.
        # MedQA dataset is in Arrow format (memory-mapped) so workers are safe.
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=prefetch,
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_persistent_workers=args.dataloader_num_workers > 0,

        # Logging / saving
        log_completions=True,
        num_completions_to_print=2,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        run_name=args.run_name,
    )

    # --- Trainer ---
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        reward_funcs=[format_reward, answer_reward, tool_quality_reward],
        tools=[search_medical_knowledge],
        peft_config=peft_config,
    )

    # --- Train ---
    print("Starting GRPO training ...")
    trainer.train()

    # --- Save ---
    final_dir = Path(args.output_dir) / "final"
    print(f"Saving model to {final_dir} ...")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print("Done.")


if __name__ == "__main__":
    main()
