"""GRPO training for medical reasoning — A100-optimised build.

Key differences vs grpo_train.py (DGX Spark / GB10):
  - use_vllm=True          : vLLM handles rollout generation (~3x throughput)
  - flash_attention_2      : default attention implementation (A100 sm_80)
  - bfloat16               : A100 native bf16 (no fp16 fallback needed)
  - larger default batches : A100 80 GB dedicated VRAM vs 128 GB shared
  - vLLM server config     : exposed as CLI flags for easy tuning

Usage:
    cd /home/vcsai/minhlbq/baseline
    source .venv_grpo_a100/bin/activate
    python -m scripts.train_rl.grpo_train_a100 \\
        --model-path outputs/medreason_stage2_from_huatuo_qwen25_3b \\
        --data-path  dataset/MedQA/train \\
        --data-dir   data/ \\
        --output-dir outputs/grpo_medical_a100 \\
        --report-to  wandb \\
        --run-name   grpo-medical-kg-a100
"""

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.chat_template_utils import qwen3_schema

from scripts.serve.retrieval_tool import MedicalKnowledgeTool, search_medical_knowledge
from scripts.train_rl.data_prep import load_medqa
from scripts.train_rl.reward_fns import answer_reward, format_reward, tool_quality_reward


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GRPO training for medical reasoning — A100-optimised. "
            "Uses vLLM for generation and Flash Attention 2 for the training model."
        )
    )

    # Model
    parser.add_argument(
        "--model-path",
        default="outputs/medreason_stage2_from_huatuo_qwen25_3b",
        help="Path to Stage 2 SFT checkpoint (local) or HF model id.",
    )

    # Data
    parser.add_argument(
        "--data-path",
        default="dataset/MedQA/train",
        help="load_from_disk path to training dataset.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/",
        help="Directory containing FAISS indices and medical_hg.json.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap training examples (for quick ablations).",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default="outputs/grpo_medical_a100",
    )

    # LoRA
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # GRPO algorithm
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument(
        "--num-generations", type=int, default=8,
        help=(
            "Rollouts (G) per prompt. Default 8 on A100 (vs 4 on DGX Spark) "
            "because vLLM amortises generation cost across larger batches."
        ),
    )
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tool-calling-iterations", type=int, default=3)

    # Training
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument(
        "--per-device-train-batch-size", type=int, default=4,
        help="Default 4 on A100 80 GB (vs 2 on DGX Spark shared memory).",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lr-scheduler-type", default="cosine",
                        help="LR scheduler type (default: cosine).")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)

    # Evaluation
    parser.add_argument(
        "--eval-data-path",
        default=None,
        help="load_from_disk path to eval dataset (e.g. dataset/MedQA/test).",
    )
    parser.add_argument("--eval-steps", type=int, default=100,
                        help="Run evaluation every N steps.")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--max-eval-samples", type=int, default=50,
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
    parser.add_argument("--run-name", default="grpo-medical-kg-a100")
    parser.add_argument("--wandb-project", default="MedGRPO")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-mode",
        default="offline",
        choices=["online", "offline", "disabled"],
    )

    # Attention
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help=(
            "flash_attention_2 (default) — requires flash-attn, A100 sm_80 supported. "
            "Fall back to sdpa only if flash-attn is not installed."
        ),
    )

    # vLLM rollout server
    parser.add_argument(
        "--use-vllm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use vLLM for rollout generation (default: True on A100). "
            "vLLM spawns a separate server process that shares the same GPU. "
            "Disable with --no-use-vllm to fall back to native PyTorch generation."
        ),
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.35,
        help=(
            "Fraction of GPU memory reserved for the vLLM server (default 0.35). "
            "On A100 80 GB: 0.35 → ~28 GB for vLLM, ~52 GB for training. "
            "Raise if generation OOMs; lower if backward pass OOMs."
        ),
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=4096,
        help="Max sequence length the vLLM server supports (default 4096).",
    )
    parser.add_argument(
        "--vllm-server-host",
        default="0.0.0.0",
        help="Host for the vLLM server (default 0.0.0.0).",
    )
    parser.add_argument(
        "--vllm-server-port",
        type=int,
        default=8000,
        help="Port for the vLLM server (default 8000).",
    )
    parser.add_argument(
        "--vllm-server-timeout",
        type=float,
        default=180.0,
        help=(
            "Seconds to wait for vLLM server startup (default 180). "
            "Increase if the model takes longer to load."
        ),
    )

    # DataLoader
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=2)

    # VRAM tuning
    parser.add_argument(
        "--torch-empty-cache-steps",
        type=int,
        default=None,
        help="Call torch.cuda.empty_cache() every N steps. Usually not needed on A100.",
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=None,
        help=(
            "Batch size for native-PyTorch rollout generation (--no-use-vllm only). "
            "Ignored when vLLM is active."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

def bf16_supported() -> bool:
    return bool(
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )


def default_torch_dtype():
    # A100 always has native bf16; no need for a float16 fallback.
    if torch.cuda.is_available() and bf16_supported():
        return torch.bfloat16
    return torch.float16


def resolve_training_precision() -> tuple[bool, bool]:
    """Returns (bf16, fp16) for TrainingArguments."""
    if bf16_supported():
        return True, False
    return False, torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Special token helpers (shared with DGX Spark script)
# ---------------------------------------------------------------------------

def normalize_special_tokens(tokenizer, model) -> None:
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
                f"Cannot resolve eos_token: {tokenizer.eos_token!r}"
            )

    if tokenizer.pad_token in {None, "", "<PAD_TOKEN>", "<EOS_TOKEN>"}:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id


# ---------------------------------------------------------------------------
# VRAM estimation (informational)
# ---------------------------------------------------------------------------

def _log_vram_plan(args) -> None:
    """Print a rough VRAM allocation plan so the user can spot misconfigs."""
    if not torch.cuda.is_available():
        return
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if args.use_vllm:
        vllm_gb = total_gb * args.vllm_gpu_memory_utilization
        train_gb = total_gb - vllm_gb
        print(
            f"[VRAM plan] Total: {total_gb:.0f} GB | "
            f"vLLM server: {vllm_gb:.0f} GB "
            f"(--vllm-gpu-memory-utilization {args.vllm_gpu_memory_utilization}) | "
            f"Training: {train_gb:.0f} GB"
        )
    else:
        print(
            f"[VRAM plan] Total: {total_gb:.0f} GB | "
            "vLLM disabled — full VRAM available for training."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- WandB ---
    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
        os.environ.setdefault("WANDB_MODE", args.wandb_mode)

    # --- Retrieval tool (CPU encoder — does not compete with GPU training) ---
    print(f"Pre-loading retrieval tool from {args.data_dir} ...")
    MedicalKnowledgeTool.load(data_dir=args.data_dir)
    print("Retrieval tool ready.")

    _log_vram_plan(args)

    # --- Model ---
    print(f"Loading model from {args.model_path} ...")
    torch_dtype = default_torch_dtype()
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
        "dtype": torch_dtype,
    }

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    normalize_special_tokens(tokenizer, model)

    # TRL 0.29 only auto-detects Qwen3 template for tool calling.
    # Qwen2.5 uses the same <tool_call> format so qwen3_schema applies.
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

    # --- Warmup steps ---
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        steps_per_epoch = len(train_ds) // (
            args.per_device_train_batch_size * args.gradient_accumulation_steps
        )
        total_steps = max(1, steps_per_epoch) * int(args.num_train_epochs)
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    print(f"Total training steps: {total_steps}, warmup steps: {warmup_steps}")

    # --- GRPOConfig ---
    use_bf16, use_fp16 = resolve_training_precision()
    prefetch = args.dataloader_prefetch_factor if args.dataloader_num_workers > 0 else None

    grpo_kwargs: dict[str, Any] = dict(
        output_dir=args.output_dir,

        # Generation
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        generation_batch_size=args.generation_batch_size,

        # Tool calling
        max_tool_calling_iterations=args.max_tool_calling_iterations,

        # GRPO algorithm
        loss_type="grpo",
        beta=args.beta,
        epsilon=args.epsilon,
        scale_rewards="group",
        num_iterations=1,

        # Reward
        reward_weights=[0.15, 0.7, 0.15],

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

        # Optimizer: CUDA-fused AdamW — fastest single-GPU option
        optim="adamw_torch_fused",

        # Memory
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16,
        fp16=use_fp16,
        torch_empty_cache_steps=args.torch_empty_cache_steps,

        # DataLoader
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

    # --- vLLM rollout server ---
    if args.use_vllm:
        grpo_kwargs.update(
            use_vllm=True,
            vllm_server_host=args.vllm_server_host,
            vllm_server_port=args.vllm_server_port,
            vllm_server_timeout=args.vllm_server_timeout,
            # Pass GPU memory utilization and max model length to the vLLM server.
            # These are forwarded via vllm_server_kwargs in TRL 0.29.x.
            vllm_server_kwargs={
                "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                "max_model_len": args.vllm_max_model_len,
                "dtype": "bfloat16",
                "trust_remote_code": True,
            },
        )
        print(
            f"[vLLM] Enabled — server at {args.vllm_server_host}:{args.vllm_server_port}, "
            f"gpu_memory_utilization={args.vllm_gpu_memory_utilization}, "
            f"max_model_len={args.vllm_max_model_len}"
        )
    else:
        print("[vLLM] Disabled — using native PyTorch generation.")

    training_args = GRPOConfig(**grpo_kwargs)

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
    print("Starting GRPO training (A100) ...")
    trainer.train()

    # --- Save ---
    final_dir = Path(args.output_dir) / "final"
    print(f"Saving model to {final_dir} ...")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print("Done.")


if __name__ == "__main__":
    main()
