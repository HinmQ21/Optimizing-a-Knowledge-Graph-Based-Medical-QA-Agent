"""GRPO training with vLLM-accelerated rollout generation.

Same reward design, LoRA config, and tool-calling setup as grpo_train.py,
but adds TRL's built-in vLLM colocate mode for the generation phase.
Run with ./vllm_venv312/bin/python (not training_venv312).

vLLM colocate mode mechanics:
  - A vLLM LLM engine lives in the same process as the training model.
  - Before each optimizer step, TRL merges LoRA weights and syncs them to vLLM.
  - vLLM generates all G rollouts per prompt in a single batched call — this is
    10–30× faster than HF model.generate() for large generation batch sizes.
  - enable_sleep_mode offloads vLLM weights to CPU/disk during backward.
    WARNING: vLLM 0.20.0 sleep(level=2) reloads from safetensors on disk every
    wake_up() call (~34s overhead per step) — SLOWER than baseline. Default off.
    On GB10's 120 GB unified memory, sleep mode is unnecessary.
  - Tool calling (_tool_call_loop) is fully supported in colocate mode.
    (vllm_mode="server" raises NotImplementedError when tools are passed.)

Expected speedup (sleep mode OFF):
  Rollout generation is typically 70–80% of GRPO wall time.
  vLLM acceleration of that phase → ~3–5× overall reduction.
  4–5 days on GB10 → ~1–2 days.

Memory profile (GB10, 120 GB unified, sleep OFF):
  Training model + LoRA + optimizer states : ~35 GB
  vLLM engine (base model weights + KV cache at util=0.5) : ~66 GB
  Total peak (both resident simultaneously) : ~101 GB — within 120 GB.

Usage:
    cd /home/vcsai/minhlbq/baseline

    # Dry run — verify setup (no checkpoint saved)
    ./vllm_venv312/bin/python -m scripts.train_rl.grpo_train_vllm \\
        --model-path outputs/stage1_5_tool_sft_v2_merged \\
        --max-steps 10 --max-eval-samples 10 \\
        --per-device-train-batch-size 1 \\
        --use-vllm

    # Full training (vLLM enabled, recommended)
    ./vllm_venv312/bin/python -m scripts.train_rl.grpo_train_vllm \\
        --model-path outputs/stage1_5_tool_sft_v2_merged \\
        --output-dir outputs/grpo_medical_lora_v6_vllm \\
        --data-dir data/ \\
        --use-vllm --vllm-gpu-mem-util 0.5 \\
        --num-generations 8 \\
        --report-to wandb --run-name grpo-medical-v6-vllm

    # Fallback: no vLLM (same behavior as grpo_train.py, uses training_venv312 args)
    ./vllm_venv312/bin/python -m scripts.train_rl.grpo_train_vllm \\
        --model-path outputs/stage1_5_tool_sft_v2_merged \\
        --output-dir outputs/grpo_medical_lora_v6 \\
        --data-dir data/
"""

import argparse
import os
import re
import warnings
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

# TRL's vLLM version check is conservative (lists 0.10-0.12) but TRL 0.29.1
# internally targets the V1 API available in 0.6+. vLLM 0.20.0 is compatible.
warnings.filterwarnings(
    "ignore",
    message="TRL currently supports vLLM versions",
    category=UserWarning,
)

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from scripts.serve.retrieval_tool import MedicalKnowledgeTool, search_medical_knowledge
from scripts.utils.model_adapter import (
    ModelFamily,
    detect_family,
    get_model_load_kwargs,
    get_trl_response_schema,
    get_tokenizer_load_kwargs,
    normalize_special_tokens,
)
from scripts.train_rl.data_prep import load_medqa
from scripts.train_rl.reward_fns import answer_reward, format_reward, enhanced_tool_quality_reward
from scripts.train_rl.reward_fns_gdpo import (
    answer_reward as gdpo_answer_reward,
    structure_reward,
    tool_reward,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GRPO training with vLLM colocate acceleration. "
            "Adds --use-vllm flag on top of grpo_train.py; all other args identical. "
            "Run with ./vllm_venv312/bin/python."
        )
    )

    # Model
    parser.add_argument(
        "--model-path",
        default="outputs/stage1_5_tool_sft_v2_merged",
        help="Path to Stage 1.5 SFT merged checkpoint.",
    )
    parser.add_argument(
        "--model-family",
        default="auto",
        choices=["auto", "qwen", "llama"],
    )

    # Data
    parser.add_argument("--data-path", default="dataset/MedQA/train")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--max-train-samples", type=int, default=None)

    # Output
    parser.add_argument("--output-dir", default="outputs/grpo_medical_vllm")

    # QLoRA / LoRA
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="QLoRA: load base weights in 4-bit NF4 via bitsandbytes.",
    )
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # GRPO algorithm
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Rollouts G per prompt. With vLLM, G=8 is cheap — try it.")
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tool-calling-iterations", type=int, default=3)

    # Training
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)

    # Evaluation
    parser.add_argument("--eval-data-path", default=None)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--max-eval-samples", type=int, default=100)

    # Logging / saving
    parser.add_argument("--logging-steps", type=int, default=15)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument(
        "--report-to", default="none", choices=["none", "wandb", "tensorboard"]
    )
    parser.add_argument("--run-name", default="grpo-medical-vllm")
    parser.add_argument("--wandb-project", default="MedGRPO")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-mode", default="offline", choices=["online", "offline", "disabled"]
    )

    # Precision
    parser.add_argument("--dtype", default=None, choices=[None, "float16", "bfloat16"])

    # HF generation options (used when --use-vllm is off)
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
        help=(
            "Attention backend for the HF training model's backward pass. "
            "Not used by the vLLM engine (which picks its own kernel). "
            "'sdpa' works on GB10 without extra packages."
        ),
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=None,
        help=(
            "Total rollout sequences per generation phase. In TRL this affects "
            "both HF and vLLM paths by setting steps_per_generation; must be "
            "divisible by num_generations and the global train batch size."
        ),
    )
    parser.add_argument("--torch-empty-cache-steps", type=int, default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=2)

    # GDPO variant
    parser.add_argument(
        "--use-gdpo",
        action="store_true",
        default=False,
        help="Use GDPO reward design (orthogonal structure/answer/tool rewards).",
    )

    # ── vLLM acceleration ──────────────────────────────────────────────────
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        default=False,
        help=(
            "Enable vLLM colocate mode for rollout generation. "
            "Requires ./vllm_venv312/bin/python. "
            "3–5× faster training; does not change reward design or LoRA config."
        ),
    )
    parser.add_argument(
        "--vllm-gpu-mem-util",
        type=float,
        default=0.5,
        help=(
            "vLLM gpu_memory_utilization (0–1). Fraction of GPU VRAM reserved "
            "for vLLM KV cache during the generation phase. "
            "Default 0.5 leaves ~60 GB for training model + LoRA + optimizer; "
            "both resident simultaneously on GB10's 120 GB unified memory."
        ),
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=6144,
        help=(
            "vLLM max_model_len: max total tokens (prompt + completion + tool turns). "
            "Default 6144 covers system prompt (~500) + question (~200) + "
            "3 tool rounds (~1000) + max_completion_length (2048). "
            "Increase if you see 'Input too long' errors."
        ),
    )
    parser.add_argument(
        "--vllm-sleep-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable vLLM sleep mode (default OFF). "
            "WARNING: vLLM 0.20.0 sleep(level=2) reloads weights from disk on "
            "every wake_up(), adding ~34s overhead per step — SLOWER than baseline. "
            "On GB10 (120 GB unified) the two peaks fit without sleep mode. "
            "Only enable if you hit OOM with sleep=off."
        ),
    )
    parser.add_argument(
        "--vllm-model-impl",
        default="vllm",
        choices=["vllm", "transformers"],
        help=(
            "vLLM model implementation backend. "
            "'vllm' (default) uses vLLM's optimised CUDA kernels. "
            "'transformers' uses HF model inside vLLM — useful for debugging."
        ),
    )
    parser.add_argument(
        "--vllm-stop-at-tags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --use-vllm is set, stop generation as soon as a family-specific "
            "tool-call terminator or </answer> is generated, while keeping the "
            "stop string in the output. This prevents over-generation past a "
            "tool call so TRL can parse and execute the tool. Use "
            "--no-vllm-stop-at-tags to restore the old behavior."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dtype helpers (identical to grpo_train.py)
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


def get_vllm_stop_strings(family: ModelFamily) -> list[str]:
    """Keep vLLM generation aligned with the family-specific tool syntax."""
    if family == "llama":
        return ["<|eom_id|>", "<|eot_id|>", "</answer>"]
    return ["</tool_call>", "</answer>"]


# ---------------------------------------------------------------------------
# Smoke-test callback
# ---------------------------------------------------------------------------

_RAW_TOOL_JSON_RE = re.compile(
    r'"name"\s*:\s*"search_medical_knowledge"', re.DOTALL
)


class ToolParserSmokeTest(TrainerCallback):
    """Abort training if the tool parser is silently broken.

    Symptom: the model emits raw tool-call JSON in completions, but TRL's
    `tools/call_frequency` metric stays at 0 — meaning `parse_response` failed
    validation and dropped the structured `tool_calls`. Running 50h on this is
    pure waste, so we fail-fast at step `abort_after_step`.

    Trigger: after step `abort_after_step`, if `tools/call_frequency == 0`
    AND >= `raw_json_threshold` of the latest completions parquet contain raw
    tool JSON, raise RuntimeError.
    """

    def __init__(
        self,
        output_dir: str,
        abort_after_step: int = 10,
        raw_json_threshold: float = 0.30,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.abort_after_step = abort_after_step
        self.raw_json_threshold = raw_json_threshold
        self.triggered = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.triggered or logs is None:
            return
        if state.global_step < self.abort_after_step:
            return
        call_freq = logs.get("tools/call_frequency")
        if call_freq is None or call_freq > 0:
            return  # parser is working

        comp_dir = self.output_dir / "completions"
        if not comp_dir.exists():
            return  # nothing to inspect yet
        parquets = sorted(comp_dir.glob("completions_*.parquet"))
        if not parquets:
            return
        try:
            import pandas as pd
            df = pd.read_parquet(parquets[-1])
        except Exception:
            return  # don't fail on inspection error

        if "completion" not in df.columns or len(df) == 0:
            return
        raw_json_freq = (
            df["completion"].astype(str).str.contains(_RAW_TOOL_JSON_RE).mean()
        )
        if raw_json_freq >= self.raw_json_threshold:
            self.triggered = True
            raise RuntimeError(
                f"\n[SMOKE TEST FAILED] step={state.global_step}, "
                f"raw tool-JSON in {raw_json_freq:.0%} of completions "
                f"but tools/call_frequency=0.\n"
                f"  → Tool parser is dropping tool_calls during "
                f"_validate_tool_calls.\n"
                f"  → Check LLAMA_TOOL_SCHEMA in scripts/utils/model_adapter.py: "
                f"function.arguments must be present (not parameters).\n"
                f"  → Latest parquet: {parquets[-1]}"
            )


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

    # --- Print config banner ---
    vllm_label = (
        f"vLLM colocate  gpu_mem={args.vllm_gpu_mem_util}  "
        f"max_len={args.vllm_max_model_len}  sleep={args.vllm_sleep_mode}  "
        f"impl={args.vllm_model_impl}"
        if args.use_vllm else "HF transformers (no vLLM)"
    )
    print(f"\n{'='*60}")
    print(f"GRPO training  [{vllm_label}]")
    print(f"  model:  {args.model_path}")
    print(f"  output: {args.output_dir}")
    print(f"  G={args.num_generations}  lr={args.learning_rate}  β={args.beta}")
    print(f"{'='*60}\n")

    # --- Detect model family ---
    family: ModelFamily = (
        detect_family(args.model_path) if args.model_family == "auto" else args.model_family
    )
    print(f"Model family: {family}")

    # --- Pre-load retrieval tool ---
    print(f"Pre-loading retrieval tool from {args.data_dir} ...")
    MedicalKnowledgeTool.load(data_dir=args.data_dir)
    print("Retrieval tool ready.")

    # --- Load model and tokenizer ---
    print(f"Loading model from {args.model_path} (4-bit={args.load_in_4bit}) ...")
    torch_dtype = default_torch_dtype(args.dtype)
    model_kwargs: dict[str, Any] = {
        **get_model_load_kwargs(family),
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
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, **get_tokenizer_load_kwargs(family)
    )

    normalize_special_tokens(tokenizer, model, family, padding_side="left")
    tokenizer.response_schema = get_trl_response_schema(family)

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
    # Same calculation as grpo_train.py: TRL treats per_device_train_batch_size
    # as total sequences (prompts × G), so unique prompts per micro-step =
    # batch_size // G.
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
    prefetch = args.dataloader_prefetch_factor if args.dataloader_num_workers > 0 else None

    # vLLM colocate params — only injected when --use-vllm is set.
    # When off, GRPOConfig falls back to HF model.generate() exactly as in grpo_train.py.
    vllm_config: dict[str, Any] = {}
    if args.use_vllm:
        generation_kwargs: dict[str, Any] | None = None
        if args.vllm_stop_at_tags:
            generation_kwargs = {
                "stop": get_vllm_stop_strings(family),
                "include_stop_str_in_output": True,
            }
        vllm_config = dict(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_enable_sleep_mode=args.vllm_sleep_mode,
            vllm_gpu_memory_utilization=args.vllm_gpu_mem_util,
            vllm_max_model_length=args.vllm_max_model_len,
            vllm_model_impl=args.vllm_model_impl,
            vllm_tensor_parallel_size=1,  # single GPU on GB10
            generation_kwargs=generation_kwargs,
        )

    training_args = GRPOConfig(
        output_dir=args.output_dir,

        # Generation
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        # TRL uses this to derive steps_per_generation for both HF and vLLM.
        generation_batch_size=args.generation_batch_size,

        # Tool calling
        max_tool_calling_iterations=args.max_tool_calling_iterations,

        # RL algorithm
        loss_type="dapo" if args.use_gdpo else "grpo",
        beta=args.beta,
        epsilon=args.epsilon,
        scale_rewards="group",
        num_iterations=1,
        multi_objective_aggregation=(
            "normalize_then_sum" if args.use_gdpo else "sum_then_normalize"
        ),
        reward_weights=[0.20, 0.50, 0.30] if args.use_gdpo else [0.25, 0.50, 0.25],

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

        # Optimizer
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

        # Pin Today Date so Llama's tool template renders deterministically
        # across runs (default would interpolate today's system date).
        chat_template_kwargs={"date_string": "26 December 2024"},

        # vLLM colocate (empty dict → no-op when --use-vllm not set)
        **vllm_config,
    )

    # --- Trainer ---
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        reward_funcs=(
            [structure_reward, gdpo_answer_reward, tool_reward]
            if args.use_gdpo
            else [format_reward, answer_reward, enhanced_tool_quality_reward]
        ),
        tools=[search_medical_knowledge],
        peft_config=peft_config,
        callbacks=[ToolParserSmokeTest(args.output_dir)],
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
