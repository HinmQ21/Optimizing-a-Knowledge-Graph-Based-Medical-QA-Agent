"""Quick timing benchmark for _run_vllm_generation.

Mirrors _time_batch.py but routes through vLLM. Run from the vllm_venv312 venv.

Usage (from baseline/):
    ./vllm_venv312/bin/python -m scripts.benchmark.grpo_eval._time_vllm \
        --model-path outputs/grpo_medical_lora_v4_merged \
        --n-samples 100
"""

import argparse
import sys
import time

from datasets import load_from_disk
from transformers import AutoTokenizer
from vllm import LLM

from scripts.benchmark.grpo_eval.grpo_eval import (
    _run_vllm_generation,
    extract_answer_letter,
    get_vllm_force_answer_lp_class,
    normalize_row,
)
from scripts.utils.model_adapter import (
    get_tokenizer_load_kwargs,
    normalize_tokenizer_only,
)

NO_TOOL_SYSTEM = (
    "You are a medical reasoning assistant.\n\n"
    "Structure your response:\n"
    "1. <think>Your reasoning here</think>\n"
    "2. <answer>Your final answer</answer>\n\n"
    "IMPORTANT: In <answer> tags, write ONLY the option letter (e.g. A)."
)


def make_states(exs: list[dict], system: str) -> list[dict]:
    return [
        {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": (
                    f"{ex['question']}\n\nOptions:\n"
                    + "\n".join(f"{k}. {v}" for k, v in ex["options"].items())
                )},
            ],
            "tool_calls": [], "tool_responses": [], "query_texts": [],
            "n_tool_calls": 0, "final_content": "", "done": False,
        }
        for ex in exs
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="outputs/grpo_medical_lora_v4_merged")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-mem", type=float, default=0.7)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--no-force-answer", action="store_true",
                   help="Disable VllmForceAnswerLP (default: enabled).")
    p.add_argument("--min-tokens", type=int, default=0,
                   help="vLLM SamplingParams.min_tokens. Default: 0.")
    p.add_argument("--enforce-eager", action="store_true",
                   help="Disable CUDAGraphs (enforce_eager=True). Default: False (CUDAGraphs on).")
    args = p.parse_args()

    family = "qwen"
    eager_tag = "eager(no-CUDAGraphs)" if args.enforce_eager else "CUDAGraphs"
    print(f"Loading vLLM from {args.model_path}  [{eager_tag}] ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, **get_tokenizer_load_kwargs(family)
    )
    normalize_tokenizer_only(tokenizer, family, padding_side="left")
    lp_classes = []
    if not args.no_force_answer:
        lp_classes.append(get_vllm_force_answer_lp_class(tokenizer))
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        logits_processors=lp_classes if lp_classes else None,
    )
    print(f"Loaded in {time.time() - t0:.1f}s\n")

    ds = (
        load_from_disk("dataset/MedQA/test")
        .shuffle(seed=args.seed)
        .select(range(args.n_samples))
    )
    all_exs = [normalize_row(dict(ex)) for ex in ds]

    print(f"Benchmark: {args.n_samples} samples  max_new_tokens={args.max_new_tokens}  "
          f"temp={args.temperature}")
    print()

    # Single shot — vLLM batches internally.
    states = make_states(all_exs, NO_TOOL_SYSTEM)
    t_start = time.time()
    _run_vllm_generation(
        llm, tokenizer, states,
        no_tool=True, max_tool_iterations=0,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        family=family,
        force_answer=not args.no_force_answer,
        min_tokens=args.min_tokens,
    )
    elapsed = time.time() - t_start

    n_correct = sum(
        1 for s, ex in zip(states, all_exs)
        if extract_answer_letter(s["final_content"]) == ex["answer_idx"]
    )
    n_answered = sum(
        1 for s in states
        if s["final_content"] and "<answer>" in s["final_content"]
    )

    print(f"{'Engine':>22}  {'Time(s)':>8}  {'s/sample':>9}  {'Acc':>6}  {'AnswerRate':>10}")
    print("-" * 65)
    print(f"{eager_tag:>22}  {elapsed:>8.1f}  {elapsed/args.n_samples:>9.2f}  "
          f"{n_correct/args.n_samples:>6.1%}  {n_answered/args.n_samples:>10.1%}")
    sys.stdout.flush()
    print("\nDone.")


if __name__ == "__main__":
    main()
