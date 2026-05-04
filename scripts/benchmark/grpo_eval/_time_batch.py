"""Quick timing benchmark for _run_batched_generation across batch sizes.

Usage (from baseline/):
    ./training_venv312/bin/python -m scripts.benchmark.grpo_eval._time_batch \
        --model-path outputs/grpo_medical_lora_v4_merged \
        --n-samples 20
"""

import argparse
import sys
import time

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.benchmark.grpo_eval.grpo_eval import (
    _run_batched_generation,
    extract_answer_letter,
    normalize_row,
)
from scripts.utils.model_adapter import (
    get_model_load_kwargs,
    get_tokenizer_load_kwargs,
    normalize_special_tokens,
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
    p.add_argument("--n-samples",  type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    p.add_argument("--min-new-tokens", type=int, default=0)
    args = p.parse_args()

    family = "qwen"
    print(f"Loading model from {args.model_path} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, **get_tokenizer_load_kwargs(family)
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", **get_model_load_kwargs(family),
    ).eval()
    normalize_special_tokens(tokenizer, model, family, padding_side="left")
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s\n")

    ds = (
        load_from_disk("dataset/MedQA/test")
        .shuffle(seed=args.seed)
        .select(range(args.n_samples))
    )
    all_exs = [normalize_row(dict(ex)) for ex in ds]

    print(f"Benchmark: {args.n_samples} samples  max_new_tokens={args.max_new_tokens}  temp={args.temperature}  min_new={args.min_new_tokens}")
    print()
    print(f"{'BatchSize':>10}  {'Time(s)':>8}  {'s/sample':>9}  {'Speedup':>8}  {'Acc':>6}  {'AnswerRate':>10}")
    print("-" * 62)

    baseline_time: float | None = None

    for bs in args.batch_sizes:
        states = make_states(all_exs, NO_TOOL_SYSTEM)
        torch.cuda.synchronize()
        t_start = time.time()

        for batch_start in range(0, args.n_samples, bs):
            _run_batched_generation(
                model, tokenizer,
                states[batch_start: batch_start + bs],
                no_tool=True, max_tool_iterations=0,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                family=family,
                min_new_tokens=args.min_new_tokens,
            )

        torch.cuda.synchronize()
        elapsed = time.time() - t_start

        if baseline_time is None:
            baseline_time = elapsed

        n_correct = sum(
            1 for s, ex in zip(states, all_exs)
            if extract_answer_letter(s["final_content"]) == ex["answer_idx"]
        )
        n_answered = sum(1 for s in states if s["final_content"] and
                         "<answer>" in s["final_content"])
        acc         = n_correct / args.n_samples
        answer_rate = n_answered / args.n_samples
        speedup     = baseline_time / elapsed

        print(f"{bs:>10}  {elapsed:>8.1f}  {elapsed/args.n_samples:>9.2f}  "
              f"{speedup:>7.1f}×  {acc:>6.1%}  {answer_rate:>10.1%}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
