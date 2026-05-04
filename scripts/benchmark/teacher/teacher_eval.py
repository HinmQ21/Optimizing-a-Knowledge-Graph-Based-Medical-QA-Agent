#!/usr/bin/env python3
"""Teacher model (GPT-OSS 120B) benchmark on 4 medical QA datasets.

Evaluates on full test/eval splits:
  MedQA       1,273 samples   dataset/MedQA/test
  MedMCQA     4,183 samples   dataset/MedMCQA_4options_fixed/validation
  PubMedQA    1,000 samples   dataset/PubMedQA/train
  MedXpertQA  2,450 samples   dataset/MedXpertQA_Text/test

Usage:
    # Quick sanity check (10 samples per dataset)
    GROQ_API_KEY=gsk_... python -m scripts.benchmark.teacher.teacher_eval --test-samples 10

    # Full eval, all 4 datasets (~6h at 30 RPM)
    GROQ_API_KEY=gsk_... python -m scripts.benchmark.teacher.teacher_eval

    # Specific datasets only
    GROQ_API_KEY=gsk_... python -m scripts.benchmark.teacher.teacher_eval \
        --datasets medqa medmcqa

    # Cerebras (60 RPM, ~3h)
    CEREBRAS_API_KEY=... python -m scripts.benchmark.teacher.teacher_eval \
        --provider cerebras

    # Resume after interruption
    GROQ_API_KEY=gsk_... python -m scripts.benchmark.teacher.teacher_eval --resume

    # Higher concurrency + custom RPM (if on paid tier)
    GROQ_API_KEY=gsk_... python -m scripts.benchmark.teacher.teacher_eval \
        --concurrency 8 --rpm 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

_BASELINE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_BASELINE_DIR))

try:
    from openai import AsyncOpenAI
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

try:
    from datasets import load_from_disk
except ImportError:
    print("ERROR: pip install datasets")
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_dotenv() -> None:
    """Load key=value pairs from .env, without overriding shell exports.

    Search order (first found wins):
      1. Same directory as this script
      2. baseline/ project root
    """
    candidates = [
        Path(__file__).parent / ".env",                  # scripts/benchmark/teacher/.env
        _BASELINE_DIR / ".env",                          # baseline/.env
        _BASELINE_DIR / "scripts" / "stage1_5" / ".env", # shared with gen_data_groq.py
    ]
    for env_path in candidates:
        if not env_path.exists():
            continue
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        logger.debug(f"Loaded .env from {env_path}")
        return

# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, dict] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "default_model": "openai/gpt-oss-120b",
        "default_rpm": 30,
        "price_input": 0.15,
        "price_output": 0.60,
        "price_cached": 0.075,
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "env_key": "CEREBRAS_API_KEY",
        "default_model": "gpt-oss-120b",
        "default_rpm": 60,
        "price_input": 0.35,
        "price_output": 0.75,
        "price_cached": 0.00,
    },
}

DATASET_CONFIGS: dict[str, dict] = {
    "medqa": {
        "path": "dataset/MedQA",
        "split": "test",
        "display": "MedQA",
        "answer_type": "letter",
    },
    "medmcqa": {
        "path": "dataset/MedMCQA_4options_fixed",
        "split": "validation",
        "display": "MedMCQA",
        "answer_type": "letter",
    },
    "pubmedqa": {
        "path": "dataset/PubMedQA",
        "split": "train",
        "display": "PubMedQA",
        "answer_type": "yesnomaybe",
    },
    "medxpertqa": {
        "path": "dataset/MedXpertQA_Text",
        "split": "test",
        "display": "MedXpertQA",
        "answer_type": "letter",
    },
}

_SYSTEM_MCQ = (
    "You are a medical expert. Read the question carefully and select the single best answer "
    "from the options provided. Reason step by step, then end your response with your final "
    'answer on a new line in the exact format: "Answer: X" where X is the letter of your choice.'
)

_SYSTEM_PUBMEDQA = (
    "You are a medical research expert. You will be given an abstract from a biomedical study "
    "and a yes/no/maybe question about its conclusion. Read carefully and reason step by step, "
    'then end your response with "Answer: yes", "Answer: no", or "Answer: maybe".'
)

# ---------------------------------------------------------------------------
# Rate limiter & cost tracker
# ---------------------------------------------------------------------------


class RateLimiter:
    """Async token-bucket rate limiter that allows true concurrent in-flight requests.

    The lock is released before sleeping, so multiple coroutines can each
    reserve their own send-slot and sleep concurrently. API calls overlap
    while still honoring the per-minute request cap.
    """

    def __init__(self, rpm: int) -> None:
        self._min_interval = 60.0 / max(rpm, 1)
        self._next_allowed = 0.0  # monotonic time when the next slot opens
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            send_at = max(now, self._next_allowed)
            self._next_allowed = send_at + self._min_interval
        # Sleep outside the lock — other coroutines can reserve their slot in parallel
        wait = send_at - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)


@dataclass
class CostTracker:
    price_input: float
    price_output: float
    price_cached: float
    total_input: int = 0
    total_output: int = 0
    total_cached: int = 0

    def add(self, usage: Any) -> None:
        if usage is None:
            return
        self.total_input += getattr(usage, "prompt_tokens", 0)
        self.total_output += getattr(usage, "completion_tokens", 0)
        cached = 0
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0)
        self.total_cached += cached

    @property
    def cost(self) -> float:
        billed_input = self.total_input - self.total_cached
        return (
            billed_input * self.price_input / 1_000_000
            + self.total_cached * self.price_cached / 1_000_000
            + self.total_output * self.price_output / 1_000_000
        )

    def summary(self) -> str:
        return (
            f"${self.cost:.3f} | "
            f"in={self.total_input:,} out={self.total_output:,} "
            f"cached={self.total_cached:,}"
        )


# ---------------------------------------------------------------------------
# Dataset loading & prompt building
# ---------------------------------------------------------------------------


def _build_mcq_prompt(question: str, options: dict[str, str]) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    return f"Question: {question}\n\nOptions:\n{opts}"


def load_medqa(base_dir: Path, n_samples: int | None) -> list[dict]:
    ds = load_from_disk(str(base_dir / "dataset/MedQA"))["test"]
    rows = []
    for i, row in enumerate(ds):
        if n_samples is not None and i >= n_samples:
            break
        rows.append({
            "id": str(i),
            "prompt": _build_mcq_prompt(row["question"], row["options"]),
            "system": _SYSTEM_MCQ,
            "gold": row["answer_idx"].strip().upper(),
            "answer_type": "letter",
            "dataset": "medqa",
            "question": row["question"],
        })
    return rows


def load_medmcqa(base_dir: Path, n_samples: int | None) -> list[dict]:
    ds = load_from_disk(str(base_dir / "dataset/MedMCQA_4options_fixed"))["validation"]
    rows = []
    for i, row in enumerate(ds):
        if n_samples is not None and i >= n_samples:
            break
        rows.append({
            "id": str(i),
            "prompt": _build_mcq_prompt(row["question"], row["options"]),
            "system": _SYSTEM_MCQ,
            "gold": row["answer_idx"].strip().upper(),
            "answer_type": "letter",
            "dataset": "medmcqa",
            "question": row["question"],
        })
    return rows


def load_pubmedqa(base_dir: Path, n_samples: int | None) -> list[dict]:
    ds = load_from_disk(str(base_dir / "dataset/PubMedQA"))["train"]
    rows = []
    for i, row in enumerate(ds):
        if n_samples is not None and i >= n_samples:
            break
        abstract = "\n\n".join(row["context"]["contexts"])
        prompt = f"Abstract:\n{abstract}\n\nQuestion: {row['question']}"
        rows.append({
            "id": str(row["pubid"]),
            "prompt": prompt,
            "system": _SYSTEM_PUBMEDQA,
            "gold": row["final_decision"].strip().lower(),
            "answer_type": "yesnomaybe",
            "dataset": "pubmedqa",
            "question": row["question"],
        })
    return rows


def load_medxpertqa(base_dir: Path, n_samples: int | None) -> list[dict]:
    ds = load_from_disk(str(base_dir / "dataset/MedXpertQA_Text"))["test"]
    rows = []
    for i, row in enumerate(ds):
        if n_samples is not None and i >= n_samples:
            break
        question_text = row["question"]
        # Strip embedded "Answer Choices: ..." block so we can reconstruct cleanly
        if "\nAnswer Choices:" in question_text:
            question_text = question_text[: question_text.index("\nAnswer Choices:")].strip()
        rows.append({
            "id": row["id"],
            "prompt": _build_mcq_prompt(question_text, row["options"]),
            "system": _SYSTEM_MCQ,
            "gold": row["label"].strip().upper(),
            "answer_type": "letter",
            "dataset": "medxpertqa",
            "question": question_text,
            "medical_task": row.get("medical_task"),
            "body_system": row.get("body_system"),
            "question_type": row.get("question_type"),
        })
    return rows


LOADERS = {
    "medqa": load_medqa,
    "medmcqa": load_medmcqa,
    "pubmedqa": load_pubmedqa,
    "medxpertqa": load_medxpertqa,
}

# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------


def parse_letter(text: str) -> str | None:
    # "Answer: X" pattern (primary)
    m = re.search(r"[Aa]nswer\s*:\s*\**([A-Ja-j])\b", text)
    if m:
        return m.group(1).upper()
    # Single letter on its own line (common fallback)
    for line in reversed(text.strip().splitlines()):
        line = line.strip().rstrip(".")
        if re.match(r"^[A-Ja-j]$", line):
            return line.upper()
    # "the answer is X"
    m = re.search(r"(?:the\s+)?answer\s+is\s+\**([A-Ja-j])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Bold letter at end: "**A**"
    m = re.search(r"\*\*([A-Ja-j])\*\*\s*$", text.strip())
    if m:
        return m.group(1).upper()
    return None


def parse_yesnomaybe(text: str) -> str | None:
    m = re.search(r"[Aa]nswer\s*:\s*\**(yes|no|maybe)\b", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    for line in reversed(text.strip().splitlines()):
        line = line.strip().lower().rstrip(".")
        if line in ("yes", "no", "maybe"):
            return line
    # last resort: find the last yes/no/maybe word
    matches = list(re.finditer(r"\b(yes|no|maybe)\b", text.lower()))
    if matches:
        return matches[-1].group(1)
    return None


def parse_answer(text: str, answer_type: str) -> str | None:
    if answer_type == "letter":
        return parse_letter(text)
    if answer_type == "yesnomaybe":
        return parse_yesnomaybe(text)
    return None

# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------


async def call_api(
    client: AsyncOpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> tuple[str, Any]:
    """Return (content, usage). Rate limiting is handled by the caller."""
    for attempt in range(max_retries + 1):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or "", resp.usage
        except Exception as exc:
            if attempt < max_retries:
                wait = 5 * (2 ** attempt)
                logger.warning(f"API error (attempt {attempt + 1}/{max_retries + 1}): {exc}. Retry in {wait}s")
                await asyncio.sleep(wait)
            else:
                logger.error(f"API permanently failed: {exc}")
                return "", None


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------


async def eval_sample(
    sample: dict,
    client: AsyncOpenAI,
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    rate_limiter: RateLimiter,
    cost_tracker: CostTracker,
    sem: asyncio.Semaphore,
    in_flight: list,  # [int] mutable counter, incremented while API call is active
) -> dict:
    async with sem:
        await rate_limiter.acquire()
        in_flight[0] += 1
        try:
            content, usage = await call_api(
                client, model, sample["system"], sample["prompt"],
                temperature, max_tokens, max_retries,
            )
        finally:
            in_flight[0] -= 1
        cost_tracker.add(usage)
        pred = parse_answer(content, sample["answer_type"])
        result: dict = {
            "id": sample["id"],
            "dataset": sample["dataset"],
            "question": sample["question"],
            "gold": sample["gold"],
            "predicted": pred,
            "correct": pred is not None and pred == sample["gold"],
            "response": content,
        }
        for key in ("medical_task", "body_system", "question_type"):
            if key in sample:
                result[key] = sample[key]
        return result


# ---------------------------------------------------------------------------
# Dataset-level evaluation loop
# ---------------------------------------------------------------------------


async def eval_dataset(
    samples: list[dict],
    client: AsyncOpenAI,
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    rate_limiter: RateLimiter,
    cost_tracker: CostTracker,
    concurrency: int,
    output_file: Path,
    done_ids: set[str],
) -> list[dict]:
    pending = [s for s in samples if s["id"] not in done_ids]
    logger.info(f"  {len(pending)} to evaluate  ({len(done_ids)} already done, skipping)")

    sem = asyncio.Semaphore(concurrency)
    in_flight: list = [0]  # mutable int shared across coroutines
    new_results: list[dict] = []

    with open(output_file, "a") as fout:
        with tqdm(total=len(pending), unit="sample", leave=False) as pbar:
            tasks = [
                eval_sample(
                    s, client, model, temperature, max_tokens,
                    max_retries, rate_limiter, cost_tracker, sem, in_flight,
                )
                for s in pending
            ]
            for future in asyncio.as_completed(tasks):
                result = await future
                new_results.append(result)
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                pbar.update(1)
                pbar.set_postfix(inf=in_flight[0], cost=cost_tracker.summary(), refresh=False)

    return new_results


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------


def compute_accuracy(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {"accuracy": 0.0, "correct": 0, "total": 0, "invalid": 0}
    correct = sum(1 for r in results if r["correct"])
    invalid = sum(1 for r in results if r["predicted"] is None)
    return {"accuracy": correct / total, "correct": correct, "total": total, "invalid": invalid}


def compute_medxpertqa_breakdown(results: list[dict]) -> dict:
    """Per-task and per-body-system breakdown for MedXpertQA."""
    by_task: dict[str, list] = {}
    by_system: dict[str, list] = {}
    for r in results:
        task = r.get("medical_task") or "Unknown"
        sys_ = r.get("body_system") or "Unknown"
        by_task.setdefault(task, []).append(r["correct"])
        by_system.setdefault(sys_, []).append(r["correct"])
    return {
        "by_medical_task": {k: {"accuracy": sum(v) / len(v), "n": len(v)} for k, v in by_task.items()},
        "by_body_system": {k: {"accuracy": sum(v) / len(v), "n": len(v)} for k, v in by_system.items()},
    }


def compute_pubmedqa_breakdown(results: list[dict]) -> dict:
    """Per-label distribution for PubMedQA."""
    by_gold: dict[str, list] = {}
    for r in results:
        by_gold.setdefault(r["gold"], []).append(r["correct"])
    return {
        "by_gold_label": {k: {"accuracy": sum(v) / len(v), "n": len(v)} for k, v in by_gold.items()}
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark teacher (GPT-OSS 120B) on MedQA / MedMCQA / PubMedQA / MedXpertQA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--provider", choices=list(PROVIDERS), default="groq",
                   help="API provider")
    p.add_argument("--model", default=None,
                   help="Override model name (default: provider default)")
    p.add_argument("--datasets", nargs="+", choices=list(DATASET_CONFIGS),
                   default=list(DATASET_CONFIGS),
                   help="Which datasets to evaluate")
    p.add_argument("--data-dir", default=".",
                   help="Base directory containing dataset/")
    p.add_argument("--output-dir", default="eval_results/teacher_benchmark",
                   help="Directory for per-dataset JSONL files and summary JSON")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (0 = greedy)")
    p.add_argument("--max-tokens", type=int, default=2048,
                   help="Maximum tokens per response")
    p.add_argument("--concurrency", type=int, default=5,
                   help="Number of concurrent API requests")
    p.add_argument("--rpm", type=int, default=None,
                   help="Override requests-per-minute (default: provider tier limit)")
    p.add_argument("--max-retries", type=int, default=2,
                   help="Max retries per sample on API error")
    p.add_argument("--test-samples", type=int, default=None,
                   help="Limit N samples per dataset (for quick testing)")
    p.add_argument("--resume", action="store_true",
                   help="Skip samples already saved in existing JSONL output")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    args = parse_args()
    cfg = PROVIDERS[args.provider]

    _load_dotenv()

    api_key = os.environ.get(cfg["env_key"])
    if not api_key:
        logger.error(
            f"Missing API key — set {cfg['env_key']} in shell, "
            f"scripts/benchmark/teacher/.env, or {_BASELINE_DIR}/.env"
        )
        sys.exit(1)

    model = args.model or cfg["default_model"]
    rpm = args.rpm or cfg["default_rpm"]

    client = AsyncOpenAI(api_key=api_key, base_url=cfg["base_url"])
    rate_limiter = RateLimiter(rpm)
    cost_tracker = CostTracker(cfg["price_input"], cfg["price_output"], cfg["price_cached"])

    base_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Provider : {args.provider}  |  Model: {model}")
    logger.info(f"RPM: {rpm}  |  Concurrency: {args.concurrency}  |  Temp: {args.temperature}")
    logger.info(f"Datasets : {args.datasets}")
    logger.info(f"Output   : {out_dir}")
    if args.test_samples:
        logger.info(f"TEST MODE — capping at {args.test_samples} samples per dataset")

    start_time = time.time()
    all_results: dict[str, list[dict]] = {}

    for ds_key in args.datasets:
        ds_cfg = DATASET_CONFIGS[ds_key]
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {ds_cfg['display']}  (split={ds_cfg['split']})")

        samples = LOADERS[ds_key](base_dir, args.test_samples)
        logger.info(f"  Loaded {len(samples)} samples")

        jsonl_path = out_dir / f"{ds_key}.jsonl"

        # Resume: read already-completed samples
        done_ids: set[str] = set()
        existing_results: list[dict] = []
        if args.resume and jsonl_path.exists():
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        if r["id"] not in done_ids:
                            done_ids.add(r["id"])
                            existing_results.append(r)
                    except (json.JSONDecodeError, KeyError):
                        pass
            logger.info(f"  Resume: {len(done_ids)} already done")

        new_results = await eval_dataset(
            samples, client, model, args.temperature, args.max_tokens,
            args.max_retries, rate_limiter, cost_tracker,
            args.concurrency, jsonl_path, done_ids,
        )

        combined = existing_results + new_results
        all_results[ds_key] = combined

        acc = compute_accuracy(combined)
        logger.info(
            f"  {ds_cfg['display']:12s}: {acc['accuracy']:.2%}  "
            f"({acc['correct']}/{acc['total']}  invalid={acc['invalid']})"
        )
        logger.info(f"  Cost so far: {cost_tracker.summary()}")

    # Build and save summary
    elapsed = time.time() - start_time
    summary: dict[str, Any] = {
        "provider": args.provider,
        "model": model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "test_samples_cap": args.test_samples,
        "elapsed_seconds": round(elapsed, 1),
        "total_cost_usd": round(cost_tracker.cost, 4),
        "total_input_tokens": cost_tracker.total_input,
        "total_output_tokens": cost_tracker.total_output,
        "total_cached_tokens": cost_tracker.total_cached,
        "datasets": {},
    }

    for ds_key in args.datasets:
        ds_cfg = DATASET_CONFIGS[ds_key]
        results = all_results.get(ds_key, [])
        acc = compute_accuracy(results)
        entry: dict[str, Any] = {
            **acc,
            "split": ds_cfg["split"],
            "jsonl": str(out_dir / f"{ds_key}.jsonl"),
        }
        if ds_key == "medxpertqa" and results:
            entry["breakdown"] = compute_medxpertqa_breakdown(results)
        if ds_key == "pubmedqa" and results:
            entry["breakdown"] = compute_pubmedqa_breakdown(results)
        summary["datasets"][ds_cfg["display"]] = entry

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print final table
    logger.info(f"\n{'='*60}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'='*60}")
    for ds_name, metrics in summary["datasets"].items():
        logger.info(
            f"  {ds_name:15s}: {metrics['accuracy']:.2%}  "
            f"({metrics['correct']:>5}/{metrics['total']:<5}  "
            f"invalid={metrics['invalid']})"
        )
    logger.info(f"\n  Total cost : ${cost_tracker.cost:.4f}")
    logger.info(f"  Elapsed    : {elapsed / 60:.1f} min")
    logger.info(f"  Summary    : {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
