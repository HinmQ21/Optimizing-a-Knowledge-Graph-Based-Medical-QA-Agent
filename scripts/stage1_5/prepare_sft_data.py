#!/usr/bin/env python3
"""Prepare Stage 1.5 SFT dataset from raw teacher traces.

Filters and augments stage1_5_traces.jsonl to produce a balanced SFT dataset
with high tool-calling ratio (~85%) for training before GRPO.

Three augmentation strategies:
  1. Split multi-call traces into prefix sub-traces (teach varied tool depth)
  2. Verbose answer variants: <answer>B</answer> → <answer>B. full text</answer>
  3. Balanced no-tool samples (teach model when NOT to call tools)

Usage:
    cd /home/vcsai/minhlbq/baseline
    python -m scripts.stage1_5.prepare_sft_data \
        --traces data/stage1_5_traces.jsonl \
        --medqa dataset/MedQA/train \
        --output data/stage1_5_sft.jsonl \
        --max-no-tool 600
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from copy import deepcopy
from pathlib import Path

from datasets import load_from_disk


# ---------------------------------------------------------------------------
# Augmentation: split multi-call traces into prefix sub-traces
# ---------------------------------------------------------------------------

def split_multi_call(trace: dict) -> list[dict]:
    """Split a multi-call trace into shorter prefix sub-traces.

    A 3-call trace (sys→user→A1→T1→A2→T2→A3→T3→A_final) produces:
      - 1-call prefix: sys → user → A1 → T1 → A_final
      - 2-call prefix: sys → user → A1 → T1 → A2 → T2 → A_final

    The final assistant message (with <answer>) is reused as-is.
    This teaches the model that shorter tool chains can also reach the answer.
    """
    msgs = trace["messages"]
    n_calls = trace.get("num_tool_calls", 0)
    if n_calls < 2:
        return []

    # Locate assistant messages that contain tool_calls
    tool_call_indices = [i for i, m in enumerate(msgs) if m.get("tool_calls")]
    final_assistant = msgs[-1]  # Last message: assistant with <answer>

    if final_assistant["role"] != "assistant":
        return []

    sub_traces = []
    for k in range(1, len(tool_call_indices)):
        tool_call_idx = tool_call_indices[k - 1]
        tool_resp_idx = tool_call_idx + 1
        if tool_resp_idx >= len(msgs) or msgs[tool_resp_idx]["role"] != "tool":
            continue

        prefix = msgs[: tool_resp_idx + 1]
        new_trace = {
            **trace,
            "messages": prefix + [final_assistant],
            "num_tool_calls": k,
            "augment_type": f"split_{k}of{n_calls}",
        }
        sub_traces.append(new_trace)

    return sub_traces


# ---------------------------------------------------------------------------
# Augmentation: verbose answer variant
# ---------------------------------------------------------------------------

def make_verbose_answer(trace: dict, answer_text: str) -> dict | None:
    """Create a variant where <answer>B</answer> → <answer>B. full_text</answer>.

    Teaches the model to optionally produce more informative answers.
    """
    new_trace = deepcopy(trace)
    for m in reversed(new_trace["messages"]):
        if m["role"] != "assistant" or not m.get("content"):
            continue
        match = re.search(r"<answer>(.*?)</answer>", m["content"], re.DOTALL)
        if match:
            letter = match.group(1).strip()
            verbose = f"{letter}. {answer_text}"
            m["content"] = m["content"].replace(
                f"<answer>{match.group(1)}</answer>",
                f"<answer>{verbose}</answer>",
            )
            new_trace["augment_type"] = "verbose_answer"
            return new_trace
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter & augment Stage 1.5 traces for SFT training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--traces", default="data/stage1_5_traces.jsonl",
        help="Path to raw traces JSONL from gen_data_groq.py",
    )
    parser.add_argument(
        "--medqa", default="dataset/MedQA/train",
        help="Path to MedQA train dataset (for answer text lookup)",
    )
    parser.add_argument(
        "--output", default="data/stage1_5_sft.jsonl",
        help="Output path for augmented SFT dataset",
    )
    parser.add_argument(
        "--max-no-tool", type=int, default=600,
        help="Maximum number of no-tool traces to include",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ── Load traces ─────────────────────────────────────────────────────
    traces_path = Path(args.traces)
    if not traces_path.exists():
        print(f"ERROR: {traces_path} not found")
        sys.exit(1)

    raw_traces: list[dict] = []
    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw_traces.append(json.loads(line))
    print(f"Loaded {len(raw_traces)} raw traces from {traces_path}")

    # ── Load MedQA for answer text lookup ───────────────────────────────
    dq = load_from_disk(args.medqa)
    qid_to_answer: dict[int, str] = {}
    for i in range(len(dq)):
        qid_to_answer[i] = dq[i]["answer"]
    print(f"Loaded {len(qid_to_answer)} MedQA answer texts")

    # ── Split into tool / no-tool ───────────────────────────────────────
    with_tool = [t for t in raw_traces if t.get("num_tool_calls", 0) >= 1]
    without_tool = [t for t in raw_traces if t.get("num_tool_calls", 0) == 0]
    print(f"With tool: {len(with_tool)}, Without tool: {len(without_tool)}")

    # ── Build augmented dataset ─────────────────────────────────────────
    sft_data: list[dict] = []

    # 1. All tool-calling traces (base)
    for t in with_tool:
        t_copy = {**t, "augment_type": "original"}
        sft_data.append(t_copy)

    # 2. Split multi-call traces into prefix sub-traces
    split_count = 0
    for t in with_tool:
        for sub in split_multi_call(t):
            sft_data.append(sub)
            split_count += 1

    # 3. Verbose answer variants for tool traces
    verbose_count = 0
    for t in with_tool:
        qid = t.get("question_id")
        answer_text = qid_to_answer.get(qid)
        if answer_text:
            variant = make_verbose_answer(t, answer_text)
            if variant:
                sft_data.append(variant)
                verbose_count += 1

    # 4. Sample no-tool traces
    rng.shuffle(without_tool)
    no_tool_sample = without_tool[: args.max_no_tool]
    for t in no_tool_sample:
        t_copy = {**t, "augment_type": "no_tool"}
        sft_data.append(t_copy)

    # ── Shuffle ─────────────────────────────────────────────────────────
    rng.shuffle(sft_data)

    # ── Stats ───────────────────────────────────────────────────────────
    n = len(sft_data)
    n_tool = sum(1 for t in sft_data if t.get("num_tool_calls", 0) >= 1)
    aug_types: dict[str, int] = {}
    for t in sft_data:
        at = t.get("augment_type", "unknown")
        aug_types[at] = aug_types.get(at, 0) + 1

    total_words = sum(
        sum(len((m.get("content") or "").split()) for m in t["messages"])
        for t in sft_data
    )
    est_tokens = int(total_words * 1.3)

    print(f"\n{'='*60}")
    print(f"AUGMENTED DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples:       {n}")
    print(f"Tool-call samples:   {n_tool} ({100*n_tool/n:.1f}%)")
    print(f"No-tool samples:     {n - n_tool} ({100*(n-n_tool)/n:.1f}%)")
    print(f"Est. tokens:         {est_tokens:,} ({est_tokens/1e6:.1f}M)")
    print(f"\nAugmentation breakdown:")
    for at, cnt in sorted(aug_types.items(), key=lambda x: -x[1]):
        print(f"  {at:25s} {cnt:5d} ({100*cnt/n:.1f}%)")
    print(f"\nSources:")
    print(f"  Base tool traces:    {len(with_tool)}")
    print(f"  Split sub-traces:   +{split_count}")
    print(f"  Verbose answer:     +{verbose_count}")
    print(f"  No-tool sampled:    +{len(no_tool_sample)}")

    # ── Write ───────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for t in sft_data:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"\nWrote {n} traces to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
