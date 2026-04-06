#!/usr/bin/env python3
"""Validate and inspect Stage 1.5 generated traces (JSONL output).

Usage:
    # Full validation report
    python validate.py output/stage1_5_traces.jsonl

    # Show N random sample traces (full conversation)
    python validate.py output/stage1_5_traces.jsonl --show-samples 3

    # Re-run filters on saved traces (e.g. after tightening gate logic)
    python validate.py output/stage1_5_traces.jsonl --refilter
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Inline filter logic (mirror of gen_data_groq.py — no import needed)
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str | None:
    m = re.search(r"<answer>\s*([A-Ea-e])\b", text)
    return m.group(1).upper() if m else None


def filter_trace(trace: dict) -> tuple[bool, str]:
    """All 6 quality gates. Returns (passed, reason)."""
    messages = trace["messages"]
    expected = trace.get("answer_idx", "").upper()

    all_assistant = " ".join(
        m.get("content", "") or ""
        for m in messages if m["role"] == "assistant"
    )

    if not re.search(r"<think>.*?</think>", all_assistant, re.DOTALL):
        return False, "missing_think"

    predicted = extract_answer(all_assistant)
    if predicted is None:
        return False, "missing_answer"

    if expected and predicted != expected:
        return False, f"wrong_answer:{predicted}≠{expected}"

    for msg in messages:
        for tc in msg.get("tool_calls") or []:
            try:
                func = tc["function"]
                assert func["name"] == "search_medical_knowledge"
                args = func["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                query = args["query"]
                assert isinstance(query, str) and 2 <= len(query.split()) <= 25
            except Exception:
                return False, "invalid_tool_call_json"

    n_calls = sum(1 for m in messages if m.get("tool_calls"))
    n_resps = sum(1 for m in messages if m["role"] == "tool")
    if n_calls != n_resps:
        return False, f"call_response_mismatch:{n_calls}/{n_resps}"

    if n_calls > 0:
        think_blocks = re.findall(r"<think>(.*?)</think>", all_assistant, re.DOTALL)
        tool_contents = [m["content"] for m in messages if m["role"] == "tool"]
        if len(think_blocks) >= 2 and tool_contents:
            post_think = think_blocks[-1].lower()
            all_tool_text = " ".join(tool_contents).lower()
            meaningful = [w for w in all_tool_text.split() if len(w) > 5 and w.isalpha()]
            if meaningful and not any(w in post_think for w in meaningful[:20]):
                return False, "no_retrieval_integration"

    return True, "ok"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_trace(trace: dict, idx: int) -> str:
    """Human-readable trace dump."""
    lines = [
        f"\n{'─'*70}",
        f"Trace #{idx} | source={trace.get('source','?')} id={trace.get('question_id','?')} "
        f"type={trace.get('trace_type','?')} tool_calls={trace.get('num_tool_calls',0)}",
        f"Question: {trace.get('question','')[:120]}...",
        f"Expected answer: {trace.get('answer_idx','?')}",
        "Messages:",
    ]
    for m in trace["messages"]:
        role = m["role"].upper()
        content = (m.get("content") or "").strip()
        tcs = m.get("tool_calls", [])

        if role == "SYSTEM":
            lines.append(f"  [SYSTEM] (omitted, {len(content)} chars)")
        elif tcs:
            for tc in tcs:
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                query = args.get("query", args) if isinstance(args, dict) else args
                lines.append(f"  [ASSISTANT/TOOL_CALL] query=\"{query}\"")
            if content:
                # Show think block preview
                think = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                if think:
                    lines.append(f"    <think> {think.group(1).strip()[:200]} </think>")
        elif role == "TOOL":
            preview = content[:200].replace("\n", " | ")
            lines.append(f"  [TOOL_RESPONSE] {preview}")
        elif role == "ASSISTANT":
            think = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            answer = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
            if think:
                lines.append(f"  [ASSISTANT/THINK] {think.group(1).strip()[:200]}")
            if answer:
                lines.append(f"  [ASSISTANT/ANSWER] {answer.group(1).strip()}")
        else:
            lines.append(f"  [{role}] {content[:200]}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate(path: Path, show_samples: int, refilter: bool, seed: int) -> None:
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    traces: list[dict] = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: Bad JSON on line {i+1}: {e}")

    if not traces:
        print("No traces found.")
        return

    print(f"\n{'='*70}")
    print(f"VALIDATION REPORT: {path}")
    print(f"Total traces loaded: {len(traces)}")
    print(f"{'='*70}")

    # ── Structure stats ──────────────────────────────────────────────────
    sources: dict[str, int] = defaultdict(int)
    call_dist: dict[int, int] = defaultdict(int)
    type_dist: dict[str, int] = defaultdict(int)
    answer_dist: dict[str, int] = defaultdict(int)

    for t in traces:
        sources[t.get("source", "unknown")] += 1
        call_dist[t.get("num_tool_calls", 0)] += 1
        type_dist[t.get("trace_type", "unknown")] += 1

        all_asst = " ".join(
            m.get("content", "") or "" for m in t["messages"] if m["role"] == "assistant"
        )
        ans = extract_answer(all_asst)
        if ans:
            answer_dist[ans] += 1

    n = len(traces)
    print("\nSource distribution:")
    for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src:30s} {cnt:5d}  ({cnt/n:.1%})")

    print("\nTrace type distribution:")
    for tt, cnt in sorted(type_dist.items(), key=lambda x: -x[1]):
        print(f"  {tt:15s} {cnt:5d}  ({cnt/n:.1%})")

    print("\nTool call distribution:")
    for nc, cnt in sorted(call_dist.items()):
        bar = "█" * int(cnt / n * 40)
        print(f"  {nc} calls: {cnt:5d} ({cnt/n:.1%})  {bar}")

    print("\nAnswer distribution (A-E balance check):")
    for letter in "ABCDE":
        cnt = answer_dist.get(letter, 0)
        bar = "█" * int(cnt / n * 40)
        print(f"  {letter}: {cnt:5d} ({cnt/n:.1%})  {bar}")

    # ── Filter re-run ────────────────────────────────────────────────────
    if refilter:
        print(f"\n{'─'*70}")
        print("RE-RUNNING FILTERS on all traces...")
        gate_fails: dict[str, int] = defaultdict(int)
        passed_refilter = 0
        for t in traces:
            ok, reason = filter_trace(t)
            if ok:
                passed_refilter += 1
            else:
                gate_fails[reason] += 1

        print(f"  Passed: {passed_refilter}/{n} ({passed_refilter/n:.1%})")
        print("  Gate failures:")
        for reason, cnt in sorted(gate_fails.items(), key=lambda x: -x[1]):
            print(f"    {reason:40s} {cnt:5d} ({cnt/n:.1%})")

    # ── Format checks ────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("FORMAT CHECKS:")

    has_think = sum(
        1 for t in traces
        if re.search(r"<think>.*?</think>",
                     " ".join(m.get("content","") or "" for m in t["messages"]), re.DOTALL)
    )
    has_answer = sum(
        1 for t in traces
        if extract_answer(" ".join(m.get("content","") or "" for m in t["messages"]))
    )
    has_tool_and_response = sum(
        1 for t in traces
        if t.get("num_tool_calls", 0) > 0
        and sum(1 for m in t["messages"] if m["role"] == "tool") == t["num_tool_calls"]
    )

    print(f"  Has <think> block:           {has_think}/{n} ({has_think/n:.1%})")
    print(f"  Has <answer> letter:          {has_answer}/{n} ({has_answer/n:.1%})")
    print(f"  Tool calls matched by response: {has_tool_and_response}/"
          f"{sum(1 for t in traces if t.get('num_tool_calls',0)>0)} "
          f"(of tool-use traces)")

    # Average message length
    avg_turns = sum(len(t["messages"]) for t in traces) / n
    avg_chars = sum(
        sum(len(m.get("content","") or "") for m in t["messages"])
        for t in traces
    ) / n
    print(f"  Avg turns per trace:          {avg_turns:.1f}")
    print(f"  Avg total chars per trace:    {avg_chars:.0f}")

    # ── Sample traces ────────────────────────────────────────────────────
    if show_samples > 0:
        print(f"\n{'='*70}")
        print(f"SAMPLE TRACES (random {show_samples})")
        rng = random.Random(seed)
        indices = rng.sample(range(n), min(show_samples, n))
        for i in indices:
            print(fmt_trace(traces[i], i))

    print(f"\n{'='*70}")
    print("DONE")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Validate Stage 1.5 JSONL trace file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("path", type=Path, help="Path to JSONL file from gen_data_groq.py")
    p.add_argument("--show-samples", type=int, default=2,
                   help="Number of random sample traces to print in full")
    p.add_argument("--refilter", action="store_true",
                   help="Re-run all filter gates on saved traces")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    validate(args.path, args.show_samples, args.refilter, args.seed)


if __name__ == "__main__":
    main()
