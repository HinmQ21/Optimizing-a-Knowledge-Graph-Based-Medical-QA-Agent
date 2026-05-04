#!/usr/bin/env python3
"""Rewrite pre-tool <think> blocks using reusable assets in existing traces.

Problem: Current traces have ~100% synthesized pre-tool thinks like
  "<think>I need to look up X.</think>"
which teach the student to call tools WITHOUT clinical deliberation,
erasing Stage 1's MedReason style.

Solution (no API calls): build richer pre-tool thinks from assets
already in each trace:
  1. metamap_phrases from MedQA  → scenario anchors
  2. leading sentences of final <think> → clinical framing / differential
  3. original tool query              → knowledge gap

Output: new traces JSONL; structure identical to input (same message
ordering, same tool_calls, same tool responses, same final answer).
Only pre-tool <think> content is replaced.

Usage:
    cd /home/vcsai/minhlbq/baseline
    ./training_venv312/bin/python -m scripts.stage1_5.rewrite_pretool_think \
        --traces data/stage1_5_traces.jsonl \
        --medqa dataset/MedQA/train \
        --output data/stage1_5_traces_v2.jsonl

    # Smoke test: dump 5 before/after samples, no save
    ./training_venv312/bin/python -m scripts.stage1_5.rewrite_pretool_think --smoke 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from datasets import load_from_disk


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Substring patterns — if any appears in a metamap phrase (lowercased),
# the phrase is rejected. Substring match handles variations like
# "year old man presents", "presents to the primary care clinic", etc.
_METAMAP_STOP_SUBSTR = (
    "year old", "years old", "year-old", "y/o", "yo ",
    "found to", "patient with", "patient presents", "presents to",
    "presents with", "most likely", "physiological consequence",
    "clinical picture", "laboratory findings", "past medical",
    "medical history", "physical examination", "examination reveals",
    "primary care", "emergency department", "clinic visit",
    "falls asleep", "falling asleep", "wakes up", "came to",
    "reports feeling", "complains of", "brought to",
)

# Exact single-token rejections (full-phrase match, lowercased)
_METAMAP_STOP_EXACT = frozenset({
    "male", "female", "boy", "girl", "man", "woman", "child",
    "adult", "infant", "teenager", "elderly", "pt", "patient",
    "old", "age", "bmi",
})

# Conclusion markers — sentences with these are the model's answer reasoning
_CONCLUSION_MARKERS = (
    "therefore", "thus", "the answer is", "the correct answer",
    "most likely answer", "so the answer", "hence", "the correct choice",
    "this corresponds to", "matches option", "among the options",
    "option a", "option b", "option c", "option d",
)

# KG-citation markers — sentences citing retrieved facts must NOT appear
# in pre-tool framing (the search hasn't happened yet at that point).
_KG_CITATION_MARKERS = (
    "knowledge graph", "kg confirms", "kg shows", "kg indicates",
    "retrieved facts", "retrieved information", "the retrieval",
    "as retrieved", "search results", "the search confirm",
    "search returned", "tool returned", "tool result",
    "confirmed by the", "according to the graph", "primekg",
)


def select_metamap_anchors(metamap: list[str], max_n: int = 5) -> list[str]:
    """Keep metamap phrases that look like clinical findings, not generic."""
    keep: list[str] = []
    seen: set[str] = set()
    for ph in metamap:
        p = ph.strip()
        low = p.lower()
        if low in _METAMAP_STOP_EXACT:
            continue
        if any(sub in low for sub in _METAMAP_STOP_SUBSTR):
            continue
        if len(p) < 4 or len(p) > 60:
            continue
        if p.isdigit():
            continue
        if low in seen:
            continue
        seen.add(low)
        keep.append(p)
    # Prefer longer (more informative) phrases
    keep.sort(key=lambda s: (-len(s.split()), -len(s)))
    return keep[:max_n]


def split_final_think(final_think: str) -> tuple[str, str]:
    """Split a final <think> into (framing, conclusion).

    Framing  = leading sentences before any conclusion marker
               (clinical reasoning used to motivate a search).
    Conclusion = remainder (cites KG facts, states the answer).

    If no conclusion marker found, takes ~60% front as framing.
    """
    text = final_think.strip()
    if not text:
        return "", ""

    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sents:
        return "", ""

    cut = None
    for i, s in enumerate(sents):
        low = s.lower()
        # Stop BEFORE any KG citation — those sentences reference facts the
        # pre-tool reasoning cannot yet know about.
        if any(mk in low for mk in _KG_CITATION_MARKERS):
            cut = i
            break
        if any(mk in low for mk in _CONCLUSION_MARKERS):
            cut = i
            break

    if cut is None:
        # No explicit marker — front 60%, but at least 1 sent, at most 3
        cut = max(1, min(3, round(len(sents) * 0.6)))

    cut = max(1, min(cut, 3))  # clamp to 1..3 sentences for pre-tool framing
    framing = " ".join(sents[:cut])
    conclusion = " ".join(sents[cut:])
    return framing, conclusion


def extract_tool_query(assistant_msg: dict) -> str:
    """Pull the first tool query text from an assistant message."""
    tcs = assistant_msg.get("tool_calls") or []
    if not tcs:
        return ""
    try:
        args = tcs[0]["function"]["arguments"]
        if isinstance(args, str):
            args = json.loads(args)
        return args.get("query", "") or ""
    except (json.JSONDecodeError, KeyError, TypeError):
        return ""


def build_pretool_think(
    anchors: list[str],
    framing: str,
    tool_query: str,
) -> str:
    """Compose a three-part pre-tool <think> block.

    Parts:
      Scenario         — anchor concepts from metamap
      Initial reasoning — leading clinical framing from final think
      Knowledge gap    — what the upcoming tool call is meant to verify
    """
    lines = ["<think>"]

    if anchors:
        scenario = "; ".join(anchors[:4])
        lines.append(f"Scenario: {scenario}.")

    if framing:
        # Trim trailing period duplication
        f = framing.rstrip()
        if not f.endswith((".", "!", "?")):
            f += "."
        lines.append(f"Initial assessment: {f}")

    if tool_query:
        # Normalize to "I need to verify ..." style
        q = tool_query.strip().rstrip(".?!")
        lines.append(
            f"Knowledge gap: I need to verify {q} before finalizing the answer."
        )
    else:
        lines.append(
            "Knowledge gap: I need to verify the key clinical relationship "
            "before finalizing the answer."
        )

    lines.append("</think>")
    return "\n".join(lines)


def rewrite_trace(
    trace: dict,
    qid_to_metamap: dict[int, list[str]],
) -> tuple[dict, str]:
    """Return (new_trace, status_tag).

    status_tag ∈ {"rewritten", "skipped_no_tool", "skipped_no_final_think",
                  "skipped_missing_qid"}.
    """
    msgs = trace["messages"]
    n_calls = trace.get("num_tool_calls", 0)

    if n_calls == 0:
        return trace, "skipped_no_tool"

    # Find first assistant with tool_calls
    first_tc_idx = None
    for i, m in enumerate(msgs):
        if m["role"] == "assistant" and m.get("tool_calls"):
            first_tc_idx = i
            break
    if first_tc_idx is None:
        return trace, "skipped_no_tool"

    # Find final assistant with <answer>
    final_think = ""
    for m in reversed(msgs):
        if m["role"] == "assistant" and m.get("content"):
            mt = _THINK_RE.search(m["content"])
            if mt:
                final_think = mt.group(1).strip()
                break
    if not final_think:
        return trace, "skipped_no_final_think"

    qid = trace.get("question_id", -1)
    metamap = qid_to_metamap.get(qid, [])
    anchors = select_metamap_anchors(metamap)

    framing, _ = split_final_think(final_think)

    first_tc_msg = msgs[first_tc_idx]
    tool_query = extract_tool_query(first_tc_msg)

    new_think = build_pretool_think(anchors, framing, tool_query)

    # Replace content; KEEP tool_calls untouched
    new_msgs = [dict(m) for m in msgs]
    new_msgs[first_tc_idx] = {
        **first_tc_msg,
        "content": new_think,
    }

    new_trace = {**trace, "messages": new_msgs, "rewritten_pretool": True}
    return new_trace, "rewritten"


def load_metamap(medqa_path: str) -> dict[int, list[str]]:
    ds = load_from_disk(medqa_path)
    # Accept either DatasetDict (with 'train') or raw Dataset
    if hasattr(ds, "column_names") and isinstance(ds.column_names, dict):
        ds = ds["train"]
    qid_to_mp: dict[int, list[str]] = {}
    for i in range(len(ds)):
        qid_to_mp[i] = ds[i].get("metamap_phrases", []) or []
    return qid_to_mp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite pre-tool <think> in Stage 1.5 traces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--traces", default="data/stage1_5_traces.jsonl")
    parser.add_argument("--medqa", default="dataset/MedQA/train")
    parser.add_argument(
        "--output", default="data/stage1_5_traces_v2.jsonl",
        help="Output JSONL; not used in --smoke mode",
    )
    parser.add_argument(
        "--smoke", type=int, default=0,
        help="Smoke test: print N before/after samples and exit (no save)",
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Compute stats without writing",
    )
    args = parser.parse_args()

    traces_path = Path(args.traces)
    if not traces_path.exists():
        print(f"ERROR: {traces_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading metamap from {args.medqa} ...")
    qid_to_metamap = load_metamap(args.medqa)
    print(f"  loaded metamap for {len(qid_to_metamap)} questions")

    print(f"Loading traces from {traces_path} ...")
    traces: list[dict] = []
    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    print(f"  loaded {len(traces)} traces")

    # Smoke mode: show first N tool-call traces before/after
    if args.smoke > 0:
        print("\n" + "=" * 70)
        print(f"SMOKE TEST — {args.smoke} samples")
        print("=" * 70)
        shown = 0
        for t in traces:
            if t.get("num_tool_calls", 0) == 0:
                continue
            new_t, tag = rewrite_trace(t, qid_to_metamap)
            if tag != "rewritten":
                continue
            shown += 1
            print(f"\n--- Sample {shown} (qid={t['question_id']}, "
                  f"n_calls={t['num_tool_calls']}) ---")
            # BEFORE
            old_first = next(
                m for m in t["messages"]
                if m["role"] == "assistant" and m.get("tool_calls")
            )
            new_first = next(
                m for m in new_t["messages"]
                if m["role"] == "assistant" and m.get("tool_calls")
            )
            print("[BEFORE pre-tool]")
            print(f"  {old_first.get('content','')}")
            print("[AFTER pre-tool]")
            print(f"  {new_first['content']}")
            # Sanity: tool_calls preserved
            old_tcs = old_first.get("tool_calls") or []
            new_tcs = new_first.get("tool_calls") or []
            assert len(old_tcs) == len(new_tcs), "tool_calls count changed"
            for oc, nc in zip(old_tcs, new_tcs):
                assert oc["function"]["arguments"] == nc["function"]["arguments"], \
                    "tool arguments changed"
            # Sanity: final answer preserved
            old_final = t["messages"][-1].get("content", "")
            new_final = new_t["messages"][-1].get("content", "")
            assert old_final == new_final, "final message changed"
            print("  [sanity] tool_calls & final answer preserved ✓")
            if shown >= args.smoke:
                break
        print("\n" + "=" * 70)
        print("Smoke test complete.")
        return

    # Full rewrite
    status_counts: dict[str, int] = {}
    new_traces: list[dict] = []
    for t in traces:
        nt, tag = rewrite_trace(t, qid_to_metamap)
        status_counts[tag] = status_counts.get(tag, 0) + 1
        new_traces.append(nt)

    print("\nRewrite status:")
    for k, v in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:30s} {v}")

    if args.stats_only:
        return

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for t in new_traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(new_traces)} traces to {out}")
    print(f"File size: {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
