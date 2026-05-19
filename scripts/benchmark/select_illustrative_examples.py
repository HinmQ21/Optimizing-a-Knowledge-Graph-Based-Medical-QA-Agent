"""Cross-reference Stage 2 vs Baseline MedQA eval JSONs to find illustrative
examples where Stage 2 reasons correctly while Baseline fails.

Inputs: 2 JSON files from grpo_eval.py --save-completions.
The shuffled-MedQA order is stable across runs (seed=42), so per_sample[i]
in each file refers to the same question.

Outputs: ranked candidate list (top-N) with full reasoning chains.
"""

import argparse
import json
import re
from pathlib import Path


def load_eval(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    bench = d["benchmarks"]
    key = next(k for k in bench if k.endswith("MedQA/test"))
    return bench[key]


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*([A-D])\s*</answer>")


def score_candidate(s2: dict, s0: dict) -> float | None:
    """Higher score = better candidate for thesis illustration.

    Hard requirement: s2 correct AND baseline wrong.
    Soft rules:
      - Stage 2 used tool successfully (n_tool_calls > 0) = +2
      - Stage 2 has clean <think>/<answer> structure = +1
      - Reasonable think length (60-300 words) = +1
      - Retrieval cosine high (KG actually helped) = +1.5 if >0.7, +0.8 if >0.5
      - Baseline reasoning is substantive (not just a guess) = +0.5
        (proxy: baseline has_think AND think_words >= 30)
      - Stage 2 query is concise, not just copy-paste of question = +0.5
        (proxy: avg_query_copy_paste < 0.6)
    """
    if not s2["is_correct"]:
        return None
    if s0["is_correct"]:
        return None

    score = 2.0  # base for satisfying the hard requirement

    if s2["n_tool_calls"] > 0 and s2["has_answer"]:
        score += 2.0
    if s2["has_think"] and s2["has_answer"]:
        score += 1.0

    tw = s2.get("think_words", 0)
    if 60 <= tw <= 300:
        score += 1.0
    elif tw > 0 and tw < 60:
        score += 0.3

    rs = s2.get("retrieval_score")
    if rs is not None:
        if rs > 0.7:
            score += 1.5
        elif rs > 0.5:
            score += 0.8

    # Substantive (not gibberish) baseline reasoning makes contrast clearer
    if s0.get("has_think") and s0.get("think_words", 0) >= 30:
        score += 0.5

    qcp = s2.get("avg_query_copy_paste")
    if qcp is not None and qcp < 0.6:
        score += 0.5

    return score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--output", default="eval_results/illustrative/candidates_ranked.json")
    args = ap.parse_args()

    b0 = load_eval(args.baseline)
    b2 = load_eval(args.stage2)

    n = len(b2["per_sample"])
    assert len(b0["per_sample"]) == n, "baseline and stage2 sample counts differ"
    print(f"Loaded {n} samples per stage")

    n_s2_correct = sum(1 for s in b2["per_sample"] if s["is_correct"])
    n_s0_correct = sum(1 for s in b0["per_sample"] if s["is_correct"])
    print(f"  Stage 2 acc: {n_s2_correct}/{n} = {n_s2_correct/n:.1%}")
    print(f"  Baseline acc: {n_s0_correct}/{n} = {n_s0_correct/n:.1%}")

    candidates = []
    for i in range(n):
        s2 = b2["per_sample"][i]
        s0 = b0["per_sample"][i]

        # Cross-check same question
        if s0.get("question") and s2.get("question") and s0["question"] != s2["question"]:
            print(f"WARN: question mismatch at idx={i}")
            continue

        sc = score_candidate(s2, s0)
        if sc is None:
            continue

        cand = {
            "idx": i,
            "score": sc,
            "question": s2.get("question", ""),
            "options": s2.get("options", {}),
            "correct": s2["correct_idx"],
            "answer_text": s2.get("answer_text", ""),
            "preds": {
                "baseline": s0["pred"],
                "stage2": s2["pred"],
            },
            "stage2_meta": {
                "n_tool_calls": s2["n_tool_calls"],
                "think_words": s2.get("think_words"),
                "retrieval_score": s2.get("retrieval_score"),
                "query_texts": s2.get("query_texts", []),
                "avg_query_copy_paste": s2.get("avg_query_copy_paste"),
            },
            "baseline_meta": {
                "think_words": s0.get("think_words"),
                "has_think": s0.get("has_think"),
            },
            "completions": {
                "baseline": s0.get("final_content", ""),
                "stage2": s2.get("final_content", ""),
            },
            "stage2_full_messages": s2.get("messages", []),
            "stage2_tool_responses": s2.get("tool_responses", []),
        }
        candidates.append(cand)

    candidates.sort(key=lambda c: (-c["score"], c["idx"]))

    print(f"\nTotal candidates (Stage 2 correct AND baseline wrong): {len(candidates)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "n_candidates": len(candidates),
            "top_k": args.top_k,
            "candidates": candidates[:args.top_k],
        }, f, indent=2, ensure_ascii=False)
    print(f"Wrote top-{args.top_k} candidates → {out_path}")

    print(f"\n── Top {min(args.top_k, len(candidates))} candidates ──")
    for c in candidates[:args.top_k]:
        print(f"\nidx={c['idx']}  score={c['score']:.1f}")
        print(f"  Q: {c['question'][:140]}...")
        print(f"  preds: baseline={c['preds']['baseline']}  stage2={c['preds']['stage2']}  gold={c['correct']} ({c['answer_text'][:50]})")
        print(f"  S2: tool_calls={c['stage2_meta']['n_tool_calls']}  "
              f"think_words={c['stage2_meta']['think_words']}  "
              f"retr={c['stage2_meta']['retrieval_score']}  "
              f"qcp={c['stage2_meta']['avg_query_copy_paste']}")
        if c['stage2_meta']['query_texts']:
            print(f"  S2 query[0]: {c['stage2_meta']['query_texts'][0][:140]}")


if __name__ == "__main__":
    main()
