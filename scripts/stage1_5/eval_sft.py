#!/usr/bin/env python3
"""Full evaluation of Stage 1.5 SFT checkpoint on MedQA test set.

Generates completions with tool calling loop, then computes:
  - tool_call_frequency
  - answer_accuracy (overall)
  - answer_accuracy WITH vs WITHOUT tool call  ← CRITICAL decision gate
  - format compliance rates
  - retrieval quality (MedEmbed semantic similarity)
  - response length stats

Decision rule:
  WITH_tool > WITHOUT_tool  → Ready for GRPO Stage 2
  WITH_tool ≈ WITHOUT_tool  → Marginal, consider retrieval_grounding_reward
  WITH_tool < WITHOUT_tool  → Iterate Stage 1.5 data, not ready for GRPO

Usage:
    cd /home/vcsai/minhlbq/baseline
    ./training_venv312/bin/python -m scripts.stage1_5.eval_sft \
        --model-path outputs/stage1_5_tool_sft_v2_merged \
        --n-samples 100 \
        --output eval_results/stage1_5_v2_eval.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.serve.retrieval_tool import MedicalKnowledgeTool, search_medical_knowledge
from scripts.train_rl.data_prep import SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_LETTER_RE = re.compile(r"^\s*([A-Ea-e])[.\):\s]")


def extract_tool_calls(text: str) -> list[dict]:
    calls = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            calls.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            pass
    return calls


def extract_answer_letter(content: str) -> str | None:
    """Extract MCQ letter from <answer>X</answer>."""
    match = _ANSWER_RE.search(content)
    if not match:
        return None
    ans = match.group(1).strip()
    letter_match = _LETTER_RE.match(ans)
    if letter_match:
        return letter_match.group(1).upper()
    if len(ans) == 1 and ans.upper() in "ABCDE":
        return ans.upper()
    for c in ans:
        if c.upper() in "ABCDE":
            return c.upper()
    return None


# ---------------------------------------------------------------------------
# Generation with tool loop
# ---------------------------------------------------------------------------

def generate_with_tools(
    model, tokenizer, question: str, options: dict,
    max_tool_iterations: int = 3, max_new_tokens: int = 1024,
    temperature: float = 0.7,
) -> dict:
    opt_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{question}\n\nOptions:\n{opt_text}"},
    ]
    tool_calls_made = []
    tool_responses = []

    for iteration in range(max_tool_iterations + 1):
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature, do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            out[0][input_ids.shape[1]:], skip_special_tokens=False,
        ).replace("<|im_end|>", "").strip()

        tool_calls = extract_tool_calls(generated)
        if tool_calls and iteration < max_tool_iterations:
            messages.append({"role": "assistant", "content": generated})
            tool_calls_made.extend(tool_calls)
            for tc in tool_calls:
                try:
                    args = tc.get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args)
                    query = args.get("query", "")
                    result = search_medical_knowledge(query)
                    tool_responses.append(result)
                    messages.append({
                        "role": "user",
                        "content": f"<tool_response>\n{result}\n</tool_response>",
                    })
                except Exception as e:
                    tool_responses.append(f"ERROR: {e}")
                    messages.append({
                        "role": "user",
                        "content": f"<tool_response>\nERROR: {e}\n</tool_response>",
                    })
        else:
            messages.append({"role": "assistant", "content": generated})
            break

    return {
        "messages": messages,
        "tool_calls": tool_calls_made,
        "tool_responses": tool_responses,
        "n_tool_calls": len(tool_calls_made),
    }


# ---------------------------------------------------------------------------
# Retrieval quality scoring (MedEmbed cosine similarity)
# ---------------------------------------------------------------------------

def score_retrieval_relevance(encoder, question: str, answer_text: str,
                              retrieved_texts: list[str]) -> float:
    """Max cosine similarity between retrieved facts and (question + answer) anchor."""
    if not retrieved_texts:
        return 0.0
    anchor = f"{question} The answer is {answer_text}"
    embeddings = encoder.encode(
        [anchor] + retrieved_texts, normalize_embeddings=True,
    )
    anchor_emb = embeddings[0]
    retrieval_embs = embeddings[1:]
    sims = retrieval_embs @ anchor_emb
    return float(sims.max())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-path", default="dataset/MedQA/test")
    p.add_argument("--data-dir", default="data/")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--max-tool-iterations", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None, help="Optional JSON output path")
    p.add_argument("--score-retrieval", action="store_true",
                   help="Score retrieval relevance with MedEmbed (slower)")
    args = p.parse_args()

    # Load retrieval tool
    print("Loading KG retrieval tool ...")
    kg = MedicalKnowledgeTool.load(data_dir=args.data_dir)

    # Load model
    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16,
        trust_remote_code=True, device_map="auto",
    ).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load eval data
    print(f"Loading eval data from {args.data_path} ...")
    ds = load_from_disk(args.data_path)
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n_samples, len(ds))))
    print(f"Evaluating on {len(ds)} samples\n")

    # Run inference
    results = []
    for i, ex in enumerate(ds):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(ds)}] ...", flush=True)

        res = generate_with_tools(
            model, tokenizer,
            question=ex["question"], options=ex["options"],
            max_tool_iterations=args.max_tool_iterations,
            temperature=args.temperature,
        )

        # Extract final assistant content
        final_content = ""
        for m in reversed(res["messages"]):
            if m["role"] == "assistant":
                final_content = m.get("content", "") or ""
                break

        pred_letter = extract_answer_letter(final_content)
        correct = pred_letter == ex["answer_idx"]
        has_think = bool(_THINK_RE.search(final_content))
        has_answer = bool(_ANSWER_RE.search(final_content))

        # Response length
        total_len = sum(
            len((m.get("content") or "").split())
            for m in res["messages"]
            if m["role"] == "assistant"
        )

        # Retrieval relevance (optional)
        retrieval_score = None
        if args.score_retrieval and res["tool_responses"]:
            retrieval_score = score_retrieval_relevance(
                kg.encoder, ex["question"], ex["answer"],
                res["tool_responses"],
            )

        results.append({
            "idx": i,
            "question_id": ex.get("question_id", i),
            "pred": pred_letter,
            "correct_idx": ex["answer_idx"],
            "is_correct": correct,
            "n_tool_calls": res["n_tool_calls"],
            "n_turns": len(res["messages"]),
            "has_think": has_think,
            "has_answer": has_answer,
            "assistant_words": total_len,
            "retrieval_score": retrieval_score,
        })

    # ── Aggregate metrics ──
    n = len(results)
    n_correct = sum(1 for r in results if r["is_correct"])
    n_with_tool = sum(1 for r in results if r["n_tool_calls"] > 0)
    n_without_tool = n - n_with_tool
    n_has_think = sum(1 for r in results if r["has_think"])
    n_has_answer = sum(1 for r in results if r["has_answer"])
    n_pred_extracted = sum(1 for r in results if r["pred"] is not None)

    correct_with_tool = sum(1 for r in results if r["is_correct"] and r["n_tool_calls"] > 0)
    correct_without_tool = sum(1 for r in results if r["is_correct"] and r["n_tool_calls"] == 0)

    acc_overall = n_correct / n
    acc_with = correct_with_tool / n_with_tool if n_with_tool > 0 else 0.0
    acc_without = correct_without_tool / n_without_tool if n_without_tool > 0 else 0.0

    avg_tool_calls = sum(r["n_tool_calls"] for r in results) / n
    avg_response_len = sum(r["assistant_words"] for r in results) / n

    # ── Report ──
    print("\n" + "=" * 70)
    print(f"STAGE 1.5 SFT EVALUATION — {args.model_path}")
    print("=" * 70)
    print(f"Samples: {n}  |  Temperature: {args.temperature}  |  Max tool iters: {args.max_tool_iterations}")
    print()
    print("=== Tool Usage ===")
    print(f"  tool_call_frequency:      {n_with_tool/n:.3f}  ({n_with_tool}/{n})")
    print(f"  avg tool calls / sample:  {avg_tool_calls:.2f}")
    print(f"  multi_turn rate:          {sum(1 for r in results if r['n_tool_calls'] >= 2)/n:.3f}")
    print()
    print("=== Answer Accuracy ===")
    print(f"  Overall:                  {acc_overall:.3f}  ({n_correct}/{n})")
    print(f"  Prediction extracted:     {n_pred_extracted/n:.3f}  ({n_pred_extracted}/{n})")
    print(f"  WITH tool:                {acc_with:.3f}  ({correct_with_tool}/{n_with_tool})")
    print(f"  WITHOUT tool:             {acc_without:.3f}  ({correct_without_tool}/{n_without_tool})")
    diff = acc_with - acc_without
    print(f"  Difference (WITH - WITHOUT): {diff:+.3f}  ({diff*100:+.1f} pts)")
    print()
    print("=== Format Compliance ===")
    print(f"  Has <think>:              {n_has_think/n:.3f}  ({n_has_think}/{n})")
    print(f"  Has <answer>:             {n_has_answer/n:.3f}  ({n_has_answer}/{n})")
    print(f"  Avg response length:      {avg_response_len:.0f} words")
    print()

    if args.score_retrieval and n_with_tool > 0:
        scores = [r["retrieval_score"] for r in results if r["retrieval_score"] is not None]
        if scores:
            print("=== Retrieval Relevance (MedEmbed cosine) ===")
            print(f"  mean:  {np.mean(scores):.3f}")
            print(f"  p25:   {np.percentile(scores, 25):.3f}")
            print(f"  p50:   {np.percentile(scores, 50):.3f}")
            print(f"  p75:   {np.percentile(scores, 75):.3f}")
            # Correlation with correctness
            correct_scores = [r["retrieval_score"] for r in results if r["is_correct"] and r["retrieval_score"] is not None]
            wrong_scores = [r["retrieval_score"] for r in results if not r["is_correct"] and r["retrieval_score"] is not None]
            if correct_scores and wrong_scores:
                print(f"  when correct: {np.mean(correct_scores):.3f}")
                print(f"  when wrong:   {np.mean(wrong_scores):.3f}")
            print()

    # ── Decision gate ──
    print("=== GRPO Readiness Decision ===")
    if n_with_tool == 0 or n_without_tool == 0:
        print("  ⚠️  Cannot assess — all samples in one category")
    elif diff > 0.02:
        print(f"  ✅ READY for GRPO  (WITH_tool beats WITHOUT_tool by {diff*100:.1f} pts)")
    elif diff > -0.02:
        print(f"  ⚠️  MARGINAL  (diff = {diff*100:+.1f} pts)")
        print("     → Consider adding retrieval_grounding_reward to GRPO")
    else:
        print(f"  ❌ NOT READY  (WITH_tool worse by {-diff*100:.1f} pts)")
        print("     → Iterate Stage 1.5 data: filter MISS retrievals, rebalance tool ratio")

    # ── Save ──
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "model_path": args.model_path,
            "n_samples": n,
            "temperature": args.temperature,
            "max_tool_iterations": args.max_tool_iterations,
            "metrics": {
                "tool_call_frequency": n_with_tool / n,
                "avg_tool_calls": avg_tool_calls,
                "accuracy_overall": acc_overall,
                "accuracy_with_tool": acc_with,
                "accuracy_without_tool": acc_without,
                "accuracy_diff": diff,
                "has_think_rate": n_has_think / n,
                "has_answer_rate": n_has_answer / n,
                "avg_response_words": avg_response_len,
            },
            "per_sample": results,
        }
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()
