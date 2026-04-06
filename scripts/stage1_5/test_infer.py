#!/usr/bin/env python3
"""Test inference on Stage 1.5 merged model with tool calling loop.

Runs a small sample of MedQA test questions through the model, executes
real KG retrieval for tool calls, and prints the full conversation for
visual inspection. Also reports basic metrics.

Usage:
    cd /home/vcsai/minhlbq/baseline
    ./training_venv312/bin/python -m scripts.stage1_5.test_infer \
        --model-path outputs/stage1_5_tool_sft_merged \
        --n-samples 5
"""

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.serve.retrieval_tool import MedicalKnowledgeTool, search_medical_knowledge
from scripts.train_rl.data_prep import SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Tool call parsing (Qwen2.5 ChatML format)
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def extract_tool_calls(text: str) -> list[dict]:
    """Extract tool_call JSON blocks from generated text."""
    calls = []
    for match in _TOOL_CALL_RE.finditer(text):
        try:
            tc = json.loads(match.group(1))
            calls.append(tc)
        except json.JSONDecodeError:
            pass
    return calls


def strip_tool_calls(text: str) -> str:
    """Remove <tool_call>...</tool_call> blocks to get clean assistant content."""
    return _TOOL_CALL_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Inference with tool loop
# ---------------------------------------------------------------------------

def generate_with_tools(
    model,
    tokenizer,
    question: str,
    options: dict,
    max_tool_iterations: int = 3,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
) -> dict:
    """Run model generation with tool calling loop."""
    opt_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    user_content = f"{question}\n\nOptions:\n{opt_text}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    tool_calls_made = []
    tool_responses = []

    for iteration in range(max_tool_iterations + 1):
        # Tokenize current conversation
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        # Generate
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Extract generated tokens
        new_tokens = output[0][input_ids.shape[1]:]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=False)

        # Strip trailing special tokens
        generated = generated.replace("<|im_end|>", "").strip()

        # Check for tool calls
        tool_calls = extract_tool_calls(generated)

        if tool_calls and iteration < max_tool_iterations:
            # Append assistant turn with tool call
            messages.append({"role": "assistant", "content": generated})
            tool_calls_made.extend(tool_calls)

            # Execute tool calls
            for tc in tool_calls:
                try:
                    args = tc.get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args)
                    query = args.get("query", "")
                    result = search_medical_knowledge(query)
                    tool_responses.append(result)
                    # Append tool response as user turn (Qwen2.5 convention)
                    messages.append({"role": "user", "content": f"<tool_response>\n{result}\n</tool_response>"})
                except Exception as e:
                    tool_responses.append(f"ERROR: {e}")
                    messages.append({"role": "user", "content": f"<tool_response>\nERROR: {e}\n</tool_response>"})
        else:
            # No tool call or max iterations reached: final answer
            messages.append({"role": "assistant", "content": generated})
            break

    return {
        "messages": messages,
        "tool_calls": tool_calls_made,
        "tool_responses": tool_responses,
        "n_tool_calls": len(tool_calls_made),
    }


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_LETTER_RE = re.compile(r"^\s*([A-Ea-e])[.\):\s]")


def extract_answer_letter(messages: list[dict]) -> str | None:
    """Extract the MCQ letter from <answer> tags in the final assistant turn."""
    for m in reversed(messages):
        if m["role"] != "assistant":
            continue
        content = m.get("content", "") or ""
        match = _ANSWER_RE.search(content)
        if not match:
            continue
        ans = match.group(1).strip()
        # Try letter patterns
        letter_match = _LETTER_RE.match(ans)
        if letter_match:
            return letter_match.group(1).upper()
        if len(ans) == 1 and ans.upper() in "ABCDE":
            return ans.upper()
        # Extract first letter A-E in the answer string
        for c in ans:
            if c.upper() in "ABCDE":
                return c.upper()
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-path", default="dataset/MedQA/test")
    p.add_argument("--data-dir", default="data/")
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--max-tool-iterations", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Load retrieval tool
    print("Loading KG retrieval tool ...")
    MedicalKnowledgeTool.load(data_dir=args.data_dir)
    print("KG ready.")

    # Load model
    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded. device={model.device}")

    # Load eval data
    print(f"Loading eval data from {args.data_path} ...")
    ds = load_from_disk(args.data_path)
    ds = ds.shuffle(seed=args.seed).select(range(args.n_samples))
    print(f"Evaluating on {len(ds)} samples\n")

    # Run inference
    results = []
    for i, ex in enumerate(ds):
        print(f"{'='*70}")
        print(f"Sample {i+1}/{len(ds)}")
        print(f"{'='*70}")
        print(f"Q: {ex['question'][:200]}...")
        print(f"Options: {ex['options']}")
        print(f"Correct: {ex['answer_idx']}. {ex['answer']}")
        print()

        result = generate_with_tools(
            model, tokenizer,
            question=ex["question"],
            options=ex["options"],
            max_tool_iterations=args.max_tool_iterations,
            temperature=args.temperature,
        )

        pred_letter = extract_answer_letter(result["messages"])
        correct = pred_letter == ex["answer_idx"]

        # Display the conversation
        print(f"Turns: {len(result['messages'])} | Tool calls: {result['n_tool_calls']}")
        for m in result["messages"][2:]:  # skip system+user
            role = m["role"]
            content = m.get("content", "") or ""
            if role == "assistant":
                print(f"\n[ASSISTANT]\n{content[:800]}")
                if len(content) > 800:
                    print(f"... ({len(content)-800} more chars)")
            elif role == "user":
                # Tool response
                print(f"\n[TOOL_RESPONSE]\n{content[:400]}")
                if len(content) > 400:
                    print(f"... ({len(content)-400} more chars)")

        print(f"\nPrediction: {pred_letter}, Correct: {ex['answer_idx']}, Match: {'✓' if correct else '✗'}")
        print()

        results.append({
            "idx": i,
            "pred": pred_letter,
            "correct_idx": ex["answer_idx"],
            "is_correct": correct,
            "n_tool_calls": result["n_tool_calls"],
            "n_turns": len(result["messages"]),
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n = len(results)
    n_correct = sum(1 for r in results if r["is_correct"])
    n_with_tool = sum(1 for r in results if r["n_tool_calls"] > 0)
    n_without_tool = n - n_with_tool

    print(f"Total samples:      {n}")
    print(f"Correct:            {n_correct}/{n} ({100*n_correct/n:.1f}%)")
    print(f"Tool calls used:    {n_with_tool}/{n} ({100*n_with_tool/n:.1f}%)")
    print(f"No tool:            {n_without_tool}/{n}")

    if n_with_tool > 0:
        correct_with = sum(1 for r in results if r["is_correct"] and r["n_tool_calls"] > 0)
        print(f"Accuracy WITH tool:    {correct_with}/{n_with_tool} ({100*correct_with/n_with_tool:.1f}%)")
    if n_without_tool > 0:
        correct_without = sum(1 for r in results if r["is_correct"] and r["n_tool_calls"] == 0)
        print(f"Accuracy WITHOUT tool: {correct_without}/{n_without_tool} ({100*correct_without/n_without_tool:.1f}%)")

    avg_calls = sum(r["n_tool_calls"] for r in results) / n
    print(f"Avg tool calls/sample: {avg_calls:.2f}")


if __name__ == "__main__":
    main()
