"""Post-GRPO evaluation script.

Extends Stage 1.5 eval_sft.py with:
  - Multi-benchmark support (all datasets below)
  - Forced no-tool ablation mode  (--no-tool)
  - Self-consistency majority vote (--self-consistency --sc-samples N)
  - Query quality scoring         (copy-paste vs reformulation)
  - Think-block depth analysis
  - GRPO-specific comparison metrics

Supported datasets (pass as --benchmarks path):
  dataset/MedQA/test                              MedQA 4-opt A-D
  dataset/MedMCQA_4options_fixed/test             MedMCQA 4-opt A-D
  dataset/MedXpertQA_Text/test                    MedXpertQA 10-opt A-J
  dataset/PubMedQA/train                          PubMedQA yes/no/maybe (raw)
  dataset/BioMed-R1-Eval/medmcqa/test             BioMed MedMCQA 4-opt
  dataset/BioMed-R1-Eval/pubmedqa/test            BioMed PubMedQA A/B/C
  dataset/BioMed-R1-Eval/GPQA_Medical_test/test   GPQA Medical 4-opt
  dataset/BioMed-R1-Eval/Lancet/test              Lancet case MCQ 4-opt
  dataset/BioMed-R1-Eval/NEJM/test                NEJM case MCQ 4-5-opt
  dataset/BioMed-R1-Eval/MedXpertQA/test          BioMed MedXpertQA 10-opt
  dataset/BioMed-R1-Eval/medbullets_op4/test      MedBullets 4-opt
  dataset/BioMed-R1-Eval/medbullets_op5/test      MedBullets 5-opt
  dataset/BioMed-R1-Eval/mmlu_health_biology/test MMLU health/bio 10-opt
  dataset/BioMed-R1-Eval/hle_biomed/test          HLE BioMed 10-opt

Schema normalization (handled automatically by normalize_row):
  options as JSON string  → json.loads()              (all BioMed-R1-Eval)
  label field (no answer_idx) → answer_idx            (MedXpertQA_Text)
  final_decision + context.contexts                   (raw PubMedQA)
    → context prepended to question, options={A:yes, B:no, C:maybe}

Output JSON:
  {
    "model_path": "...",
    "mode": "with_tool" | "no_tool",
    "benchmarks": {
      "MedQA/test": { "metrics": {...}, "per_sample": [...] },
      ...
    }
  }

Usage:
    cd /home/vcsai/minhlbq/baseline

    # Merge LoRA adapter first (if needed):
    ./training_venv312/bin/python scripts/finetune/merge_peft_adapter.py \\
        --adapter-path outputs/grpo_medical_lora_v4/checkpoint-200 \\
        --output-dir outputs/grpo_medical_lora_v4_merged

    # Standard eval — multiple benchmarks in one run:
    ./training_venv312/bin/python -m scripts.benchmark.grpo_eval.grpo_eval \\
        --model-path outputs/grpo_medical_lora_v4_merged \\
        --benchmarks dataset/MedQA/test dataset/MedMCQA_4options_fixed/test \\
            dataset/BioMed-R1-Eval/GPQA_Medical_test/test \\
            dataset/BioMed-R1-Eval/MedXpertQA/test \\
            dataset/BioMed-R1-Eval/pubmedqa/test \\
        --n-samples 200 --score-retrieval \\
        --output eval_results/grpo_v4_multi.json

    # No-tool ablation:
    ./training_venv312/bin/python -m scripts.benchmark.grpo_eval.grpo_eval \\
        --model-path outputs/grpo_medical_lora_v4_merged \\
        --no-tool \\
        --benchmarks dataset/MedQA/test \\
        --n-samples 200 \\
        --output eval_results/grpo_v4_no_tool.json

    # Self-consistency (majority vote, 5 samples) — use training temperature:
    ./training_venv312/bin/python -m scripts.benchmark.grpo_eval.grpo_eval \\
        --model-path outputs/grpo_medical_lora_v4_merged \\
        --benchmarks dataset/MedQA/test \\
        --n-samples 200 --temperature 0.8 \\
        --self-consistency --sc-samples 5 \\
        --output eval_results/grpo_v4_sc5.json

    # SC ablation without tool:
    ./training_venv312/bin/python -m scripts.benchmark.grpo_eval.grpo_eval \\
        --model-path outputs/grpo_medical_lora_v4_merged \\
        --no-tool --temperature 0.8 \\
        --self-consistency --sc-samples 5 \\
        --benchmarks dataset/MedQA/test \\
        --n-samples 200 \\
        --output eval_results/grpo_v4_notool_sc5.json

    # LoRA adapter directly (auto-detects adapter_config.json):
    ./training_venv312/bin/python -m scripts.benchmark.grpo_eval.grpo_eval \\
        --model-path outputs/grpo_medical_lora_v4/checkpoint-200 \\
        --benchmarks dataset/MedQA/test \\
        --n-samples 200 \\
        --output eval_results/grpo_v4_lora_with_tool.json

SC output metrics (per benchmark):
    sc_accuracy_majority:      majority@k accuracy (== accuracy_overall when SC on)
    sc_accuracy_greedy:        pass@1 — first sample only (proxy for greedy)
    sc_pass_at_k:              pass@k — correct if any sample gets it right
    sc_majority_vs_greedy_pts: lift of majority vote over greedy (pts)
    sc_vote_confidence_mean:   avg fraction of votes for winning answer (1.0=unanimous)
    sc_vote_confidence_p50:    median vote confidence
    sc_unanimous_rate:         fraction of questions where all votes agree

Per-sample SC fields:
    sc_vote_counts:     {letter: count} over k samples
    sc_vote_confidence: majority_count / n_valid_votes
    sc_unanimous:       all valid votes agree
    sc_all_preds:       list of all k predictions
    sc_greedy_pred:     prediction from sample 0
    sc_pass_at_k:       any sample correct
"""

import argparse
import importlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from scripts.serve.retrieval_tool import MedicalKnowledgeTool, search_medical_knowledge
from scripts.train_rl.data_prep import SYSTEM_PROMPT
from scripts.utils.model_adapter import (
    ModelFamily,
    detect_family,
    get_eos_for_generation,
    get_model_load_kwargs,
    get_strip_tokens,
    get_trl_response_schema,
    get_tokenizer_load_kwargs,
    get_tool_call_regex,
    get_tool_response_message,
    get_tools_for_template,
    normalize_special_tokens,
)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

NO_TOOL_SYSTEM_PROMPT = (
    "You are a medical reasoning assistant.\n\n"
    "Structure your response:\n"
    "1. <think>Your reasoning here</think>\n"
    "2. <answer>Your final answer</answer>\n\n"
    "IMPORTANT: In <answer> tags, write ONLY the option letter (e.g. A) "
    "or a short answer, NOT an explanation."
)


# ---------------------------------------------------------------------------
# Regex / tokenization helpers
# ---------------------------------------------------------------------------

# _TOOL_CALL_RE is set per-run via get_tool_call_regex(family) after model is loaded.
# Kept as None here; initialised in main() and threaded through generate_* functions.
_TOOL_CALL_RE = None
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_LETTER_RE = re.compile(r"^\s*([A-Ja-j])[.\):\s]")  # A-J for up to 10-option MCQs
_VALID_LETTERS = frozenset("ABCDEFGHIJ")
_STANDALONE_LETTER_RE = re.compile(r"\b([A-Ja-j])\b")  # standalone letter in free text
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "and", "or", "not", "with", "that",
    "this", "which", "what", "how", "does", "do", "from",
})


def _tokenize(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def _fmt_eta(secs: float) -> str:
    """Format seconds as Hh:MMm or MM.Ss for compact display."""
    if secs < 60:
        return f"{secs:5.1f}s"
    if secs < 3600:
        m = int(secs // 60); s = int(secs - m * 60)
        return f"{m:>3d}m{s:02d}s"
    h = int(secs // 3600); m = int((secs - h * 3600) // 60)
    return f"{h:>2d}h{m:02d}m"


def _print_progress(
    done: int, total: int, t_start: float, n_correct: int,
    prefix: str = "    ", extra: str = "",
) -> None:
    """Uniform progress line: [done/total]  rate  elapsed  ETA  acc=…"""
    elapsed = time.time() - t_start
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    acc = n_correct / done if done > 0 else 0.0
    pct = 100 * done / total if total else 0
    bar = f"[{done:>5}/{total} {pct:>4.1f}%]"
    rate_str = f"{rate:>5.1f}/s" if rate > 0 else "  ·  "
    print(
        f"{prefix}{bar}  {rate_str}  "
        f"elapsed {_fmt_eta(elapsed)}  ETA {_fmt_eta(eta)}  "
        f"acc={acc:>5.1%} ({n_correct}/{done}){extra}",
        flush=True,
    )


def _copy_paste_ratio(query: str, question: str) -> float:
    """Fraction of query tokens present in the question.

    1.0 = pure copy-paste; low = reformulated / focused query.
    """
    q_tokens = _tokenize(query)
    src_tokens = _tokenize(question)
    if not q_tokens:
        return 0.0
    return len(q_tokens & src_tokens) / len(q_tokens)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def extract_tool_calls(text: str, tool_re: re.Pattern) -> list[dict]:
    calls = []
    for m in tool_re.finditer(text):
        try:
            calls.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            pass
    return calls


def extract_answer_letter(content: str) -> str | None:
    """Extract MCQ letter from <answer>X</answer>. Supports A-J (up to 10 options)."""
    match = _ANSWER_RE.search(content)
    if not match:
        return None
    ans = match.group(1).strip()
    letter_match = _LETTER_RE.match(ans)
    if letter_match:
        return letter_match.group(1).upper()
    if len(ans) == 1 and ans.upper() in _VALID_LETTERS:
        return ans.upper()
    matches = _STANDALONE_LETTER_RE.findall(ans)
    if matches:
        return matches[-1].upper()
    return None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_with_tools(
    model, tokenizer, question: str, options: dict, system_prompt: str,
    max_tool_iterations: int = 3, max_new_tokens: int = 1024,
    temperature: float = 0.3, family: ModelFamily = "qwen",
) -> dict:
    """Run one sample through the model with (optional) tool loop."""
    opt_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{question}\n\nOptions:\n{opt_text}"},
    ]
    tool_calls_made: list[dict] = []
    tool_responses: list[str] = []
    query_texts: list[str] = []

    tool_re = get_tool_call_regex(family)
    for iteration in range(max_tool_iterations + 1):
        text = tokenizer.apply_chat_template(
            messages, tools=get_tools_for_template(family),
            add_generation_prompt=True, tokenize=False,
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
                eos_token_id=get_eos_for_generation(family, tokenizer),
                use_cache=True,
            )
        from scripts.utils.model_adapter import strip_generation_artifacts
        raw_text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=False)
        tool_calls = extract_tool_calls(raw_text, tool_re)
        generated = strip_generation_artifacts(raw_text, family)

        if tool_calls and iteration < max_tool_iterations:
            messages.append({"role": "assistant", "content": generated})
            tool_calls_made.extend(tool_calls)
            for tc in tool_calls:
                try:
                    # Qwen uses "arguments", Llama uses "parameters"
                    args = tc.get("arguments") or tc.get("parameters", {})
                    if isinstance(args, str):
                        args = json.loads(args)
                    query = args.get("query", "")
                    if query:
                        query_texts.append(query)
                    result = search_medical_knowledge(query)
                    tool_responses.append(result)
                    messages.append(get_tool_response_message(result, family))
                except Exception as e:
                    tool_responses.append(f"ERROR: {e}")
                    messages.append(get_tool_response_message(f"ERROR: {e}", family))
        else:
            messages.append({"role": "assistant", "content": generated})
            break

    return {
        "messages": messages,
        "tool_calls": tool_calls_made,
        "tool_responses": tool_responses,
        "query_texts": query_texts,
        "n_tool_calls": len(tool_calls_made),
    }


def generate_no_tool(
    model, tokenizer, question: str, options: dict,
    max_new_tokens: int = 1024, temperature: float = 0.3,
    family: ModelFamily = "qwen",
) -> dict:
    """Single-turn generation without tool loop."""
    from scripts.utils.model_adapter import strip_generation_artifacts
    opt_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    messages = [
        {"role": "system", "content": NO_TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": f"{question}\n\nOptions:\n{opt_text}"},
    ]
    text = tokenizer.apply_chat_template(
        messages, tools=get_tools_for_template(family),
        add_generation_prompt=True, tokenize=False,
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
            eos_token_id=get_eos_for_generation(family, tokenizer),
            use_cache=True,
        )
    generated = strip_generation_artifacts(
        tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=False),
        family,
    )

    return {
        "messages": messages + [{"role": "assistant", "content": generated}],
        "tool_calls": [],
        "tool_responses": [],
        "query_texts": [],
        "n_tool_calls": 0,
    }


# ---------------------------------------------------------------------------
# Self-consistency (majority vote)
# ---------------------------------------------------------------------------

def generate_with_selfconsistency(
    model, tokenizer, question: str, options: dict,
    no_tool: bool, system_prompt: str,
    max_tool_iterations: int, max_new_tokens: int,
    temperature: float, sc_samples: int, family: ModelFamily = "qwen",
) -> dict:
    """Batched self-consistency: all sc_samples run in parallel per turn.

    Strategy — turn-level batching:
      - Each tool-loop iteration packs all still-active samples into a single
        model.generate(batch_size=n_active) call instead of n_active serial calls.
      - After generation, samples that produced tool calls get their tool
        responses injected and remain active; finished samples are graduated out.
      - With left-padding, heterogeneous message histories (post-tool divergence)
        are handled correctly each turn by re-tokenizing fresh.

    Speedup vs sequential: ~sc_samples× GPU calls per turn.
    E.g. sc_samples=5, 1 tool call → 2 batched calls instead of 10 serial calls.

    Returns:
        pred:              majority-vote answer letter (or None if no valid votes)
        vote_counts:       {letter: count} over valid samples
        vote_confidence:   majority_count / n_valid_votes  (1.0 = unanimous)
        n_valid_votes:     number of samples that produced an extractable answer
        all_preds:         [pred_0, pred_1, ...] for pass@k analysis
        greedy_pred:       pred from sample 0 (pass@1 proxy)
        rep_res:           full generation dict from the representative sample
                           (first sample whose pred matches the majority vote)
        rep_final_content: final assistant content of the representative sample
    """
    opt_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    system = NO_TOOL_SYSTEM_PROMPT if no_tool else system_prompt
    base_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{question}\n\nOptions:\n{opt_text}"},
    ]

    # Per-sample mutable state
    states = [
        {
            "messages": [m.copy() for m in base_messages],
            "tool_calls": [],
            "tool_responses": [],
            "query_texts": [],
            "n_tool_calls": 0,
            "final_content": "",
            "done": False,
        }
        for _ in range(sc_samples)
    ]

    orig_padding_side = tokenizer.padding_side

    for iteration in range(max_tool_iterations + 1):
        active_idx = [i for i, s in enumerate(states) if not s["done"]]
        if not active_idx:
            break

        # ── Tokenize all active histories (left-pad for decoder-only generation) ──
        texts = [
            tokenizer.apply_chat_template(
                states[i]["messages"], tools=get_tools_for_template(family),
                add_generation_prompt=True, tokenize=False,
            )
            for i in active_idx
        ]

        tokenizer.padding_side = "left"
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        tokenizer.padding_side = orig_padding_side

        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)
        prompt_len = input_ids.shape[1]  # same for all after left-padding

        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=get_eos_for_generation(family, tokenizer),
                use_cache=True,
            )  # [n_active, prompt_len + gen_len]

        # ── Process each active sample's output ──
        from scripts.utils.model_adapter import strip_generation_artifacts
        tool_re = get_tool_call_regex(family)
        for batch_pos, sample_idx in enumerate(active_idx):
            state = states[sample_idx]
            raw_text = tokenizer.decode(out[batch_pos][prompt_len:], skip_special_tokens=False)
            tool_calls = extract_tool_calls(raw_text, tool_re) if not no_tool else []
            generated = strip_generation_artifacts(raw_text, family)

            if tool_calls and iteration < max_tool_iterations:
                # Inject tool response and keep active for next iteration
                state["messages"].append({"role": "assistant", "content": generated})
                state["n_tool_calls"] += len(tool_calls)
                state["tool_calls"].extend(tool_calls)
                for tc in tool_calls:
                    try:
                        # Qwen uses "arguments", Llama uses "parameters"
                        args = tc.get("arguments") or tc.get("parameters", {})
                        if isinstance(args, str):
                            args = json.loads(args)
                        query = args.get("query", "")
                        if query:
                            state["query_texts"].append(query)
                        result = search_medical_knowledge(query)
                        state["tool_responses"].append(result)
                        state["messages"].append(get_tool_response_message(result, family))
                    except Exception as e:
                        state["tool_responses"].append(f"ERROR: {e}")
                        state["messages"].append(get_tool_response_message(f"ERROR: {e}", family))
            else:
                # No tool call or max iterations reached: graduate this sample
                state["messages"].append({"role": "assistant", "content": generated})
                state["final_content"] = generated
                state["done"] = True

    # ── Build sample dicts ──
    samples = []
    for state in states:
        res = {
            "messages": state["messages"],
            "tool_calls": state["tool_calls"],
            "tool_responses": state["tool_responses"],
            "query_texts": state["query_texts"],
            "n_tool_calls": state["n_tool_calls"],
        }
        pred = extract_answer_letter(state["final_content"])
        samples.append({"res": res, "pred": pred, "final_content": state["final_content"]})

    # ── Majority vote ──
    valid_preds = [s["pred"] for s in samples if s["pred"] is not None]
    if valid_preds:
        vote_counts = Counter(valid_preds)
        majority_pred, majority_count = vote_counts.most_common(1)[0]
        vote_confidence = majority_count / len(valid_preds)
    else:
        vote_counts = Counter()
        majority_pred = None
        vote_confidence = 0.0

    # Representative sample: first whose pred agrees with the majority vote
    rep_sample = next(
        (s for s in samples if s["pred"] == majority_pred),
        samples[0],
    )

    return {
        "pred": majority_pred,
        "vote_counts": dict(vote_counts),
        "vote_confidence": vote_confidence,
        "n_valid_votes": len(valid_preds),
        "all_preds": [s["pred"] for s in samples],
        "greedy_pred": samples[0]["pred"],
        "rep_res": rep_sample["res"],
        "rep_final_content": rep_sample["final_content"],
    }


# ---------------------------------------------------------------------------
# LogitsProcessor: suppress EOS until any valid ending appears at tail
# ---------------------------------------------------------------------------

class MultiEndingProcessor(LogitsProcessor):
    """Suppress EOS for each batch sample until ANY of the valid endings
    appears at the tail of the generated tokens.

    Replaces the older single-ending ForceAnswerProcessor. The list of valid
    endings is determined by the caller per-iteration:
      - final-answer iteration:  endings = [</answer>]
      - intermediate tool iter:  endings = [</tool_call>, </answer>]

    This way EVERY iteration in a tool-calling loop has proper EOS-suppression
    regardless of whether the model wants to emit a tool call or a direct answer.
    `min_new_tokens` is no longer required for correctness.

    Instantiate fresh per model.generate() call (state is per-call).
    Per-sample completion is tracked, so samples that produce a valid ending
    early can stop naturally while slower samples in the same batch continue.

    Args:
        endings: list of token-id sequences. Each is checked at the tail.
        eos_ids: all EOS token IDs to suppress until any ending appears.
    """

    def __init__(self, endings: list[list[int]], eos_ids: list[int]) -> None:
        # Sort by descending length so longer endings (e.g. </tool_call>) are
        # checked before any shorter endings that could appear as a suffix.
        self._endings: list[list[int]] = sorted(endings, key=len, reverse=True)
        self._max_end_len: int = max((len(e) for e in self._endings), default=0)
        self._eos_ids = eos_ids
        self._done: list[bool] = []  # lazy-init on first call (batch size unknown at construction)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        if not self._done:
            self._done = [False] * batch_size
        if not self._endings:
            return scores  # nothing to enforce
        for i in range(batch_size):
            if self._done[i]:
                continue
            seq = input_ids[i].tolist()
            tail = seq[-self._max_end_len:]
            matched = False
            for ending in self._endings:
                elen = len(ending)
                if len(tail) >= elen and tail[-elen:] == ending:
                    matched = True
                    break
            if matched:
                self._done[i] = True
            else:
                for eos_id in self._eos_ids:
                    scores[i, eos_id] = float("-inf")
        return scores


# Backward-compatibility alias — older callers may still import the old name.
ForceAnswerProcessor = MultiEndingProcessor


def _tool_end_markers(family: ModelFamily) -> list[str]:
    """Return textual endings that close a tool-call generation for a family."""
    if family == "llama":
        return ["<|eom_id|>", "<|eot_id|>"]
    return ["</tool_call>"]


def _force_answer_stop_strings(family: ModelFamily, final_only: bool) -> list[str]:
    """Stop strings that are also valid EOS-gating endings for one turn."""
    if final_only:
        return ["</answer>"]
    return _tool_end_markers(family) + ["</answer>"]


def _encode_end_markers(tokenizer, markers: list[str]) -> list[list[int]]:
    """Encode ending markers, dropping any marker that is not in the tokenizer."""
    endings: list[list[int]] = []
    for marker in markers:
        ids = tokenizer.encode(marker, add_special_tokens=False)
        if ids:
            endings.append(ids)
    return endings


# ---------------------------------------------------------------------------
# Batched generation helper
# ---------------------------------------------------------------------------

def _run_batched_generation(
    model, tokenizer, states: list[dict],
    no_tool: bool, max_tool_iterations: int,
    max_new_tokens: int, temperature: float,
    family: ModelFamily,
    min_new_tokens: int = 0,
    force_answer: bool = True,
) -> None:
    """Turn-level batched generation across a mini-batch of sample states.

    At each tool-calling iteration all unfinished samples are grouped into a
    single model.generate() call (left-padded). Tool responses are injected
    per-sample in sequence. Modifies states in-place.

    Required state keys (initialised by caller):
        messages, tool_calls, tool_responses, query_texts,
        n_tool_calls, final_content, done

    EOS-suppression endings (via MultiEndingProcessor) are picked per-iteration:
      - no_tool, or last iteration in tool mode → only </answer> allowed to end
      - intermediate tool iteration            → family tool-end OR </answer>
    `min_new_tokens` is defence-in-depth only and defaults to 0.
    """
    from scripts.utils.model_adapter import strip_generation_artifacts

    tool_re = get_tool_call_regex(family)
    eos = get_eos_for_generation(family, tokenizer)
    eos_ids_list = [eos] if isinstance(eos, int) else list(eos)
    answer_endings = _encode_end_markers(tokenizer, ["</answer>"])
    tool_endings = _encode_end_markers(tokenizer, _tool_end_markers(family))
    orig_padding_side = tokenizer.padding_side

    for iteration in range(max_tool_iterations + 1):
        active_idx = [i for i, s in enumerate(states) if not s["done"]]
        if not active_idx:
            break

        # Tokenize every active history, left-pad so all prompts end at the
        # same position — required for correct batched decoder-only generation.
        texts = [
            tokenizer.apply_chat_template(
                states[i]["messages"],
                tools=get_tools_for_template(family),
                add_generation_prompt=True,
                tokenize=False,
            )
            for i in active_idx
        ]
        tokenizer.padding_side = "left"
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        tokenizer.padding_side = orig_padding_side

        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)
        prompt_len = input_ids.shape[1]  # uniform after left-padding

        # Pick valid endings for this iteration:
        #   - no_tool                   → only </answer>
        #   - last iteration in tool    → only </answer> (tool quota exhausted)
        #   - intermediate tool iter    → family tool-end OR </answer>
        if force_answer:
            if no_tool or iteration == max_tool_iterations:
                endings = answer_endings
            else:
                endings = tool_endings + answer_endings
            lp = LogitsProcessorList(
                [MultiEndingProcessor(endings, eos_ids_list)]
            )
        else:
            lp = None

        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos,
                use_cache=True,
                logits_processor=lp,
            )

        for batch_pos, sample_idx in enumerate(active_idx):
            state = states[sample_idx]
            raw = tokenizer.decode(out[batch_pos][prompt_len:], skip_special_tokens=False)
            generated = strip_generation_artifacts(raw, family)

            tool_calls = extract_tool_calls(raw, tool_re) if not no_tool else []

            if tool_calls and iteration < max_tool_iterations:
                state["messages"].append({"role": "assistant", "content": generated})
                state["n_tool_calls"] += len(tool_calls)
                state["tool_calls"].extend(tool_calls)
                for tc in tool_calls:
                    try:
                        args = tc.get("arguments") or tc.get("parameters", {})
                        if isinstance(args, str):
                            args = json.loads(args)
                        query = args.get("query", "")
                        if query:
                            state["query_texts"].append(query)
                        result = search_medical_knowledge(query)
                        state["tool_responses"].append(result)
                        state["messages"].append(get_tool_response_message(result, family))
                    except Exception as e:
                        state["tool_responses"].append(f"ERROR: {e}")
                        state["messages"].append(get_tool_response_message(f"ERROR: {e}", family))
            else:
                state["messages"].append({"role": "assistant", "content": generated})
                state["final_content"] = generated
                state["done"] = True


# ---------------------------------------------------------------------------
# vLLM generation path (separate venv: vllm_venv312)
# ---------------------------------------------------------------------------

try:
    from vllm.v1.sample.logits_processor import AdapterLogitsProcessor as _VllmAdapterLogitsProcessor
except Exception:
    _VllmAdapterLogitsProcessor = object


class VllmForceAnswerLP(_VllmAdapterLogitsProcessor):
    """Suppress EOS tokens until a valid family-specific ending appears.

    This class must be module-level. vLLM may force multiprocessing `spawn`,
    and nested classes cannot be pickled into worker processes.
    """

    # Marker strings whose presence in SamplingParams.stop activates the LP.
    # Mapped to their token-id sequences in __init__.
    _ENDING_MARKERS: tuple[str, ...] = (
        "</answer>",
        "</tool_call>",
        "<|eom_id|>",
        "<|eot_id|>",
    )

    def __init__(self, vllm_config, device, is_pin_memory):
        if _VllmAdapterLogitsProcessor is object:
            raise RuntimeError("vLLM is required to use VllmForceAnswerLP.")
        super().__init__(vllm_config, device, is_pin_memory)
        from transformers import AutoTokenizer
        model_path = vllm_config.model_config.model
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # marker -> token-id sequence, picklable for multi-proc executor.
        self._end_ids_by_marker: dict[str, list[int]] = {
            m: list(tok.encode(m, add_special_tokens=False))
            for m in self._ENDING_MARKERS
        }
        # Aggregate EOS-like ids: tokenizer + hf_config.
        eos_set: set[int] = set()
        if tok.eos_token_id is not None:
            eos_set.add(int(tok.eos_token_id))
        hf_eos = getattr(vllm_config.model_config.hf_config, "eos_token_id", None)
        if isinstance(hf_eos, int):
            eos_set.add(hf_eos)
        elif isinstance(hf_eos, (list, tuple)):
            eos_set.update(int(x) for x in hf_eos)
        self._eos_ids: list[int] = sorted(eos_set)

    def is_argmax_invariant(self) -> bool:
        return False  # may flip argmax via EOS suppression

    def new_req_logits_processor(self, params):
        stop = params.stop or ()
        # Endings active for this request = intersection of markers and stop.
        endings: list[list[int]] = [
            self._end_ids_by_marker[m]
            for m in self._ENDING_MARKERS
            if m in stop
        ]
        if not endings:
            return None  # caller did not ask for force-answer
        endings.sort(key=len, reverse=True)
        max_end_len = len(endings[0])

        # Suppress every EOS variant + per-request stop token ids.
        stop_tids = list(set(self._eos_ids) | set(params.all_stop_token_ids or []))

        def lp(output_ids, logits):
            if len(output_ids) >= 1:  # cheap early-out
                tail = (
                    list(output_ids[-max_end_len:])
                    if len(output_ids) >= max_end_len
                    else list(output_ids)
                )
                for ending in endings:
                    elen = len(ending)
                    if len(tail) >= elen and tail[-elen:] == ending:
                        return logits  # valid ending at tail -> allow EOS
            for tid in stop_tids:
                logits[tid] = float("-inf")
            return logits

        return lp


def get_vllm_force_answer_lp_class(tokenizer=None):
    """Return an AdapterLogitsProcessor class that suppresses EOS tokens until
    one of the per-request "valid endings" appears at the tail of generation.
    vLLM equivalent of MultiEndingProcessor.

    Valid endings are signalled per-request via the `stop` field:
      - stop=['</answer>']                               → only </answer> allowed
      - stop=['</tool_call>', '</answer>']               → Qwen tool or answer
      - stop=['<|eom_id|>', '<|eot_id|>', '</answer>']   → Llama tool or answer
    A request without either string in `stop` opts out (LP returns None).

    The class is self-contained: it reloads the tokenizer in __init__ from
    vllm_config.model_config.model so no closure variables need to survive
    multi-process executor pickling. The `tokenizer` arg is unused (kept for
    API compatibility).
    """
    if __name__ == "__main__":
        module = importlib.import_module("scripts.benchmark.grpo_eval.grpo_eval")
        return module.VllmForceAnswerLP
    return VllmForceAnswerLP


def _run_vllm_generation(
    llm,                                # vllm.LLM instance
    tokenizer, states: list[dict],
    no_tool: bool, max_tool_iterations: int,
    max_new_tokens: int, temperature: float,
    family: ModelFamily,
    force_answer: bool = True,
    min_tokens: int = 0,
) -> None:
    """vLLM equivalent of _run_batched_generation.

    vLLM handles batching/scheduling internally — we just pass a list of prompts.
    For tool-calling, iterations are still done by us (one llm.generate per turn).

    Per-iteration stop strings (also used by VllmForceAnswerLP to decide which
    endings allow EOS):
      - no_tool, or last iteration in tool mode → stop=['</answer>']
      - intermediate tool iteration             → family tool-end OR </answer>
    `min_tokens` is defence-in-depth and only applied on force-answer iterations
    (never on intermediate tool iterations, where it would cause the model to
    over-generate past a short </tool_call>).

    Modifies states in-place; expects same state schema as _run_batched_generation.
    """
    from vllm import SamplingParams
    from scripts.utils.model_adapter import strip_generation_artifacts

    tool_re = get_tool_call_regex(family)

    for iteration in range(max_tool_iterations + 1):
        active_idx = [i for i, s in enumerate(states) if not s["done"]]
        if not active_idx:
            break

        prompts = [
            tokenizer.apply_chat_template(
                states[i]["messages"],
                tools=get_tools_for_template(family),
                add_generation_prompt=True,
                tokenize=False,
            )
            for i in active_idx
        ]

        # Pick stop strings per iteration. Same gating rule as the LP.
        if force_answer:
            if no_tool or iteration == max_tool_iterations:
                stop_strs = ["</answer>"]
                is_force_answer_iter = True
            else:
                stop_strs = _force_answer_stop_strings(family, final_only=False)
                is_force_answer_iter = False
        else:
            stop_strs = None
            is_force_answer_iter = False

        # min_tokens guards premature EOS only when a final answer is expected.
        # On tool-calling iterations the model legitimately emits short outputs
        # ending in </tool_call>, so min_tokens stays 0 there.
        cur_min = min_tokens if is_force_answer_iter else 0

        sp = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            min_tokens=cur_min,
            stop=stop_strs,
            include_stop_str_in_output=True,
            skip_special_tokens=False,
            n=1,
        )

        outputs = llm.generate(prompts, sp, use_tqdm=False)

        for batch_pos, sample_idx in enumerate(active_idx):
            state = states[sample_idx]
            raw_generated = outputs[batch_pos].outputs[0].text
            generated = strip_generation_artifacts(raw_generated, family)

            tool_calls = extract_tool_calls(raw_generated, tool_re) if not no_tool else []

            if tool_calls and iteration < max_tool_iterations:
                state["messages"].append({"role": "assistant", "content": generated})
                state["n_tool_calls"] += len(tool_calls)
                state["tool_calls"].extend(tool_calls)
                for tc in tool_calls:
                    try:
                        args = tc.get("arguments") or tc.get("parameters", {})
                        if isinstance(args, str):
                            args = json.loads(args)
                        query = args.get("query", "")
                        if query:
                            state["query_texts"].append(query)
                        result = search_medical_knowledge(query)
                        state["tool_responses"].append(result)
                        state["messages"].append(get_tool_response_message(result, family))
                    except Exception as e:
                        state["tool_responses"].append(f"ERROR: {e}")
                        state["messages"].append(get_tool_response_message(f"ERROR: {e}", family))
            else:
                state["messages"].append({"role": "assistant", "content": generated})
                state["final_content"] = generated
                state["done"] = True


# ---------------------------------------------------------------------------
# Retrieval quality
# ---------------------------------------------------------------------------

def score_retrieval_relevance(
    encoder, question: str, answer_text: str, retrieved_texts: list[str],
) -> float:
    """Max cosine similarity between retrieved facts and (question + answer) anchor."""
    if not retrieved_texts:
        return 0.0
    anchor = f"{question} The answer is {answer_text}"
    embeddings = encoder.encode(
        [anchor] + retrieved_texts, normalize_embeddings=True,
    )
    sims = embeddings[1:] @ embeddings[0]
    return float(sims.max())


# ---------------------------------------------------------------------------
# Dataset normalization — canonical schema: {question, options, answer_idx, answer}
# ---------------------------------------------------------------------------

# PubMedQA (raw HF): final_decision → MCQ letter mapping
_PUBMEDQA_DECISION_TO_IDX = {"yes": "A", "no": "B", "maybe": "C"}
_PUBMEDQA_OPTIONS = {"A": "yes", "B": "no", "C": "maybe"}


def normalize_row(row: dict) -> dict:
    """Normalize any supported dataset row to the canonical eval schema.

    Canonical output keys:
        question   str   question text (includes context prefix for raw PubMedQA)
        options    dict  {letter: option_text, ...}
        answer_idx str   correct letter ("A"–"J")
        answer     str   full answer text

    Handled schema variants:
      MedQA / MedMCQA_4options_fixed
        options: dict, answer_idx: str           → pass-through

      BioMed-R1-Eval (all datasets)
        options: JSON str, answer_idx: str       → json.loads(options)

      MedXpertQA_Text
        options: dict, label: str (no answer_idx) → label → answer_idx

      PubMedQA (raw HF, split=train)
        context: {contexts: [str]}, final_decision: str, no options
        → context prepended to question, options built as A/B/C yes/no/maybe
    """
    row = dict(row)

    # ── options: JSON string → dict ──
    if isinstance(row.get("options"), str):
        try:
            row["options"] = json.loads(row["options"])
        except (json.JSONDecodeError, TypeError):
            row["options"] = {}

    # ── answer_idx: missing → use label field (MedXpertQA_Text) ──
    if not row.get("answer_idx") and row.get("label"):
        row["answer_idx"] = str(row["label"]).strip().upper()

    # ── Raw PubMedQA: build MCQ from context + final_decision ──
    if "final_decision" in row and not row.get("answer_idx"):
        decision = str(row["final_decision"]).strip().lower()
        row["answer_idx"] = _PUBMEDQA_DECISION_TO_IDX.get(decision, "A")
        if not row.get("options"):
            row["options"] = _PUBMEDQA_OPTIONS.copy()
        # Prepend abstract context to question so model can reason over it
        context_obj = row.get("context", {})
        if isinstance(context_obj, dict):
            paragraphs = context_obj.get("contexts", [])
            if paragraphs:
                context_text = "\n\n".join(p.strip() for p in paragraphs if p and p.strip())
                row["question"] = f"Context:\n{context_text}\n\nQuestion: {row['question']}"

    # ── answer: derive from options if missing ──
    if not row.get("answer") and row.get("answer_idx") and isinstance(row.get("options"), dict):
        row["answer"] = row["options"].get(row["answer_idx"], row["answer_idx"])

    return row


# ---------------------------------------------------------------------------
# Dataset loading helper
# ---------------------------------------------------------------------------

def load_benchmark(path: str, n_samples: int, seed: int):
    """Load benchmark dataset from path (handles both Dataset and DatasetDict)."""
    ds = load_from_disk(path)
    # DatasetDict: pick the split named in path or default to 'test'
    if hasattr(ds, "column_names") and isinstance(ds.column_names, dict):
        # It's a DatasetDict
        split_name = Path(path).name  # e.g. "test"
        if split_name in ds:
            ds = ds[split_name]
        elif "test" in ds:
            ds = ds["test"]
        else:
            ds = ds[list(ds.keys())[0]]
    ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
    return ds


# ---------------------------------------------------------------------------
# Per-benchmark evaluation
# ---------------------------------------------------------------------------

def eval_benchmark(
    model, tokenizer, ds, benchmark_name: str,
    no_tool: bool, max_tool_iterations: int,
    temperature: float, score_retrieval: bool,
    encoder=None,
    self_consistency: bool = False,
    sc_samples: int = 5,
    family: ModelFamily = "qwen",
    eval_batch_size: int = 32,
    min_new_tokens: int = 50,
    use_vllm: bool = False,
    vllm_force_answer: bool = True,
    vllm_min_tokens: int = 0,
    vllm_chunk_size: int = 256,
    save_completions: bool = False,
) -> dict:
    """Run inference on one benchmark split and return metrics + per_sample list.

    Non-SC path: turn-level batching via _run_batched_generation (transformers)
      or _run_vllm_generation (vLLM, when use_vllm=True).
      For transformers: all samples in a mini-batch are processed in a single
      model.generate() call per tool-calling iteration. Length-sorted to reduce
      padding-induced FP divergence.
      For vLLM: batching/scheduling is handled internally — we pass the whole
      eval set as one list per iteration (eval_batch_size is ignored).
    SC path: unchanged — transformers only via generate_with_selfconsistency.
      use_vllm=True with self_consistency=True is unsupported (raise upstream).
    """
    system_prompt = NO_TOOL_SYSTEM_PROMPT if no_tool else SYSTEM_PROMPT
    results = []
    n = len(ds)
    all_exs = [normalize_row(dict(ex)) for ex in ds]

    mode_str = "no-tool" if no_tool else "with-tool"
    sc_str = f"  sc={sc_samples}" if self_consistency else ""
    if self_consistency:
        batch_str = ""
    elif use_vllm:
        batch_str = "  engine=vllm"
    else:
        batch_str = f"  batch={eval_batch_size}"
    print(f"\n  Benchmark: {benchmark_name}  ({n} samples, {mode_str}{sc_str}{batch_str})")

    # ── Batched pre-generation (non-SC only) ──
    gen_states: list[dict] | None = None
    if not self_consistency:
        gen_states = [
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": (
                        f"{ex['question']}\n\nOptions:\n"
                        + "\n".join(f"{k}. {v}" for k, v in ex["options"].items())
                    )},
                ],
                "tool_calls": [],
                "tool_responses": [],
                "query_texts": [],
                "n_tool_calls": 0,
                "final_content": "",
                "done": False,
            }
            for ex in all_exs
        ]

        if use_vllm:
            # vLLM handles batching/scheduling internally; for visibility we
            # split the eval set into chunks and print progress between them.
            chunk_size = vllm_chunk_size if vllm_chunk_size > 0 else n
            t_start = time.time()
            n_correct_running = 0
            print(f"    starting vLLM eval — chunk_size={chunk_size}", flush=True)
            for chunk_start in range(0, n, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n)
                _run_vllm_generation(
                    model, tokenizer, gen_states[chunk_start:chunk_end],
                    no_tool=no_tool,
                    max_tool_iterations=max_tool_iterations,
                    max_new_tokens=1024,
                    temperature=temperature,
                    family=family,
                    force_answer=vllm_force_answer,
                    min_tokens=vllm_min_tokens,
                )
                # Running accuracy: gen_states is in original order in vLLM path.
                for s, ex in zip(
                    gen_states[chunk_start:chunk_end],
                    all_exs[chunk_start:chunk_end],
                ):
                    if extract_answer_letter(s["final_content"]) == ex["answer_idx"]:
                        n_correct_running += 1
                _print_progress(chunk_end, n, t_start, n_correct_running)
        else:
            # Sort by prompt token count so samples within a batch have similar
            # lengths → minimal left-padding → reduced FP divergence across batch sizes.
            prompt_texts = [
                tokenizer.apply_chat_template(
                    s["messages"],
                    tools=get_tools_for_template(family),
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for s in gen_states
            ]
            prompt_lens = [
                len(tokenizer.encode(t, add_special_tokens=False)) for t in prompt_texts
            ]
            orig_order = sorted(range(n), key=lambda i: prompt_lens[i])
            gen_states = [gen_states[i] for i in orig_order]

            t_start = time.time()
            n_correct_running = 0
            for batch_start in range(0, n, eval_batch_size):
                batch_end = min(batch_start + eval_batch_size, n)
                _run_batched_generation(
                    model, tokenizer,
                    gen_states[batch_start:batch_end],
                    no_tool=no_tool,
                    max_tool_iterations=max_tool_iterations,
                    max_new_tokens=1024,
                    temperature=temperature,
                    family=family,
                    min_new_tokens=min_new_tokens,
                )
                # Running accuracy: gen_states is in length-sorted order, so
                # map back to all_exs via orig_order.
                for sorted_idx in range(batch_start, batch_end):
                    s = gen_states[sorted_idx]
                    ex = all_exs[orig_order[sorted_idx]]
                    if extract_answer_letter(s["final_content"]) == ex["answer_idx"]:
                        n_correct_running += 1
                _print_progress(batch_end, n, t_start, n_correct_running)

            # Restore original order so per-sample metrics align with all_exs.
            inv_order = [0] * n
            for sorted_pos, orig_pos in enumerate(orig_order):
                inv_order[orig_pos] = sorted_pos
            gen_states = [gen_states[inv_order[i]] for i in range(n)]

    # ── Per-sample metrics ──
    # SC path: per-sample generation here, so track running progress.
    t_start_sc = time.time() if self_consistency else None
    n_correct_sc = 0
    for i, ex in enumerate(all_exs):
        # ── Generation ──
        sc_meta: dict | None = None
        if self_consistency:
            sc_meta = generate_with_selfconsistency(
                model, tokenizer, ex["question"], ex["options"],
                no_tool=no_tool, system_prompt=system_prompt,
                max_tool_iterations=max_tool_iterations,
                max_new_tokens=1024, temperature=temperature,
                sc_samples=sc_samples, family=family,
            )
            res = sc_meta["rep_res"]
            final_content = sc_meta["rep_final_content"]
            pred_letter = sc_meta["pred"]
        else:
            state = gen_states[i]
            final_content = state["final_content"]
            res = {
                "messages": state["messages"],
                "tool_calls": state["tool_calls"],
                "tool_responses": state["tool_responses"],
                "query_texts": state["query_texts"],
                "n_tool_calls": state["n_tool_calls"],
            }
            pred_letter = extract_answer_letter(final_content)

        is_correct = pred_letter == ex["answer_idx"]

        # SC progress (every 20 samples + at the very end).
        if self_consistency:
            if is_correct:
                n_correct_sc += 1
            if (i + 1) % 20 == 0 or (i + 1) == n:
                _print_progress(i + 1, n, t_start_sc, n_correct_sc)

        # Think block analysis (on representative/single sample)
        think_matches = _THINK_RE.findall(final_content)
        think_text = " ".join(think_matches)
        think_words = len(think_text.split()) if think_text else 0
        has_think = bool(think_matches)
        has_answer = bool(_ANSWER_RE.search(final_content))

        # Query quality: copy-paste ratio per query
        copy_paste_ratios = [
            _copy_paste_ratio(q, ex["question"]) for q in res["query_texts"]
        ]
        avg_copy_paste = float(np.mean(copy_paste_ratios)) if copy_paste_ratios else None

        # Retrieval score
        retrieval_score = None
        if score_retrieval and encoder and res["tool_responses"]:
            retrieval_score = score_retrieval_relevance(
                encoder, ex["question"], ex["answer"], res["tool_responses"],
            )

        entry: dict = {
            "idx": i,
            "pred": pred_letter,
            "correct_idx": ex["answer_idx"],
            "is_correct": is_correct,
            "n_tool_calls": res["n_tool_calls"],
            "has_think": has_think,
            "has_answer": has_answer,
            "think_words": think_words,
            "query_texts": res["query_texts"],
            "avg_query_copy_paste": avg_copy_paste,
            "retrieval_score": retrieval_score,
        }

        if save_completions:
            entry["question"] = ex["question"]
            entry["options"] = ex["options"]
            entry["answer_text"] = ex["answer"]
            entry["final_content"] = final_content
            entry["tool_responses"] = res.get("tool_responses", [])
            entry["messages"] = res.get("messages", [])

        if sc_meta is not None:
            greedy_pred = sc_meta["greedy_pred"]
            entry["sc_vote_counts"] = sc_meta["vote_counts"]
            entry["sc_vote_confidence"] = sc_meta["vote_confidence"]
            entry["sc_n_valid_votes"] = sc_meta["n_valid_votes"]
            entry["sc_all_preds"] = sc_meta["all_preds"]
            entry["sc_unanimous"] = sc_meta["vote_confidence"] == 1.0 and sc_meta["n_valid_votes"] > 0
            entry["sc_greedy_pred"] = greedy_pred
            entry["sc_greedy_correct"] = greedy_pred == ex["answer_idx"]
            # pass@k: correct if any of the k samples is right
            entry["sc_pass_at_k"] = any(p == ex["answer_idx"] for p in sc_meta["all_preds"] if p is not None)

        results.append(entry)

    # ── Aggregate ──
    n_correct = sum(1 for r in results if r["is_correct"])
    n_with_tool = sum(1 for r in results if r["n_tool_calls"] > 0)
    n_without_tool = n - n_with_tool
    n_has_think = sum(1 for r in results if r["has_think"])
    n_has_answer = sum(1 for r in results if r["has_answer"])
    n_extracted = sum(1 for r in results if r["pred"] is not None)

    correct_with = sum(1 for r in results if r["is_correct"] and r["n_tool_calls"] > 0)
    correct_without = sum(1 for r in results if r["is_correct"] and r["n_tool_calls"] == 0)

    acc_overall = n_correct / n
    acc_with = correct_with / n_with_tool if n_with_tool > 0 else None
    acc_without = correct_without / n_without_tool if n_without_tool > 0 else None

    think_words_list = [r["think_words"] for r in results if r["has_think"]]
    copy_paste_list = [r["avg_query_copy_paste"] for r in results if r["avg_query_copy_paste"] is not None]
    retrieval_scores = [r["retrieval_score"] for r in results if r["retrieval_score"] is not None]

    metrics = {
        "n_samples": n,
        "accuracy_overall": round(acc_overall, 4),
        "accuracy_with_tool": round(acc_with, 4) if acc_with is not None else None,
        "accuracy_without_tool": round(acc_without, 4) if acc_without is not None else None,
        "accuracy_diff_pts": round((acc_with - acc_without) * 100, 1) if (acc_with is not None and acc_without is not None) else None,
        "tool_call_frequency": round(n_with_tool / n, 4),
        "avg_tool_calls": round(sum(r["n_tool_calls"] for r in results) / n, 3),
        "multi_turn_rate": round(sum(1 for r in results if r["n_tool_calls"] >= 2) / n, 4),
        "has_think_rate": round(n_has_think / n, 4),
        "has_answer_rate": round(n_has_answer / n, 4),
        "pred_extracted_rate": round(n_extracted / n, 4),
        # Think depth
        "avg_think_words": round(float(np.mean(think_words_list)), 1) if think_words_list else 0,
        "median_think_words": int(np.median(think_words_list)) if think_words_list else 0,
        # Query quality
        "avg_query_copy_paste": round(float(np.mean(copy_paste_list)), 3) if copy_paste_list else None,
        "high_copy_paste_rate": round(
            sum(1 for x in copy_paste_list if x > 0.85) / len(copy_paste_list), 3
        ) if copy_paste_list else None,
    }

    if retrieval_scores:
        correct_ret = [r["retrieval_score"] for r in results if r["is_correct"] and r["retrieval_score"] is not None]
        wrong_ret = [r["retrieval_score"] for r in results if not r["is_correct"] and r["retrieval_score"] is not None]
        metrics["retrieval_cosine_mean"] = round(float(np.mean(retrieval_scores)), 3)
        metrics["retrieval_cosine_p50"] = round(float(np.median(retrieval_scores)), 3)
        metrics["retrieval_when_correct"] = round(float(np.mean(correct_ret)), 3) if correct_ret else None
        metrics["retrieval_when_wrong"] = round(float(np.mean(wrong_ret)), 3) if wrong_ret else None

    # ── Self-consistency aggregate metrics ──
    sc_entries = [r for r in results if "sc_vote_confidence" in r]
    if sc_entries:
        sc_n = len(sc_entries)
        confidences = [r["sc_vote_confidence"] for r in sc_entries]
        greedy_correct = sum(1 for r in sc_entries if r["sc_greedy_correct"])
        pass_at_k = sum(1 for r in sc_entries if r["sc_pass_at_k"])
        unanimous = sum(1 for r in sc_entries if r["sc_unanimous"])
        metrics["sc_samples"] = sc_samples
        metrics["sc_accuracy_majority"] = round(acc_overall, 4)
        metrics["sc_accuracy_greedy"] = round(greedy_correct / sc_n, 4)
        metrics["sc_pass_at_k"] = round(pass_at_k / sc_n, 4)
        metrics["sc_vote_confidence_mean"] = round(float(np.mean(confidences)), 3)
        metrics["sc_vote_confidence_p50"] = round(float(np.median(confidences)), 3)
        metrics["sc_unanimous_rate"] = round(unanimous / sc_n, 4)
        # Lift: majority vote vs greedy
        metrics["sc_majority_vs_greedy_pts"] = round(
            (acc_overall - greedy_correct / sc_n) * 100, 1
        )

    # ── Print summary ──
    print(f"\n  ── {benchmark_name} Results ──")
    print(f"  Accuracy:   {acc_overall:.1%}  ({n_correct}/{n})")
    if sc_entries:
        sc_greedy_acc = metrics["sc_accuracy_greedy"]
        print(f"    majority@{sc_samples}:  {acc_overall:.1%}   greedy(pass@1): {sc_greedy_acc:.1%}"
              f"   lift: {metrics['sc_majority_vs_greedy_pts']:+.1f} pts")
        print(f"    pass@{sc_samples}:      {metrics['sc_pass_at_k']:.1%}"
              f"   confidence: avg={metrics['sc_vote_confidence_mean']:.2f}"
              f"   unanimous={metrics['sc_unanimous_rate']:.1%}")
    if acc_with is not None and acc_without is not None:
        print(f"    with tool:    {acc_with:.1%}  ({correct_with}/{n_with_tool})")
        print(f"    without tool: {acc_without:.1%}  ({correct_without}/{n_without_tool})")
        diff = acc_with - acc_without
        print(f"    delta: {diff*100:+.1f} pts")
    print(f"  Tool freq:  {n_with_tool/n:.1%}  avg_calls={metrics['avg_tool_calls']:.2f}")
    print(f"  Format:     <think>={n_has_think/n:.1%}  <answer>={n_has_answer/n:.1%}")
    print(f"  Think depth: avg={metrics['avg_think_words']:.0f} words  median={metrics['median_think_words']}")
    if copy_paste_list:
        print(f"  Query copy-paste: avg={metrics['avg_query_copy_paste']:.2f}  high_rate={metrics['high_copy_paste_rate']:.2f}")
    if retrieval_scores:
        print(f"  Retrieval:  mean={metrics['retrieval_cosine_mean']:.3f}  correct={metrics['retrieval_when_correct']:.3f}  wrong={metrics['retrieval_when_wrong']:.3f}")

    return {"metrics": metrics, "per_sample": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-GRPO evaluation script")
    p.add_argument("--model-path", required=True, help="Path to merged model directory.")
    p.add_argument(
        "--benchmarks", nargs="+",
        default=["dataset/MedQA/test"],
        help="One or more load_from_disk paths. Split can be a sub-path (e.g. dataset/MedMCQA/test).",
    )
    p.add_argument("--data-dir", default="data/", help="KG data directory for retrieval tool.")
    p.add_argument("--no-tool", action="store_true", help="Forced no-tool ablation.")
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--max-tool-iterations", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.3,
                   help="Lower than GRPO training temp (0.8) for deterministic eval.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--score-retrieval", action="store_true",
                   help="Score retrieval with MedEmbed (requires --no-tool=False).")
    # Self-consistency
    p.add_argument("--self-consistency", action="store_true",
                   help=(
                       "Enable self-consistency (majority vote) decoding. "
                       "Runs --sc-samples independent rollouts per question and "
                       "picks the most frequent answer. Use --temperature >= 0.6 "
                       "to get meaningful diversity (matches GRPO training temp)."
                   ))
    p.add_argument("--sc-samples", type=int, default=5,
                   help="Number of independent samples for majority voting (default 5).")
    p.add_argument(
        "--model-family",
        default="auto",
        choices=["auto", "qwen", "llama"],
        help="Model family for token format. 'auto' detects from model path.",
    )
    p.add_argument("--output", default=None, help="Path to save JSON report.")
    p.add_argument(
        "--eval-batch-size", type=int, default=32,
        help=(
            "Samples processed together per model.generate() call (non-SC path). "
            "Higher values improve GPU utilisation. Lower if OOM. Default: 32."
        ),
    )
    p.add_argument(
        "--min-new-tokens", type=int, default=0,
        help=(
            "Minimum tokens to generate before EOS is allowed (iteration 0 only). "
            "ForceAnswerProcessor already blocks EOS until </answer> is generated, "
            "so this is defence-in-depth only. Default: 0 (disabled)."
        ),
    )
    p.add_argument(
        "--use-vllm", action="store_true",
        help=(
            "Run inference via vLLM instead of HF transformers. Requires the "
            "separate vllm_venv312 venv (vLLM 0.20+). Significantly faster than "
            "transformers for large batches. Not yet supported with "
            "--self-consistency. Default: False (use transformers)."
        ),
    )
    p.add_argument(
        "--vllm-gpu-mem", type=float, default=0.6,
        help="vLLM gpu_memory_utilization (0–1). Default: 0.6",
    )
    p.add_argument(
        "--vllm-max-model-len", type=int, default=4096,
        help="vLLM max_model_len (prompt + max_tokens). Default: 4096.",
    )
    p.add_argument(
        "--vllm-no-force-answer", action="store_true",
        help=(
            "Disable VllmForceAnswerLP (suppress-EOS-until-</answer> processor). "
            "By default, force-answer is ON when --use-vllm is set."
        ),
    )
    p.add_argument(
        "--vllm-min-tokens", type=int, default=200,
        help=(
            "vLLM SamplingParams.min_tokens. Empirically the LP alone gives "
            "weaker forcing in vLLM than in transformers (~60%% vs ~88%% "
            "AnswerRate). Combining LP with min_tokens=200 reaches ~82%% — "
            "the recommended default. Set to 0 to disable. Default: 200."
        ),
    )
    p.add_argument(
        "--vllm-chunk-size", type=int, default=256,
        help=(
            "vLLM eval is split into chunks of this size, with a progress line "
            "(elapsed / ETA / running accuracy) printed between chunks. Set to "
            "0 to run the whole benchmark in a single shot (no progress lines). "
            "Default: 256."
        ),
    )
    p.add_argument(
        "--save-completions", action="store_true",
        help=(
            "Dump full conversation messages, final_content, and tool_responses "
            "into per_sample entries. Greatly increases output JSON size (~5-10x). "
            "Use for qualitative analysis / picking illustrative examples."
        ),
    )
    p.add_argument(
        "--vllm-enforce-eager", action="store_true",
        help=(
            "Pass enforce_eager=True to vLLM LLM(), disabling CUDAGraphs and "
            "torch.compile. Use only as a fallback if CUDAGraphs fail at runtime "
            "(e.g. OOM during graph capture). By default CUDAGraphs are enabled, "
            "giving ~1.5-3x decode speedup via FlashInfer on GB10 sm_12.1."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.use_vllm and args.self_consistency:
        raise SystemExit("ERROR: --use-vllm + --self-consistency is not supported yet.")

    mode = "no_tool" if args.no_tool else "with_tool"
    sc_label = f"  sc={args.sc_samples}" if args.self_consistency else ""
    engine_label = "  engine=vllm" if args.use_vllm else "  engine=transformers"
    print(f"\n{'='*65}")
    print(f"Post-GRPO Evaluation  [{mode}{sc_label}{engine_label}]")
    print(f"  model:      {args.model_path}")
    print(f"  benchmarks: {args.benchmarks}")
    print(f"  n_samples:  {args.n_samples}  temp={args.temperature}  seed={args.seed}")
    if args.self_consistency:
        print(f"  self-consistency: {args.sc_samples} samples/question")
    print(f"{'='*65}")

    # Load retrieval tool (always load for with-tool mode; skip heavy load for no-tool)
    encoder = None
    if not args.no_tool:
        print("\nLoading KG retrieval tool ...")
        kg = MedicalKnowledgeTool.load(data_dir=args.data_dir)
        if args.score_retrieval:
            encoder = kg.encoder

    # --- Detect model family ---
    family: ModelFamily = (
        detect_family(args.model_path) if args.model_family == "auto" else args.model_family
    )
    print(f"Model family: {family}")

    # Load model  (must be a merged/dense model — merge LoRA adapters first with
    # scripts/finetune/merge_peft_adapter.py if starting from a GRPO checkpoint)
    print(f"Loading model from {args.model_path} ...")
    adapter_cfg = Path(args.model_path) / "adapter_config.json"
    if adapter_cfg.exists():
        raise SystemExit(
            f"ERROR: {args.model_path} appears to be a LoRA adapter "
            "(adapter_config.json found).\n"
            "Merge it first:\n"
            "  python scripts/finetune/merge_peft_adapter.py \\\n"
            f"    --adapter-path {args.model_path} \\\n"
            "    --output-dir outputs/grpo_v4_merged"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, **get_tokenizer_load_kwargs(family)
    )
    if args.use_vllm:
        # vLLM path — runs on the separate vllm_venv312 venv.
        from vllm import LLM
        from scripts.utils.model_adapter import normalize_tokenizer_only
        # Tokenizer-side EOS/pad fixes must happen BEFORE building the LP class
        # (which captures eos_token_id at construction time).
        normalize_tokenizer_only(tokenizer, family, padding_side="left")
        vllm_lp_classes = []
        if not args.vllm_no_force_answer:
            vllm_lp_classes.append(get_vllm_force_answer_lp_class(tokenizer))
        model = LLM(
            model=args.model_path,
            dtype="bfloat16",
            gpu_memory_utilization=args.vllm_gpu_mem,
            max_model_len=args.vllm_max_model_len,
            enforce_eager=args.vllm_enforce_eager,    # default False; use --vllm-enforce-eager if CUDAGraphs OOM
            trust_remote_code=family == "qwen",
            logits_processors=vllm_lp_classes if vllm_lp_classes else None,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            device_map="auto", **get_model_load_kwargs(family),
        ).eval()
        normalize_special_tokens(tokenizer, model, family, padding_side="left")

    # Evaluate each benchmark
    benchmark_results = {}
    for bench_path in args.benchmarks:
        bench_name = Path(bench_path).name  # "test" or dataset name
        # Use parent dir name as readable name if split is "test"/"validation"
        if bench_name in ("test", "train", "validation"):
            bench_name = f"{Path(bench_path).parent.name}/{bench_name}"

        print(f"\nLoading {bench_path} ...")
        ds = load_benchmark(bench_path, args.n_samples, args.seed)

        result = eval_benchmark(
            model, tokenizer, ds, bench_name,
            no_tool=args.no_tool,
            max_tool_iterations=args.max_tool_iterations,
            temperature=args.temperature,
            score_retrieval=args.score_retrieval,
            encoder=encoder,
            self_consistency=args.self_consistency,
            sc_samples=args.sc_samples,
            family=family,
            eval_batch_size=args.eval_batch_size,
            min_new_tokens=args.min_new_tokens,
            use_vllm=args.use_vllm,
            vllm_force_answer=not args.vllm_no_force_answer,
            vllm_min_tokens=args.vllm_min_tokens,
            vllm_chunk_size=args.vllm_chunk_size,
            save_completions=args.save_completions,
        )
        benchmark_results[bench_name] = result

    # ── Final summary ──
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    if args.self_consistency:
        print(f"{'Benchmark':<30} {'Maj@k':>6} {'Pass@1':>7} {'Pass@k':>7} {'Conf':>6} {'Unani':>6}")
        print("-" * 65)
        for bench, data in benchmark_results.items():
            m = data["metrics"]
            maj = f"{m['accuracy_overall']:.1%}"
            g = f"{m.get('sc_accuracy_greedy', m['accuracy_overall']):.1%}"
            pk = f"{m.get('sc_pass_at_k', m['accuracy_overall']):.1%}"
            conf = f"{m.get('sc_vote_confidence_mean', 1.0):.2f}"
            unani = f"{m.get('sc_unanimous_rate', 1.0):.1%}"
            print(f"{bench:<30} {maj:>6} {g:>7} {pk:>7} {conf:>6} {unani:>6}")
    else:
        print(f"{'Benchmark':<30} {'Acc':>6} {'W/tool':>8} {'Wo/tool':>8} {'Delta':>7} {'Think':>6}")
        print("-" * 65)
        for bench, data in benchmark_results.items():
            m = data["metrics"]
            acc = f"{m['accuracy_overall']:.1%}"
            wt = f"{m['accuracy_with_tool']:.1%}" if m["accuracy_with_tool"] is not None else "  N/A "
            wot = f"{m['accuracy_without_tool']:.1%}" if m["accuracy_without_tool"] is not None else "  N/A "
            delta = f"{m['accuracy_diff_pts']:+.1f}" if m["accuracy_diff_pts"] is not None else " N/A "
            think = f"{m['avg_think_words']:.0f}w"
            print(f"{bench:<30} {acc:>6} {wt:>8} {wot:>8} {delta:>7} {think:>6}")

    # ── Save ──
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "model_path": args.model_path,
            "mode": mode,
            "temperature": args.temperature,
            "n_samples_per_benchmark": args.n_samples,
            "seed": args.seed,
            "self_consistency": args.self_consistency,
            "sc_samples": args.sc_samples if args.self_consistency else None,
            "benchmarks": benchmark_results,
        }
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
