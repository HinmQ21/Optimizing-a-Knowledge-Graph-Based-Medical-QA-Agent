#!/usr/bin/env python3
"""Unified SFT post-training evaluation pipeline for medical QA datasets.

Supported datasets (--dataset):
  medqa      – MedQA 4-option MCQ (A-D), split=test
  medmcqa    – MedMCQA 4-option MCQ (A-D), split=validation
  medxpertqa – MedXpertQA 10-option MCQ (A-J), split=test
  pubmedqa   – PubMedQA yes/no/maybe, split=train

Supported training styles (--training):
  huatuo    – ## Thinking / ## Final Response  (markdown header format)
  medreason – ## Thinking / ## Final Response  (markdown header format)
  think_tag – <think>...</think> <answer>...</answer>  (XML tag format)
  custom    – fully manual: supply --thinking-section-title / --final-section-title

Output formats:
  header – model outputs: ## Thinking ... ## Final Response ... The answer is X
  tag    – model outputs: <think>...</think><answer>The answer is X</answer>
"""
import argparse
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from datasets import load_from_disk
from tqdm.auto import tqdm
import transformers.utils.import_utils as hf_import_utils

hf_import_utils._torchvision_available = False

from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# DatasetAdapter – everything that differs between datasets
# ---------------------------------------------------------------------------

@dataclass
class DatasetAdapter:
    """Encapsulates all dataset-specific behaviour for a single benchmark.

    Each callable receives/returns only plain Python dicts and strings so the
    shared evaluation loop stays completely dataset-agnostic.
    """

    # CLI / path defaults
    default_path: str
    default_split: str
    default_num_samples: int
    default_output_dir: str

    # (raw HF row) -> normalized dict:
    #   MCQ:      {question, options, gold}        gold = "A".."D" or "A".."J"
    #   PubMedQA: {question, context_text, gold}   gold = "yes"|"no"|"maybe"
    #   + any extra keys preserved verbatim in prediction records
    normalize: Callable[[dict], dict]

    # (norm, thinking_title, final_title, format_type) -> prompt string
    build_prompt: Callable[[dict, str, str, str], str]

    # (output_text, norm, final_section_re, format_type) -> (pred|None, parse_source)
    parse: Callable[[str, dict, "re.Pattern | None", str], "tuple[str | None, str]"]

    # (norm, draft_response) -> fallback extraction prompt string
    build_fallback_prompt: Callable[[dict, str], str]


# ---------------------------------------------------------------------------
# Shared text utilities
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def _make_final_section_re(final_title: str) -> re.Pattern:
    return re.compile(
        rf"##\s*{re.escape(final_title)}\s*(.*)$",
        re.IGNORECASE | re.DOTALL,
    )


def _extract_final_section(text: str, final_section_re: re.Pattern) -> str | None:
    """Extract text after a markdown header like '## Final Response'."""
    m = final_section_re.search(text)
    return m.group(1).strip() if m else None


_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _extract_answer_tag(text: str) -> str | None:
    """Extract text between <answer> and </answer> tags."""
    m = _ANSWER_TAG_RE.search(text)
    return m.group(1).strip() if m else None


def _extract_answer_section(
    text: str, final_section_re: re.Pattern | None, format_type: str
) -> str | None:
    """Unified extraction: tag format uses <answer> tags, header format uses ## header."""
    if format_type == "tag":
        return _extract_answer_tag(text)
    if final_section_re is not None:
        return _extract_final_section(text, final_section_re)
    return None


def _clean_text_for_fallback(text: str, format_type: str) -> str:
    """Remove the thinking/reasoning section to get clean text for fallback parsing."""
    if format_type == "tag":
        # Remove entire <think>...</think> block
        cleaned = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
        # Also remove leftover tags
        cleaned = cleaned.replace("<answer>", " ").replace("</answer>", " ")
        return cleaned.strip()
    # Header format: just strip stray <think> tags (model might emit them)
    return text.replace("<think>", " ").replace("</think>", " ").strip()


def render_chat(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"### User:\n{prompt}\n\n### Assistant:\n"


# ---------------------------------------------------------------------------
# MCQ A-D adapter  (MedQA / MedMCQA)
# ---------------------------------------------------------------------------

_MCQ4_LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
_MCQ4_VALID = {"A", "B", "C", "D"}
_COP_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D", "a": "A", "b": "B", "c": "C", "d": "D"}

_MCQ4_EXPLICIT_PATTERNS = [
    re.compile(r"final\s+answer\s*[:\-]\s*([ABCD])\b", re.IGNORECASE),
    re.compile(r"the\s+answer\s+is\s+([ABCD])\b", re.IGNORECASE),
    re.compile(r"the\s+correct\s+answer\s+is\s+([ABCD])\b", re.IGNORECASE),
    re.compile(r"the\s+correct\s+option\s+is\s+([ABCD])\b", re.IGNORECASE),
    re.compile(r"option\s+([ABCD])\b", re.IGNORECASE),
    re.compile(r"answer\s*[:\-]\s*([ABCD])\b", re.IGNORECASE),
]


def _mcq4_normalize(row: dict) -> dict:
    """MedQA / MedMCQA_4options (options+answer_idx) or raw MedMCQA (opa-opd+cop)."""
    if "options" in row and "answer_idx" in row:
        return {
            "question": row["question"],
            "options": dict(row["options"]),
            "gold": str(row["answer_idx"]).upper(),
        }
    options = {"A": row["opa"], "B": row["opb"], "C": row["opc"], "D": row["opd"]}
    cop = row.get("cop")
    gold = _COP_TO_LETTER.get(cop, "") if cop is not None else ""
    return {"question": row["question"], "options": options, "gold": gold}


def _mcq4_build_prompt(
    norm: dict, thinking_title: str, final_title: str, format_type: str = "header"
) -> str:
    ordered = [f"{k}. {norm['options'][k]}" for k in ["A", "B", "C", "D"] if k in norm["options"]]
    options_str = chr(10).join(ordered)

    if format_type == "tag":
        return (
            "You are answering a medical multiple-choice question.\n"
            "Respond in exactly this format:\n"
            "<think>\n<step-by-step reasoning>\n</think>\n"
            "<answer>\nThe answer is <A/B/C/D>. <brief justification>\n</answer>\n\n"
            f"Question:\n{norm['question']}\n\n"
            f"Options:\n{options_str}\n"
        )
    return (
        "You are answering a medical multiple-choice question.\n"
        "Respond in exactly this format:\n"
        f"## {thinking_title}\n"
        "<step-by-step reasoning>\n\n"
        f"## {final_title}\n"
        "The answer is <A/B/C/D>. <brief justification>\n\n"
        f"Question:\n{norm['question']}\n\n"
        f"Options:\n{options_str}\n"
    )


def _mcq4_parse_by_explicit(text: str) -> str | None:
    t = text.strip().upper()
    if t in _MCQ4_VALID:
        return t
    for pat in _MCQ4_EXPLICIT_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    return None


def _mcq4_parse_by_option_text(text: str, options: dict) -> str | None:
    norm_text = _normalize_text(text)
    found = []
    for label, opt in options.items():
        norm_opt = _normalize_text(opt)
        if norm_opt and norm_opt in norm_text:
            found.append((label, norm_text.rfind(norm_opt)))
    if len(found) == 1:
        return found[0][0]
    if len(found) > 1:
        return max(found, key=lambda x: x[1])[0]
    return None


def _mcq4_parse(
    text: str,
    norm: dict,
    final_section_re: re.Pattern | None,
    format_type: str = "header",
) -> tuple[str | None, str]:
    cleaned = _clean_text_for_fallback(text, format_type)
    final = _extract_answer_section(text, final_section_re, format_type)

    # 1. explicit patterns – prefer answer section, fallback to cleaned text
    for scope, src in [(final, "answer_section"), (cleaned, "full_text")]:
        if scope is not None:
            pred = _mcq4_parse_by_explicit(scope)
            if pred:
                return pred, f"{src}:explicit"

    # 2. option text matching
    if final:
        pred = _mcq4_parse_by_option_text(final, norm["options"])
        if pred:
            return pred, "answer_section:option_text"
    pred = _mcq4_parse_by_option_text(cleaned, norm["options"])
    if pred:
        return pred, "full_text:option_text"

    # 3. single letter in answer section
    if final:
        letters = _MCQ4_LETTER_RE.findall(final.upper())
        if len(letters) == 1:
            return letters[0].upper(), "answer_section:single_letter"

    return None, "unparsed"


def _mcq4_build_fallback_prompt(norm: dict, draft_response: str) -> str:
    ordered = [f"{k}. {norm['options'][k]}" for k in ["A", "B", "C", "D"] if k in norm["options"]]
    return (
        "You are extracting the final option from a draft answer to a medical multiple-choice question.\n"
        "Return exactly one line in this format and nothing else:\n"
        "Final Answer: <A/B/C/D>\n\n"
        f"Question:\n{norm['question']}\n\n"
        f"Options:\n{chr(10).join(ordered)}\n\n"
        f"Draft response:\n{draft_response}"
    )


# ---------------------------------------------------------------------------
# MCQ A-J adapter  (MedXpertQA – 10 options)
# ---------------------------------------------------------------------------

_MCQ10_LETTER_RE = re.compile(r"\b([A-J])\b", re.IGNORECASE)
_MCQ10_VALID = set("ABCDEFGHIJ")

_MCQ10_EXPLICIT_PATTERNS = [
    re.compile(r"final\s+answer\s*[:\-]\s*([A-J])\b", re.IGNORECASE),
    re.compile(r"the\s+answer\s+is\s+([A-J])\b", re.IGNORECASE),
    re.compile(r"the\s+correct\s+answer\s+is\s+([A-J])\b", re.IGNORECASE),
    re.compile(r"the\s+correct\s+option\s+is\s+([A-J])\b", re.IGNORECASE),
    re.compile(r"option\s+([A-J])\b", re.IGNORECASE),
    re.compile(r"answer\s*[:\-]\s*([A-J])\b", re.IGNORECASE),
]


def _mcq10_normalize(row: dict) -> dict:
    return {
        "question": row["question"],       # already contains embedded "Answer Choices: (A)...(J)..."
        "options": dict(row["options"]),   # kept for option-text fallback matching
        "gold": str(row["label"]).strip().upper(),
        "id": row.get("id"),
        "medical_task": row.get("medical_task"),
        "body_system": row.get("body_system"),
        "question_type": row.get("question_type"),
    }


def _mcq10_build_prompt(
    norm: dict, thinking_title: str, final_title: str, format_type: str = "header"
) -> str:
    if format_type == "tag":
        return (
            "You are answering a medical multiple-choice question.\n"
            "Respond in exactly this format:\n"
            "<think>\n<step-by-step reasoning>\n</think>\n"
            "<answer>\nThe answer is <A/B/C/D/E/F/G/H/I/J>. <brief justification>\n</answer>\n\n"
            f"{norm['question']}\n"
        )
    return (
        "You are answering a medical multiple-choice question.\n"
        "Respond in exactly this format:\n"
        f"## {thinking_title}\n"
        "<step-by-step reasoning>\n\n"
        f"## {final_title}\n"
        "The answer is <A/B/C/D/E/F/G/H/I/J>. <brief justification>\n\n"
        f"{norm['question']}\n"
    )


def _mcq10_parse_by_explicit(text: str) -> str | None:
    t = text.strip().upper()
    if t in _MCQ10_VALID:
        return t
    for pat in _MCQ10_EXPLICIT_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    return None


def _mcq10_parse(
    text: str,
    norm: dict,
    final_section_re: re.Pattern | None,
    format_type: str = "header",
) -> tuple[str | None, str]:
    cleaned = _clean_text_for_fallback(text, format_type)
    final = _extract_answer_section(text, final_section_re, format_type)

    # 1. explicit patterns
    for scope, src in [(final, "answer_section"), (cleaned, "full_text")]:
        if scope is not None:
            pred = _mcq10_parse_by_explicit(scope)
            if pred:
                return pred, f"{src}:explicit"

    # 2. option text matching
    if final:
        pred = _mcq4_parse_by_option_text(final, norm["options"])
        if pred and pred in _MCQ10_VALID:
            return pred, "answer_section:option_text"
    pred = _mcq4_parse_by_option_text(cleaned, norm["options"])
    if pred and pred in _MCQ10_VALID:
        return pred, "full_text:option_text"

    # 3. single letter in answer section
    if final:
        letters = _MCQ10_LETTER_RE.findall(final.upper())
        if len(letters) == 1:
            return letters[0].upper(), "answer_section:single_letter"

    return None, "unparsed"


def _mcq10_build_fallback_prompt(norm: dict, draft_response: str) -> str:
    return (
        "You are extracting the final option from a draft answer to a medical multiple-choice question.\n"
        "Return exactly one line in this format and nothing else:\n"
        "Final Answer: <A/B/C/D/E/F/G/H/I/J>\n\n"
        f"Question:\n{norm['question']}\n\n"
        f"Draft response:\n{draft_response}"
    )


# ---------------------------------------------------------------------------
# PubMedQA adapter  (yes / no / maybe)
# ---------------------------------------------------------------------------

_PUBMED_RE = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)
_PUBMED_VALID = {"yes", "no", "maybe"}

_PUBMED_EXPLICIT_PATTERNS = [
    re.compile(r"final\s+answer\s*[:\-]\s*(yes|no|maybe)\b", re.IGNORECASE),
    re.compile(r"the\s+answer\s+is\s+(yes|no|maybe)\b", re.IGNORECASE),
    re.compile(r"the\s+correct\s+answer\s+is\s+(yes|no|maybe)\b", re.IGNORECASE),
    re.compile(r"answer\s*[:\-]\s*(yes|no|maybe)\b", re.IGNORECASE),
]


def _pubmedqa_normalize(row: dict) -> dict:
    paragraphs = row.get("context", {}).get("contexts", [])
    context_text = "\n\n".join(p.strip() for p in paragraphs if p and p.strip())
    return {
        "question": row["question"],
        "context_text": context_text,
        "gold": str(row["final_decision"]).strip().lower(),
        "pubid": row.get("pubid"),
    }


def _pubmedqa_build_prompt(
    norm: dict, thinking_title: str, final_title: str, format_type: str = "header"
) -> str:
    if format_type == "tag":
        return (
            "You are answering a biomedical yes/no/maybe question based on the abstract.\n"
            "Respond in exactly this format:\n"
            "<think>\n<step-by-step reasoning>\n</think>\n"
            "<answer>\nThe answer is yes/no/maybe. <brief justification>\n</answer>\n\n"
            f"Abstract:\n{norm['context_text']}\n\n"
            f"Question:\n{norm['question']}\n"
        )
    return (
        "You are answering a biomedical yes/no/maybe question based on the abstract.\n"
        "Respond in exactly this format:\n"
        f"## {thinking_title}\n"
        "<step-by-step reasoning>\n\n"
        f"## {final_title}\n"
        "The answer is yes/no/maybe. <brief justification>\n\n"
        f"Abstract:\n{norm['context_text']}\n\n"
        f"Question:\n{norm['question']}\n"
    )


def _pubmedqa_parse_by_explicit(text: str) -> str | None:
    t = text.strip().lower()
    if t in _PUBMED_VALID:
        return t
    for pat in _PUBMED_EXPLICIT_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).lower()
    return None


def _pubmedqa_parse(
    text: str,
    norm: dict,
    final_section_re: re.Pattern | None,
    format_type: str = "header",
) -> tuple[str | None, str]:
    cleaned = _clean_text_for_fallback(text, format_type)
    final = _extract_answer_section(text, final_section_re, format_type)

    # 1. explicit patterns
    for scope, src in [(final, "answer_section"), (cleaned, "full_text")]:
        if scope is not None:
            pred = _pubmedqa_parse_by_explicit(scope)
            if pred:
                return pred, f"{src}:explicit"

    # 2. single word in answer section
    if final:
        words = _PUBMED_RE.findall(final.lower())
        if len(words) == 1:
            return words[0].lower(), "answer_section:single_word"

    return None, "unparsed"


def _pubmedqa_build_fallback_prompt(norm: dict, draft_response: str) -> str:
    return (
        "You are extracting the final answer from a draft response to a biomedical question.\n"
        "Return exactly one line in this format and nothing else:\n"
        "Final Answer: yes/no/maybe\n\n"
        f"Question:\n{norm['question']}\n\n"
        f"Draft response:\n{draft_response}"
    )


# ---------------------------------------------------------------------------
# ADAPTERS registry
# ---------------------------------------------------------------------------

ADAPTERS: dict[str, DatasetAdapter] = {
    "medqa": DatasetAdapter(
        default_path="dataset/MedQA",
        default_split="test",
        default_num_samples=20,
        default_output_dir="results/medqa_sft_eval",
        normalize=_mcq4_normalize,
        build_prompt=_mcq4_build_prompt,
        parse=_mcq4_parse,
        build_fallback_prompt=_mcq4_build_fallback_prompt,
    ),
    "medmcqa": DatasetAdapter(
        default_path="dataset/MedMCQA_4options",
        default_split="validation",
        default_num_samples=20,
        default_output_dir="results/medmcqa_sft_eval",
        normalize=_mcq4_normalize,
        build_prompt=_mcq4_build_prompt,
        parse=_mcq4_parse,
        build_fallback_prompt=_mcq4_build_fallback_prompt,
    ),
    "medxpertqa": DatasetAdapter(
        default_path="dataset/MedXpertQA_Text",
        default_split="test",
        default_num_samples=2450,
        default_output_dir="results/medxpertqa_sft_eval",
        normalize=_mcq10_normalize,
        build_prompt=_mcq10_build_prompt,
        parse=_mcq10_parse,
        build_fallback_prompt=_mcq10_build_fallback_prompt,
    ),
    "pubmedqa": DatasetAdapter(
        default_path="dataset/PubMedQA",
        default_split="train",
        default_num_samples=1000,
        default_output_dir="results/pubmedqa_sft_eval",
        normalize=_pubmedqa_normalize,
        build_prompt=_pubmedqa_build_prompt,
        parse=_pubmedqa_parse,
        build_fallback_prompt=_pubmedqa_build_fallback_prompt,
    ),
}

# Training style presets: map --training → default section titles + format type.
# "format" controls prompt construction and output parsing:
#   "header" – ## Thinking / ## Final Response  (markdown headers)
#   "tag"    – <think>...</think> <answer>...</answer>  (XML-like tags)
TRAINING_STYLES: dict[str, dict[str, str]] = {
    "huatuo":    {"thinking_title": "Thinking", "final_title": "Final Response", "format": "header"},
    "medreason": {"thinking_title": "Thinking", "final_title": "Final Response", "format": "header"},
    "think_tag": {"thinking_title": "", "final_title": "", "format": "tag"},
    "custom":    {"thinking_title": "Thinking", "final_title": "Final Response", "format": "header"},
}


# ---------------------------------------------------------------------------
# Shared evaluation loop
# ---------------------------------------------------------------------------

def _force_extract(
    *,
    tokenizer,
    model,
    model_input_device,
    norm: dict,
    draft_response: str,
    adapter: DatasetAdapter,
    use_cache: bool,
    prompt_max_length: int,
    final_section_re: re.Pattern | None,
    format_type: str,
) -> tuple[str | None, str]:
    """Re-prompt the model to extract a clean answer from a malformed draft."""
    prompt = adapter.build_fallback_prompt(norm, draft_response)
    text = render_chat(tokenizer, prompt)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=prompt_max_length,
    ).to(model_input_device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False,
            use_cache=use_cache,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    decoded = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
    pred, _ = adapter.parse(decoded, norm, final_section_re, format_type)
    return pred, decoded


def run_benchmark(
    *,
    model_path: Path,
    dataset,
    adapter: DatasetAdapter,
    num_samples: int,
    batch_size: int,
    seed: int,
    max_new_tokens: int,
    prompt_max_length: int,
    use_cache: bool,
    run_tag: str,
    output_dir: Path,
    device_preference: str,
    thinking_title: str,
    final_title: str,
    format_type: str,
    dataset_name: str,
    training_style: str,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    wants_cuda = device_preference == "cuda" or (
        device_preference == "auto" and torch.cuda.is_available()
    )
    load_kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}
    if wants_cuda:
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()
    model_input_device = next(model.parameters()).device

    if format_type == "tag":
        final_section_re = None
    else:
        final_section_re = _make_final_section_re(final_title)

    sampled = dataset.shuffle(seed=seed).select(range(min(num_samples, len(dataset))))
    records: list[dict] = []
    correct = invalid = fallback_attempts = fallback_recovered = 0
    started = time.time()

    total_batches = (len(sampled) + batch_size - 1) // batch_size
    progress = tqdm(
        range(0, len(sampled), batch_size),
        total=total_batches,
        desc=model_path.name,
        unit="batch",
        leave=False,
    )

    for i in progress:
        batch = sampled.select(range(i, min(i + batch_size, len(sampled))))
        rows = [adapter.normalize(row) for row in batch]
        prompts = [
            adapter.build_prompt(r, thinking_title, final_title, format_type)
            for r in rows
        ]
        texts = [render_chat(tokenizer, p) for p in prompts]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=prompt_max_length,
        ).to(model_input_device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=use_cache,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        for idx, norm in enumerate(rows):
            decoded = tokenizer.decode(outputs[idx][prompt_len:], skip_special_tokens=True).strip()
            pred, parse_source = adapter.parse(decoded, norm, final_section_re, format_type)
            fallback_used = False
            fallback_output = None

            if pred is None:
                fallback_attempts += 1
                fallback_used = True
                pred, fallback_output = _force_extract(
                    tokenizer=tokenizer,
                    model=model,
                    model_input_device=model_input_device,
                    norm=norm,
                    draft_response=decoded,
                    adapter=adapter,
                    use_cache=use_cache,
                    prompt_max_length=prompt_max_length,
                    final_section_re=final_section_re,
                    format_type=format_type,
                )
                if pred is not None:
                    fallback_recovered += 1
                    parse_source = "fallback_extract"

            gold = norm["gold"]
            ok = pred == gold
            if ok:
                correct += 1
            if pred is None:
                invalid += 1

            record: dict = {
                "question": norm["question"],
                "pred": pred,
                "gold": gold,
                "raw_output": decoded,
                "correct": ok,
                "run_tag": run_tag,
                "parse_source": parse_source,
                "fallback_used": fallback_used,
                "fallback_output": fallback_output,
            }
            # Preserve extra fields from normalized row (id, pubid, medical_task, …)
            record.update(
                {k: v for k, v in norm.items() if k not in ("question", "gold", "options", "context_text")}
            )
            records.append(record)

        processed = min(i + len(batch), len(sampled))
        if processed > 0:
            progress.set_postfix(
                {"acc": f"{correct / processed:.3f}", "invalid": invalid, "fb_ok": fallback_recovered},
                refresh=False,
            )

    elapsed = time.time() - started
    total = len(records)
    acc = correct / total if total else 0.0

    if format_type == "tag":
        expected_fmt = "<think>...</think> <answer>The answer is <label></answer>"
    else:
        expected_fmt = f"## {thinking_title} / ## {final_title} / The answer is <label>"

    result = {
        "model": str(model_path),
        "dataset": dataset_name,
        "training_style": training_style,
        "format_type": format_type,
        "thinking_section_title": thinking_title if format_type == "header" else "<think>",
        "final_section_title": final_title if format_type == "header" else "<answer>",
        "num_samples": total,
        "correct": correct,
        "accuracy": acc,
        "invalid_predictions": invalid,
        "elapsed_seconds": elapsed,
        "samples_per_second": (total / elapsed) if elapsed > 0 else 0.0,
        "device": str(model_input_device),
        "max_new_tokens": max_new_tokens,
        "prompt_max_length": prompt_max_length,
        "use_cache": use_cache,
        "run_tag": run_tag,
        "fallback_attempts": fallback_attempts,
        "fallback_recovered": fallback_recovered,
        "expected_output_format": expected_fmt,
    }

    model_out = output_dir / model_path.name
    model_out.mkdir(parents=True, exist_ok=True)
    (model_out / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (model_out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    del model
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified SFT post-training evaluation for medical QA datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Available datasets : {', '.join(ADAPTERS)}\n"
            f"Available trainings: {', '.join(TRAINING_STYLES)}"
        ),
    )
    parser.add_argument(
        "--training",
        required=True,
        choices=list(TRAINING_STYLES),
        help="Training pipeline whose format this model was trained with.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(ADAPTERS),
        help="Benchmark dataset to evaluate on.",
    )
    parser.add_argument("--model-path", required=True, help="Path to the fine-tuned model checkpoint.")
    parser.add_argument("--dataset-path", default="", help="Override dataset path (adapter default used if empty).")
    parser.add_argument("--split", default="", help="Dataset split (adapter default used if empty).")
    parser.add_argument("--num-samples", type=int, default=0, help="Samples to evaluate (adapter default if 0).")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--prompt-max-length", type=int, default=3072)
    parser.add_argument("--output-dir", default="", help="Override output directory (adapter default if empty).")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--thinking-section-title",
        default="",
        help="Override thinking section header (training style default used if empty). Ignored for think_tag format.",
    )
    parser.add_argument(
        "--final-section-title",
        default="",
        help="Override final answer section header (training style default used if empty). Ignored for think_tag format.",
    )
    parser.add_argument(
        "--use-cache", dest="use_cache", action="store_true", help="Enable KV cache (default).",
    )
    parser.add_argument(
        "--no-use-cache", dest="use_cache", action="store_false", help="Disable KV cache.",
    )
    parser.add_argument("--tag", default="", help="Optional label appended to run_tag in output metadata.")
    parser.set_defaults(use_cache=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = ADAPTERS[args.dataset]
    style_defaults = TRAINING_STYLES[args.training]

    format_type = style_defaults.get("format", "header")
    dataset_path = args.dataset_path or adapter.default_path
    split = args.split or adapter.default_split
    num_samples = args.num_samples or adapter.default_num_samples
    output_dir = Path(args.output_dir or adapter.default_output_dir)
    thinking_title = args.thinking_section_title or style_defaults.get("thinking_title", "")
    final_title = args.final_section_title or style_defaults.get("final_title", "")

    if args.training == "pubmedqa" and args.dataset != "pubmedqa":
        print(
            "[warning] PubMedQA models expect yes/no/maybe output — using one on an MCQ "
            "dataset will likely yield poor results."
        )
    if args.dataset == "pubmedqa" and args.training not in ("custom",):
        print(
            f"[info] Evaluating '{args.dataset}' with training style '{args.training}'. "
            "Ensure the model was trained on PubMedQA-style data for meaningful results."
        )

    dataset = load_from_disk(dataset_path)[split]
    model_path = Path(args.model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_tag = f"{args.dataset}_{args.training}_sft_eval"
    if args.tag:
        run_tag = f"{run_tag}_{args.tag}"

    if format_type == "tag":
        sections_display = "<think>...</think> / <answer>...</answer>"
    else:
        sections_display = f"## {thinking_title} / ## {final_title}"

    print(
        f"\n=== SFT Eval ===\n"
        f"  model    : {model_path.name}\n"
        f"  dataset  : {args.dataset.upper()} ({split}, n={num_samples})\n"
        f"  training : {args.training} (format={format_type})\n"
        f"  sections : {sections_display}\n"
        f"  output   : {output_dir}\n"
    )

    result = run_benchmark(
        model_path=model_path,
        dataset=dataset,
        adapter=adapter,
        num_samples=num_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        prompt_max_length=args.prompt_max_length,
        use_cache=args.use_cache,
        run_tag=run_tag,
        output_dir=output_dir,
        device_preference=args.device,
        thinking_title=thinking_title,
        final_title=final_title,
        format_type=format_type,
        dataset_name=args.dataset,
        training_style=args.training,
    )

    print(
        f"Accuracy : {result['accuracy']:.4f} ({result['correct']}/{result['num_samples']}) | "
        f"invalid={result['invalid_predictions']} | "
        f"fallback={result['fallback_recovered']}/{result['fallback_attempts']} | "
        f"time={result['elapsed_seconds']:.1f}s | "
        f"samples/s={result['samples_per_second']:.3f}"
    )


if __name__ == "__main__":
    main()
