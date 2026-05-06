"""Reward functions redesigned for GDPO orthogonality.

GDPO requires each reward to measure a truly independent behavioral dimension
so that per-reward normalized advantages carry distinct gradient signals.

Problem with reward_fns.py (original):
    format_reward  → rewards tool call (+0.25 if call+response)
    enhanced_tool_quality_reward → ALSO rewards tool call (-0.30/+0.10 base)
    → both push gradient toward/against tool calling simultaneously
    → effective tool-call weight ≈ 0.50 (0.25+0.25), squeezing answer+quality

Fix — strict separation into 3 orthogonal dimensions:

  [0.20]  structure_reward  — tag compliance (<think>, <answer>) ONLY
  [0.50]  answer_reward     — correctness (unchanged from original)
  [0.30]  tool_reward       — ALL tool signals: frequency + semantic quality

GDPO config to use with these:
    GRPOConfig(
        multi_objective_aggregation="normalize_then_sum",
        reward_weights=[0.20, 0.50, 0.30],
        scale_rewards="group",
        loss_type="dapo",
    )
    reward_funcs=[structure_reward, answer_reward, tool_reward]
"""

import json
import re
from collections import Counter

import numpy as np

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_LETTER_RE = re.compile(r"^\s*([A-Ea-e])[.\):\s]")
_FINAL_LETTER_RE = re.compile(
    r"(?:answer\s+is|therefore|correct\s+(?:answer|option)[:\s]*)\s*"
    r"(?:\*\*)?([A-Ea-e])\b",
    re.IGNORECASE,
)
_TOOL_CALL_JSON_RE = re.compile(
    r"(?:<tool_call>|<\|python_tag\|>)\s*(\{.*?\})\s*(?:</tool_call>|<\|eom_id\|>)"
    r"|(\{(?:[^{}]|\{[^{}]*\})*\})\s*(?:<\|eot_id\|>|<\|eom_id\|>)",
    re.DOTALL,
)
_WORD_RE = re.compile(r"[a-z0-9]{3,}")
_GROUNDING_STOPS = frozenset({
    "the", "and", "are", "for", "from", "has", "have", "that", "this",
    "with", "was", "were", "been", "being", "will", "would", "could",
    "should", "may", "can", "not", "but", "also", "which", "their",
    "there", "than", "other", "more", "most", "such", "when", "what",
    "its", "does", "did", "had", "into", "over", "between", "through",
    "often", "includes", "including", "associated", "patients",
})


def _get_assistant_content(completion: list[dict]) -> str:
    return " ".join(
        turn.get("content", "")
        for turn in completion
        if turn.get("role") == "assistant" and turn.get("content")
    )


# ---------------------------------------------------------------------------
# R1: structure_reward — tags ONLY, no tool component
# ---------------------------------------------------------------------------

def structure_reward(completions, **kwargs) -> list[float]:
    """Reward for <think> and <answer> tag compliance.

    Intentionally does NOT reward tool calls — that is tool_reward's job.
    This separation is required for GDPO orthogonality.

    Scoring:
        +0.25  <think>...</think> present
        +0.50  <answer>...</answer> present
        +0.25  bonus: answer is concise (≤10 words) → MCQ compliance
    Max: 1.00
    """
    rewards = []
    for completion in completions:
        score = 0.0
        all_content = _get_assistant_content(completion)

        if _THINK_RE.search(all_content):
            score += 0.25

        answer_match = _ANSWER_RE.search(all_content)
        if answer_match:
            score += 0.50
            if len(answer_match.group(1).strip().split()) <= 10:
                score += 0.25

        rewards.append(score)
    return rewards


# ---------------------------------------------------------------------------
# R2: answer_reward — correctness (unchanged from original)
# ---------------------------------------------------------------------------

def _token_f1(pred: str, gt: str) -> float:
    pred_tokens = pred.lower().split()
    gt_tokens = gt.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_reward(completions, answer, answer_idx=None, **kwargs) -> list[float]:
    """MCQ-aware reward: exact > letter > in-text letter > substring > token F1.

    Unchanged from original reward_fns.py — correctness is already orthogonal.
    """
    gt_indices = answer_idx if answer_idx is not None else [None] * len(answer)
    rewards = []

    for completion, gt, gt_idx in zip(completions, answer, gt_indices):
        all_content = _get_assistant_content(completion)

        match = _ANSWER_RE.search(all_content)
        if match:
            pred = match.group(1).strip()
        else:
            pred = ""
            for turn in reversed(completion):
                if turn.get("role") == "assistant" and turn.get("content"):
                    pred = turn["content"].strip()
                    break

        if not pred:
            rewards.append(0.0)
            continue

        if pred.lower() == gt.lower():
            rewards.append(1.0)
            continue

        if gt_idx:
            gt_letter = gt_idx.strip().upper()
            letter_match = _LETTER_RE.match(pred)
            if letter_match and letter_match.group(1).upper() == gt_letter:
                rewards.append(1.0)
                continue
            if pred.strip().upper() == gt_letter:
                rewards.append(1.0)
                continue
            final_letter = _FINAL_LETTER_RE.search(pred)
            if final_letter and final_letter.group(1).upper() == gt_letter:
                rewards.append(0.8)
                continue

        if gt.lower() in pred.lower():
            rewards.append(0.5)
            continue

        rewards.append(_token_f1(pred, gt))

    return rewards


# ---------------------------------------------------------------------------
# R3: tool_reward — ALL tool signals (frequency + quality)
# ---------------------------------------------------------------------------

def _get_encoder():
    from scripts.serve.retrieval_tool import MedicalKnowledgeTool
    inst = MedicalKnowledgeTool._instance
    return inst.encoder if inst is not None else None


def _extract_tool_queries(completion: list[dict]) -> list[str]:
    queries = []
    for turn in completion:
        if turn.get("tool_calls"):
            for tc in turn["tool_calls"]:
                try:
                    fn = tc.get("function", {})
                    args = fn.get("arguments") or fn.get("parameters", "{}")
                    if isinstance(args, str):
                        args = json.loads(args)
                    q = args.get("query", "")
                    if q:
                        queries.append(q)
                except (json.JSONDecodeError, AttributeError):
                    pass
        content = turn.get("content", "")
        has_tool = (
            "<tool_call>" in content
            or "<|python_tag|>" in content
            or ('"name"' in content and '"parameters"' in content)
        )
        if content and has_tool:
            for m in _TOOL_CALL_JSON_RE.finditer(content):
                try:
                    tc = json.loads(m.group(1) or m.group(2))
                    args = tc.get("arguments") or tc.get("parameters", {})
                    if isinstance(args, str):
                        args = json.loads(args)
                    q = args.get("query", "")
                    if q and q not in queries:
                        queries.append(q)
                except (json.JSONDecodeError, AttributeError):
                    pass
    return queries


def _tokenize_for_grounding(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _GROUNDING_STOPS}


def tool_reward(
    completions, prompts=None, answer=None, **kwargs,
) -> list[float]:
    """Single reward owning ALL tool-related gradient signal.

    Combines frequency decision + semantic quality into one orthogonal dimension.
    structure_reward must NOT include any tool component for this to work.

    Frequency tier (base score):
        0 calls  → -0.40  stronger penalty since structure_reward no longer
                           provides implicit -0.25 for missing tool
        1-2 calls → +0.10 base
        3+ calls  → +0.05 base (neutral, don't punish exploration)

    Semantic quality (added on top when n_calls > 0):
        Signal 1 — query relevance:   cosine(question, tool_query)       → [0, +0.10]
        Signal 2 — retrieval quality: cosine(question+answer, KG_result) → [0, +0.15]
        Signal 3 — grounding bonus:   post-tool reasoning cites KG       → [0, +0.05]

    Total range: [-0.40, +0.40]

    Why -0.40 (vs -0.30 in original)?
        original format_reward gave implicit -0.25 penalty when no tool was called
        (because 0.25 points were not awarded). Removing that component from
        structure_reward means tool_reward must compensate with a stronger
        explicit penalty to maintain the same incentive gradient.
    """
    encoder = _get_encoder()

    # --- Phase 1: Extract per-completion data ---
    _TOOL_CALL_TEXT_RE = re.compile(
        r"<tool_call>|<\|python_tag\|>|\"name\"\s*:\s*\"search_medical_knowledge\""
    )

    batch = []
    for i, completion in enumerate(completions):
        # Count structured tool_calls (OpenAI-style) OR text-based <tool_call> (Qwen3-style)
        n_calls = sum(
            1 for t in completion
            if t.get("tool_calls") or (
                t.get("role") == "assistant"
                and _TOOL_CALL_TEXT_RE.search(t.get("content", ""))
            )
        )

        if n_calls == 0:
            batch.append({"base": -0.40, "skip": True})
            continue

        base = 0.10 if n_calls <= 2 else 0.05

        question = ""
        if prompts and i < len(prompts):
            for turn in prompts[i]:
                if turn.get("role") == "user":
                    question = turn.get("content", "")
                    break

        queries = _extract_tool_queries(completion)
        tool_responses = [
            t.get("content", "") for t in completion if t.get("role") == "tool"
        ]

        post_tool_thinks = []
        saw_tool = False
        for turn in completion:
            if turn.get("role") == "tool":
                saw_tool = True
            elif turn.get("role") == "assistant" and saw_tool:
                post_tool_thinks.extend(_THINK_RE.findall(turn.get("content", "")))

        gt = answer[i] if answer and i < len(answer) else ""

        batch.append({
            "base": base, "skip": False,
            "question": question, "queries": queries,
            "tool_responses": tool_responses,
            "post_tool_thinks": post_tool_thinks,
            "gt_answer": gt,
        })

    if encoder is None:
        return [d["base"] for d in batch]

    # --- Phase 2: Batch encode ---
    texts_to_encode: list[str] = []
    text_map: list[tuple[int, str, int]] = []

    for i, d in enumerate(batch):
        if d["skip"]:
            continue
        if d["question"]:
            text_map.append((i, "question", len(texts_to_encode)))
            texts_to_encode.append(d["question"])
        if d["question"] and d["gt_answer"]:
            text_map.append((i, "qa_anchor", len(texts_to_encode)))
            texts_to_encode.append(f"{d['question']} The answer is {d['gt_answer']}")
        for j, q in enumerate(d["queries"]):
            text_map.append((i, f"query_{j}", len(texts_to_encode)))
            texts_to_encode.append(q)
        for j, r in enumerate(d["tool_responses"]):
            text_map.append((i, f"response_{j}", len(texts_to_encode)))
            texts_to_encode.append(r[:512])

    if texts_to_encode:
        all_embs = encoder.encode(
            texts_to_encode, normalize_embeddings=True, batch_size=64,
        )
    else:
        all_embs = np.empty((0, 0))

    emb_lookup: dict[tuple[int, str], np.ndarray] = {}
    for idx, label, pos in text_map:
        emb_lookup[(idx, label)] = all_embs[pos]

    # --- Phase 3: Compute signals ---
    rewards = []
    for i, d in enumerate(batch):
        if d["skip"]:
            rewards.append(d["base"])
            continue

        score = d["base"]

        # Signal 1: Query relevance (capped at 0.80 to discourage copy-paste queries)
        q_emb = emb_lookup.get((i, "question"))
        if q_emb is not None and d["queries"]:
            sims = [
                float(np.dot(q_emb, emb_lookup[(i, f"query_{j}")]))
                for j in range(len(d["queries"]))
                if (i, f"query_{j}") in emb_lookup
            ]
            if sims:
                best = max(sims)
                if best > 0.80:
                    best = 0.80 + (best - 0.80) * 0.3
                score += min(0.10, max(0.0, (best - 0.5) * 0.2))

        # Signal 2: Retrieval quality (primary signal — outcome > query phrasing)
        qa_emb = emb_lookup.get((i, "qa_anchor"))
        if qa_emb is not None and d["tool_responses"]:
            sims = [
                float(np.dot(qa_emb, emb_lookup[(i, f"response_{j}")]))
                for j in range(len(d["tool_responses"]))
                if (i, f"response_{j}") in emb_lookup
            ]
            if sims:
                score += min(0.15, max(0.0, (max(sims) - 0.55) * 0.33))

        # Signal 3: Grounding (post-tool reasoning references KG entities)
        if d["post_tool_thinks"] and d["tool_responses"]:
            tool_tokens = _tokenize_for_grounding(" ".join(d["tool_responses"]))
            post_tokens = _tokenize_for_grounding(" ".join(d["post_tool_thinks"]))
            if post_tokens and tool_tokens:
                overlap = len(post_tokens & tool_tokens)
                if overlap >= 3 and overlap / min(len(post_tokens), 15) >= 0.2:
                    score += 0.05

        rewards.append(score)

    return rewards
