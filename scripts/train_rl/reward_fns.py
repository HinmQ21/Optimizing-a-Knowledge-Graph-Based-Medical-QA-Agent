"""Reward functions for GRPO medical training.

Three reward functions scored independently, combined via reward_weights:
  [0.25]  format_reward       — structured output compliance
  [0.50]  answer_reward       — exact / letter / substring / token-F1
  [0.25]  tool_quality_reward — tool usage quality
"""

import re
from collections import Counter

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# Patterns for extracting a letter answer from free-form text
_LETTER_RE = re.compile(r"^\s*([A-Ea-e])[.\):\s]")  # "A." / "A)" / "A:" / "A "
_FINAL_LETTER_RE = re.compile(
    r"(?:answer\s+is|therefore|correct\s+(?:answer|option)[:\s]*)\s*"
    r"(?:\*\*)?([A-Ea-e])\b",
    re.IGNORECASE,
)


def _get_assistant_content(completion: list[dict]) -> str:
    """Concatenate all assistant message content from a multi-turn completion."""
    return " ".join(
        turn.get("content", "")
        for turn in completion
        if turn.get("role") == "assistant" and turn.get("content")
    )


def format_reward(completions, **kwargs) -> list[float]:
    """Reward for structured output: tool usage + <think>/<answer> tags.

    Scoring:
        +0.25  tool call issued AND tool response received
        +0.25  <think>...</think> present
        +0.50  <answer>...</answer> present (bonus +0.25 if answer is concise)
    """
    rewards = []
    for completion in completions:
        score = 0.0
        all_content = _get_assistant_content(completion)

        has_tool_call = any(t.get("tool_calls") for t in completion)
        has_tool_response = any(t.get("role") == "tool" for t in completion)
        if has_tool_call and has_tool_response:
            score += 0.25

        if _THINK_RE.search(all_content):
            score += 0.25

        answer_match = _ANSWER_RE.search(all_content)
        if answer_match:
            score += 0.50
            # Bonus for concise MCQ-style answers (<=10 words)
            answer_text = answer_match.group(1).strip()
            if len(answer_text.split()) <= 10:
                score += 0.25

        rewards.append(score)
    return rewards


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

    Args:
        completions: list of completions (each is list[dict] of conversation turns).
        answer:      list[str] ground-truth full answer text.
        answer_idx:  list[str] correct option letter (e.g. "A","B","C","D").
                     Passed automatically when 'answer_idx' column is in the dataset.
    """
    gt_indices = answer_idx if answer_idx is not None else [None] * len(answer)
    rewards = []

    for completion, gt, gt_idx in zip(completions, answer, gt_indices):
        all_content = _get_assistant_content(completion)

        # Extract prediction from <answer> tag; fallback to last assistant turn
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

        # 1. Exact text match (case-insensitive)
        if pred.lower() == gt.lower():
            rewards.append(1.0)
            continue

        # 2. Letter match — model outputs "A" / "A." / "A)" and it's the right letter
        if gt_idx:
            gt_letter = gt_idx.strip().upper()

            letter_match = _LETTER_RE.match(pred)
            if letter_match and letter_match.group(1).upper() == gt_letter:
                rewards.append(1.0)
                continue
            # Bare single-letter answer: "A"
            if pred.strip().upper() == gt_letter:
                rewards.append(1.0)
                continue

            # 3. In-text letter — "the answer is A", "therefore A", "correct option: A"
            final_letter = _FINAL_LETTER_RE.search(pred)
            if final_letter and final_letter.group(1).upper() == gt_letter:
                rewards.append(0.8)
                continue

        # 4. Substring match — GT text appears within the prediction
        if gt.lower() in pred.lower():
            rewards.append(0.5)
            continue

        # 5. Token F1 fallback (partial credit)
        rewards.append(_token_f1(pred, gt))

    return rewards


def tool_quality_reward(completions, **kwargs) -> list[float]:
    """Reward for appropriate tool usage frequency.

    Scoring:
        0 calls  -> -0.3  (strong penalty to prevent tool-calling collapse)
        1-2 calls -> +0.4 (good usage, rewarded more strongly)
        3+ calls  ->  0.0 (neutral — don't punish exploration)
    """
    rewards = []
    for completion in completions:
        n_calls = sum(1 for turn in completion if turn.get("tool_calls"))
        if n_calls == 0:
            rewards.append(-0.3)
        elif n_calls <= 2:
            rewards.append(0.4)
        else:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Enhanced tool quality reward
# ---------------------------------------------------------------------------

import json
import numpy as np

_TOOL_CALL_JSON_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_WORD_RE = re.compile(r"[a-z0-9]{3,}")
_GROUNDING_STOPS = frozenset({
    "the", "and", "are", "for", "from", "has", "have", "that", "this",
    "with", "was", "were", "been", "being", "will", "would", "could",
    "should", "may", "can", "not", "but", "also", "which", "their",
    "there", "than", "other", "more", "most", "such", "when", "what",
    "its", "does", "did", "had", "into", "over", "between", "through",
    "often", "includes", "including", "associated", "patients",
})


def _get_encoder():
    """Get MedEmbed encoder from the already-loaded MedicalKnowledgeTool singleton."""
    from scripts.serve.retrieval_tool import MedicalKnowledgeTool
    inst = MedicalKnowledgeTool._instance
    if inst is not None:
        return inst.encoder
    return None


def _extract_tool_queries(completion: list[dict]) -> list[str]:
    """Extract search queries from tool calls in a completion."""
    queries = []
    for turn in completion:
        # Structured tool_calls (TRL parsed format)
        if turn.get("tool_calls"):
            for tc in turn["tool_calls"]:
                try:
                    args = tc.get("function", {}).get("arguments", "{}")
                    if isinstance(args, str):
                        args = json.loads(args)
                    q = args.get("query", "")
                    if q:
                        queries.append(q)
                except (json.JSONDecodeError, AttributeError):
                    pass
        # Raw <tool_call> tags in content (fallback)
        content = turn.get("content", "")
        if content and "<tool_call>" in content:
            for m in _TOOL_CALL_JSON_RE.finditer(content):
                try:
                    tc = json.loads(m.group(1))
                    args = tc.get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args)
                    q = args.get("query", "")
                    if q and q not in queries:
                        queries.append(q)
                except (json.JSONDecodeError, AttributeError):
                    pass
    return queries


def _tokenize_for_grounding(text: str) -> set[str]:
    """Tokenize text for grounding overlap, removing stop words."""
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _GROUNDING_STOPS}


def enhanced_tool_quality_reward(
    completions, prompts=None, answer=None, **kwargs,
) -> list[float]:
    """Enhanced reward for tool usage with semantic quality signals.

    Three signals beyond frequency:
      1. Query relevance:   cosine(question, tool_query)        → [0, 0.10]
      2. Retrieval quality: cosine(question+answer, KG_result)  → [0, 0.15]
      3. Grounding bonus:   post-tool reasoning references KG   → [0, 0.05]

    Signal 2 weighted higher than Signal 1 because retrieval outcome matters
    more than query phrasing (copy-paste queries inflate query cosine but
    may still retrieve poorly).

    Scoring:
        0 calls  → -0.30  (prevent collapse)
        1-2 calls → +0.10 base + signals  → up to +0.40
        3+ calls  → +0.05 base + signals  → up to +0.35

    Total range: [-0.30, +0.40] (compatible with existing reward_weights)
    """
    encoder = _get_encoder()

    # --- Phase 1: Extract data per completion ---
    batch = []
    for i, completion in enumerate(completions):
        n_calls = sum(1 for t in completion if t.get("tool_calls"))

        if n_calls == 0:
            batch.append({"base": -0.3, "skip": True})
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

        # Post-tool reasoning: <think> in assistant turns after any tool response
        post_tool_thinks = []
        saw_tool = False
        for turn in completion:
            if turn.get("role") == "tool":
                saw_tool = True
            elif turn.get("role") == "assistant" and saw_tool:
                post_tool_thinks.extend(
                    _THINK_RE.findall(turn.get("content", ""))
                )

        gt = answer[i] if answer and i < len(answer) else ""

        batch.append({
            "base": base, "skip": False,
            "question": question, "queries": queries,
            "tool_responses": tool_responses,
            "post_tool_thinks": post_tool_thinks,
            "gt_answer": gt,
        })

    # --- Phase 2: Batch encode with MedEmbed ---
    if encoder is None:
        return [d["base"] for d in batch]

    texts_to_encode = []
    text_map: list[tuple[int, str, int]] = []

    for i, d in enumerate(batch):
        if d["skip"]:
            continue
        if d["question"]:
            text_map.append((i, "question", len(texts_to_encode)))
            texts_to_encode.append(d["question"])
        if d["question"] and d["gt_answer"]:
            text_map.append((i, "qa_anchor", len(texts_to_encode)))
            texts_to_encode.append(
                f"{d['question']} The answer is {d['gt_answer']}"
            )
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

        # Signal 1: Query relevance — cosine(question, best tool query)
        # Capped at 0.10; diminishing returns above cos=0.80 to discourage
        # lazy copy-paste of the question as the search query.
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

        # Signal 2: Retrieval quality — cosine(question+answer, best KG fact)
        # Primary semantic signal: did the KG return facts relevant to the
        # correct answer? Weighted higher than query relevance.
        qa_emb = emb_lookup.get((i, "qa_anchor"))
        if qa_emb is not None and d["tool_responses"]:
            sims = [
                float(np.dot(qa_emb, emb_lookup[(i, f"response_{j}")]))
                for j in range(len(d["tool_responses"]))
                if (i, f"response_{j}") in emb_lookup
            ]
            if sims:
                score += min(0.15, max(0.0, (max(sims) - 0.55) * 0.33))

        # Signal 3: Grounding — post-tool reasoning references KG entities
        # Requires >= 3 overlapping medical terms to avoid false positives
        # from common words (e.g. "coronary artery" appearing generically).
        if d["post_tool_thinks"] and d["tool_responses"]:
            tool_tokens = _tokenize_for_grounding(
                " ".join(d["tool_responses"])
            )
            post_tokens = _tokenize_for_grounding(
                " ".join(d["post_tool_thinks"])
            )
            if post_tokens and tool_tokens:
                overlap = len(post_tokens & tool_tokens)
                if overlap >= 3 and overlap / min(len(post_tokens), 15) >= 0.2:
                    score += 0.05

        rewards.append(score)

    return rewards
