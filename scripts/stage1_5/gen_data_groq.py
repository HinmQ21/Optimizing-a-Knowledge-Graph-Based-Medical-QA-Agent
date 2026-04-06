#!/usr/bin/env python3
"""Stage 1.5 data generation via API (Groq / Cerebras).

Teacher: GPT-OSS 120B — interacts with the REAL local KG (same FAISS indices
as Stage 2 GRPO). Zero distribution gap between SFT data and GRPO environment.

Usage:
    # Quick test: 5 samples with verbose per-trace output
    GROQ_API_KEY=gsk_... python gen_data_groq.py --test-samples 5

    # Full run (Groq, GPT-OSS 120B, ~1.5h):
    GROQ_API_KEY=gsk_... python gen_data_groq.py --n-samples 3500

    # Cerebras (fastest, ~15 min):
    CEREBRAS_API_KEY=... python gen_data_groq.py --provider cerebras --n-samples 3500

    # Resume interrupted run:
    python gen_data_groq.py --n-samples 3500 --resume

    # Custom output path + concurrency:
    python gen_data_groq.py --n-samples 3500 --output /path/to/traces.jsonl --concurrency 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup — add baseline/ to sys.path so we can import from scripts/
# ---------------------------------------------------------------------------
_BASELINE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_BASELINE_DIR))

try:
    from openai import AsyncOpenAI
except ImportError:
    print("ERROR: openai package not found. Run: pip install openai")
    sys.exit(1)

try:
    from datasets import load_from_disk
except ImportError:
    print("ERROR: datasets package not found. Run: pip install datasets")
    sys.exit(1)

from scripts.train_rl.data_prep import SYSTEM_PROMPT
from scripts.serve.retrieval_tool import MedicalKnowledgeTool


def _load_dotenv() -> None:
    """Load key=value pairs from .env in the same directory as this script.

    Only sets variables that are not already present in the environment,
    so shell exports always take precedence. Skips blank lines and comments.
    """
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "default_model": "openai/gpt-oss-120b",
        "default_rpm": 30,       # free tier
        # GPT-OSS 120B pricing on Groq ($/1M tokens)
        "price_input":  0.15,
        "price_output": 0.60,
        "price_cached": 0.075,
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "env_key": "CEREBRAS_API_KEY",
        "default_model": "gpt-oss-120b",
        "default_rpm": 60,
        # GPT-OSS 120B pricing on Cerebras ($/1M tokens)
        "price_input":  0.35,
        "price_output": 0.75,
        "price_cached": 0.00,   # Cerebras does not publish cached pricing
    },
}

# Tool schema — must exactly match Stage 2 GRPO (derived from retrieval_tool.py docstring)
TOOLS: list[dict] = [{
    "type": "function",
    "function": {
        "name": "search_medical_knowledge",
        "description": (
            "Search the medical knowledge graph for relevant clinical facts.\n\n"
            "Use this tool to retrieve information about diseases, drugs,\n"
            "symptoms, proteins, pathways, and their relationships."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A medical search query describing what you need to know.",
                }
            },
            "required": ["query"],
        },
    },
}]

# ---------------------------------------------------------------------------
# Teacher system prompts (used ONLY during generation — SYSTEM_PROMPT is
# restored before saving so SFT data always has the Stage 2 prompt).
# More explicit than SYSTEM_PROMPT to enforce <think> compliance.
# ---------------------------------------------------------------------------

_TEACHER_BASE = (
    "You are generating TRAINING DATA for a medical student AI. "
    "A student model will learn to reason and use tools by imitating your output.\n\n"
    "You have access to search_medical_knowledge — a tool that queries PrimeKG, "
    "a medical knowledge graph with 54,000 entities (diseases, drugs, genes, "
    "symptoms, pathways) and 81,000+ relationships.\n\n"
    "YOUR TASK: Demonstrate step-by-step medical reasoning. When you are unsure "
    "about a mechanism, drug target, disease relationship, or clinical fact, "
    "call search_medical_knowledge to retrieve evidence from the KG. "
    "Then integrate the retrieved facts into your reasoning before answering.\n\n"
    "STRICT OUTPUT FORMAT:\n"
    "1. ALWAYS start with <think> — write your initial clinical reasoning\n"
    "2. If the question involves drug mechanisms, disease pathways, or clinical "
    "relationships that benefit from verification, call search_medical_knowledge\n"
    "3. After receiving KG results, write ANOTHER <think> block that explicitly "
    "references and integrates the retrieved facts\n"
    "4. ALWAYS end with <answer>LETTER</answer>\n\n"
    "WHEN TO SEARCH:\n"
    "- Drug mechanism / target (e.g. \"haloperidol D2 receptor mechanism\")\n"
    "- Disease-symptom relationships (e.g. \"akathisia extrapyramidal symptoms\")\n"
    "- Treatment options (e.g. \"atypical antipsychotic 5HT2A lower EPS\")\n"
    "- Pathophysiology (e.g. \"nigrostriatal pathway dopamine blockade\")\n"
    "WHEN NOT TO SEARCH:\n"
    "- Simple factual recall (e.g. normal lab values, anatomy basics)\n"
    "- The question is about study design or statistics\n\n"
    "TOOL CALL LIMIT: You may call search_medical_knowledge at most 2 times.\n"
    "If you have not found what you need after 2 searches, stop searching and\n"
    "reason with what you have. Do NOT repeat similar queries.\n\n"
    "EXAMPLE WITH TOOL:\n"
    "<think>\n"
    "Patient on haloperidol with restlessness — likely akathisia (EPS).\n"
    "I need to verify the mechanism and what alternative targets reduce EPS.\n"
    "</think>\n"
    "[calls search_medical_knowledge(\"haloperidol D2 receptor akathisia mechanism\")]\n"
    "[KG returns: Haloperidol targets DRD2, atypical antipsychotics use 5-HT2A...]\n"
    "<think>\n"
    "The KG confirms haloperidol blocks D2 in nigrostriatal pathway causing EPS.\n"
    "Atypical antipsychotics with 5-HT2A antagonism have lower EPS risk.\n"
    "Therefore the replacement drug should target 5-HT2A → option C.\n"
    "</think>\n"
    "<answer>C</answer>\n\n"
    "EXAMPLE WITHOUT TOOL:\n"
    "<think>\n"
    "This is a biostatistics question about study design.\n"
    "The scenario describes a cohort study with stratified analysis.\n"
    "No KG lookup needed — this is methodological knowledge.\n"
    "[reasoning...]\n"
    "</think>\n"
    "<answer>C</answer>"
)

_NO_CALL_SUFFIX = (
    "\n\nFor this question, answer from your own medical knowledge — "
    "do NOT call the search tool. "
    "Start with <think>, reason step-by-step, end with <answer>LETTER</answer>."
)

_MULTIHOP_SUFFIX = (
    "\n\nThis is a complex question — search for multiple concepts separately "
    "(e.g. first the disease mechanism, then the drug/treatment). "
    "Write a <think> block BEFORE each search explaining what you need, "
    "and AFTER receiving results explaining what the KG told you."
)

DATA_PATHS = {
    "medqa_train": _BASELINE_DIR / "dataset" / "MedQA",
    "medmcqa_test": _BASELINE_DIR / "dataset" / "MedMCQA_4options_fixed",
}

DEFAULT_OUTPUT = _BASELINE_DIR / "data" / "stage1_5_traces.jsonl"
DEFAULT_DATA_DIR = str(_BASELINE_DIR / "data")

# Stage 2 uses indices 0..2047 of MedQA train — never touch those
# Stage 2 GRPO uses first 2048 of MedQA train (confirmed from lora_grpo_med.log)
GRPO_RESERVED_MEDQA = 4096


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GenConfig:
    provider: str = "groq"
    model: str = "openai/gpt-oss-120b"
    api_key: str = ""
    base_url: str = "https://api.groq.com/openai/v1"
    n_samples: int = 3500
    no_call_ratio: float = 0.15   # fraction forced to no-call
    multihop_ratio: float = 0.08  # fraction hinted toward multi-hop
    max_tool_iter: int = 3
    max_retries: int = 2        # max retries per API call on tool_use_failed
    temperature: float = 0.7
    max_tokens: int = 2048
    concurrency: int = 3
    rpm: int = 30
    output: Path = DEFAULT_OUTPUT
    data_dir: str = DEFAULT_DATA_DIR
    resume: bool = False
    test_samples: int = 0         # if >0, test mode: verbose, no save
    seed: int = 42


# ---------------------------------------------------------------------------
# Rate limiter — sliding window, respects per-minute RPM limit
# ---------------------------------------------------------------------------

class RateLimiter:
    """Sliding-window rate limiter. Thread-safe for asyncio."""

    def __init__(self, rpm: int) -> None:
        self._rpm = rpm
        self._times: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            # Evict calls older than 60s
            while self._times and now - self._times[0] > 60.0:
                self._times.popleft()
            if len(self._times) >= self._rpm:
                wait = 60.0 - (now - self._times[0]) + 0.1
                logger.debug("Rate limit: sleeping %.1fs", wait)
                await asyncio.sleep(wait)
                now = time.monotonic()
                while self._times and now - self._times[0] > 60.0:
                    self._times.popleft()
            self._times.append(time.monotonic())


class CostTracker:
    """Accumulate token usage and compute dollar cost across all API calls.

    Groq GPT-OSS 120B:  $0.15/1M input | $0.60/1M output | $0.075/1M cached
    Cerebras GPT-OSS:   $0.35/1M input | $0.75/1M output
    """

    def __init__(self, price_input: float, price_output: float, price_cached: float) -> None:
        self.price_input = price_input    # $/1M input tokens
        self.price_output = price_output  # $/1M output tokens
        self.price_cached = price_cached  # $/1M cached input tokens
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.api_calls = 0

    def add(self, usage: Any) -> None:
        """Add token counts from one API response's usage object."""
        if usage is None:
            return
        self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
        # Groq returns cached tokens under prompt_tokens_details.cached_tokens
        details = getattr(usage, "prompt_tokens_details", None)
        self.cached_tokens += getattr(details, "cached_tokens", 0) or 0
        self.api_calls += 1

    @property
    def cost_usd(self) -> float:
        """Total cost in USD based on accumulated token counts."""
        # Cached tokens are billed at cached rate (not the full input rate)
        billable_input = max(0, self.input_tokens - self.cached_tokens)
        return (
            billable_input      * self.price_input  / 1_000_000
            + self.cached_tokens * self.price_cached / 1_000_000
            + self.output_tokens * self.price_output / 1_000_000
        )

    def summary(self) -> str:
        """One-line cost summary for logging."""
        return (
            f"cost=${self.cost_usd:.4f} | "
            f"in={self.input_tokens:,} out={self.output_tokens:,} "
            f"cached={self.cached_tokens:,} tok | "
            f"api_calls={self.api_calls}"
        )

    def projection(self, done: int, total: int) -> str:
        """Project total cost assuming linear token usage."""
        if done == 0:
            return "projection=n/a"
        projected = self.cost_usd / done * total
        return f"projected_total=${projected:.3f} ({done}/{total} done)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_question(question: str, options: dict) -> str:
    """Format question exactly as Stage 2 data_prep.py does."""
    opt_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"{question}\n\nOptions:\n{opt_text}"


def kg_retrieve(kg_tool: MedicalKnowledgeTool, query: str) -> str:
    """Execute real KG retrieval — same code path as Stage 2 GRPO."""
    results = kg_tool.retrieve(query, top_k=5)
    if not results:
        return "No relevant knowledge found."
    return "\n".join(f"- {r}" for r in results)


def extract_answer(text: str) -> str | None:
    """Extract answer letter from <answer> tag. Tolerant of minor formatting."""
    m = re.search(r"<answer>\s*([A-Ea-e])\b", text)
    return m.group(1).upper() if m else None


def filter_trace(trace: dict) -> tuple[bool, str]:
    """Run all 6 quality gates. Returns (passed, failure_reason)."""
    messages = trace["messages"]
    expected = trace["answer_idx"].upper()

    # Collect all assistant text
    all_assistant = " ".join(
        m.get("content", "") or ""
        for m in messages if m["role"] == "assistant"
    )

    # Gate 1: Has <think>...</think>
    if not re.search(r"<think>.*?</think>", all_assistant, re.DOTALL):
        return False, "missing_think"

    # Gate 2: Has <answer> with valid MCQ letter
    predicted = extract_answer(all_assistant)
    if predicted is None:
        return False, "missing_answer"

    # Gate 3: Answer is correct
    if predicted != expected:
        return False, f"wrong_answer:{predicted}≠{expected}"

    # Gate 4: Tool call JSON is well-formed
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
            except (json.JSONDecodeError, KeyError, AssertionError, TypeError):
                return False, "invalid_tool_call_json"

    # Gate 5: Tool call / response count matches
    n_calls = sum(1 for m in messages if m.get("tool_calls"))
    n_resps = sum(1 for m in messages if m["role"] == "tool")
    if n_calls != n_resps:
        return False, f"call_response_mismatch:{n_calls}calls/{n_resps}resps"

    # Gate 6: Post-retrieval reasoning references retrieved content
    if n_calls > 0:
        think_blocks = re.findall(r"<think>(.*?)</think>", all_assistant, re.DOTALL)
        tool_contents = [m["content"] for m in messages if m["role"] == "tool"]
        if len(think_blocks) >= 2 and tool_contents:
            post_think = think_blocks[-1].lower()
            all_tool_text = " ".join(tool_contents).lower()
            # Check any non-trivial word from tool response appears in post-retrieval think
            meaningful = [
                w for w in all_tool_text.split()
                if len(w) > 5 and w.isalpha()
            ]
            if meaningful and not any(w in post_think for w in meaningful[:20]):
                return False, "no_retrieval_integration"

    return True, "ok"


def _synthesize_think_for_tool_call(tool_calls: list[dict]) -> str:
    """Generate a <think> block from tool call queries.

    When the API returns assistant messages with tool_calls but empty content,
    the student would learn to call tools without reasoning first. This
    synthesizes a minimal <think> block that demonstrates the "reason before
    searching" pattern Stage 2 GRPO expects.
    """
    queries = []
    for tc in tool_calls:
        try:
            args = tc["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            queries.append(args.get("query", ""))
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    if not queries:
        return "<think>\nI need to search the medical knowledge graph for more information.\n</think>"

    if len(queries) == 1:
        return f"<think>\nI need to look up information about {queries[0]}.\n</think>"

    lines = "\n".join(f"- {q}" for q in queries)
    return f"<think>\nI need to search for multiple concepts:\n{lines}\n</think>"


def normalize_messages_for_sft(messages: list[dict]) -> list[dict]:
    """Clean up API-specific fields before saving as SFT data.

    - Ensures content is str (not None)
    - If assistant has tool_calls but empty content, synthesizes a <think>
      block from the query so student learns "reason → search" pattern
    - Normalizes tool_calls to minimal schema TRL expects
    - Keeps tool_call_id (TRL's chat template may use it)
    """
    clean = []
    for msg in messages:
        m: dict[str, Any] = {"role": msg["role"]}

        content = msg.get("content") or ""

        # Fix empty assistant content before tool calls — Stage 2 GRPO
        # expects <think> reasoning before every tool call
        if msg["role"] == "assistant" and msg.get("tool_calls") and not content.strip():
            content = _synthesize_think_for_tool_call(msg["tool_calls"])

        m["content"] = content

        if msg.get("tool_calls"):
            m["tool_calls"] = [
                {
                    "id": tc.get("id", f"call_{i}"),
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        # Ensure arguments is a JSON string (not dict)
                        "arguments": (
                            tc["function"]["arguments"]
                            if isinstance(tc["function"]["arguments"], str)
                            else json.dumps(tc["function"]["arguments"])
                        ),
                    },
                }
                for i, tc in enumerate(msg["tool_calls"])
            ]

        if msg["role"] == "tool" and msg.get("tool_call_id"):
            m["tool_call_id"] = msg["tool_call_id"]

        clean.append(m)
    return clean


# ---------------------------------------------------------------------------
# Async generation — single trace
# ---------------------------------------------------------------------------

async def generate_trace(
    client: AsyncOpenAI,
    kg_tool: MedicalKnowledgeTool,
    item: dict,
    cfg: GenConfig,
    rate_limiter: RateLimiter,
    trace_type: str = "auto",   # "auto" | "no_call" | "multihop"
    cost_tracker: CostTracker | None = None,
) -> dict | None:
    """Generate one trace via teacher API + real KG tool execution.

    trace_type:
        "auto"     — tool_choice="auto", teacher decides (most traces)
        "no_call"  — tool_choice="none", forced parametric reasoning
        "multihop" — tool_choice="auto" with multi-hop hint in system

    Returns None on unrecoverable error.
    """
    question = item["question"]
    options = item["options"]
    answer_idx = item["answer_idx"]
    user_content = format_question(question, options)

    # Build teacher system prompt (more explicit than SYSTEM_PROMPT to enforce
    # <think> compliance). SYSTEM_PROMPT is restored before saving.
    if trace_type == "no_call":
        sys_content = _TEACHER_BASE + _NO_CALL_SUFFIX
        tool_choice = "none"
    elif trace_type == "multihop":
        sys_content = _TEACHER_BASE + _MULTIHOP_SUFFIX
        tool_choice = "auto"
    else:
        sys_content = _TEACHER_BASE
        tool_choice = "auto"

    # API messages (with tool_call_id for proper back-and-forth)
    api_messages: list[dict] = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_content},
    ]

    # max_tool_iter iterations of tool calls + 1 final response without tools
    for iteration in range(cfg.max_tool_iter + 1):
        # On the last iteration, don't offer tools — force final think+answer
        is_last_iter = iteration == cfg.max_tool_iter
        # Respect rate limit
        await rate_limiter.acquire()

        try:
            # Build API call kwargs
            api_kwargs: dict[str, Any] = dict(
                model=cfg.model,
                messages=api_messages,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
            )
            # Only offer tools when: not no_call AND not the forced final iter
            if trace_type != "no_call" and not is_last_iter:
                api_kwargs["tools"] = TOOLS
                api_kwargs["tool_choice"] = tool_choice

            response = await client.chat.completions.create(**api_kwargs)
        except Exception as exc:
            exc_str = str(exc)

            if "429" in exc_str or "rate" in exc_str.lower():
                logger.debug("Rate limited, sleeping 15s")
                await asyncio.sleep(15)
                continue

            # Groq "tool_use_failed" — model generated a tool call when
            # tools were not offered, or generated a corrupted tool name.
            # Retry up to 2 times with lower temperature to get text output.
            if "tool_use_failed" in exc_str or "tool_call" in exc_str.lower():
                for retry in range(cfg.max_retries):
                    try:
                        await rate_limiter.acquire()
                        fallback = await client.chat.completions.create(
                            model=cfg.model,
                            messages=api_messages,
                            max_tokens=cfg.max_tokens,
                            temperature=max(0.3, cfg.temperature - 0.2 * (retry + 1)),
                        )
                        if cost_tracker is not None:
                            cost_tracker.add(fallback.usage)
                        fb_msg = fallback.choices[0].message
                        fb_content = fb_msg.content or ""
                        api_messages.append({"role": "assistant", "content": fb_content})
                        break  # Got text response
                    except Exception:
                        continue
                else:
                    logger.warning("API error (iter %d), retries exhausted: %s", iteration, exc_str[:120])
                    return None
                break  # Exit main loop — we got a fallback response

            logger.warning("API error (iter %d): %s", iteration, exc_str[:200])
            return None

        if cost_tracker is not None:
            cost_tracker.add(response.usage)

        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""

        # Build assistant message dict for API + saved data
        assistant_entry: dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }

        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        api_messages.append(assistant_entry)

        # No tool calls → generation complete
        if not msg.tool_calls:
            break

        # Execute each tool call against REAL KG
        for tc in msg.tool_calls:
            if tc.function.name != "search_medical_knowledge":
                logger.warning("Unknown tool call: %s", tc.function.name)
                continue

            try:
                args = json.loads(tc.function.arguments)
                query = args.get("query", tc.function.arguments)
            except json.JSONDecodeError:
                query = tc.function.arguments

            # *** REAL KG RETRIEVAL — same indices as Stage 2 GRPO ***
            tool_result = kg_retrieve(kg_tool, query)

            api_messages.append({
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tc.id,
            })

    # Restore original system prompt for saved SFT data
    api_messages[0]["content"] = SYSTEM_PROMPT

    n_tool_calls = sum(1 for m in api_messages if m.get("tool_calls"))

    return {
        "messages": normalize_messages_for_sft(api_messages),
        "answer_idx": answer_idx,
        "question": question,
        "num_tool_calls": n_tool_calls,
        "source": item.get("_source", "unknown"),
        "question_id": item.get("_id", -1),
        "trace_type": trace_type,
    }


# ---------------------------------------------------------------------------
# Batch generation with progress
# ---------------------------------------------------------------------------

async def generate_batch(
    items: list[dict],
    cfg: GenConfig,
    kg_tool: MedicalKnowledgeTool,
    already_done: set[tuple],
) -> list[dict]:
    """Generate traces for all items with concurrency + rate limiting."""
    prov = PROVIDERS[cfg.provider]
    client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    rate_limiter = RateLimiter(cfg.rpm)
    sem = asyncio.Semaphore(cfg.concurrency)
    cost_tracker = CostTracker(
        price_input=prov["price_input"],
        price_output=prov["price_output"],
        price_cached=prov["price_cached"],
    )

    import random
    rng = random.Random(cfg.seed)

    # Assign trace types
    n = len(items)
    n_no_call = int(n * cfg.no_call_ratio)
    n_multihop = int(n * cfg.multihop_ratio)
    types = ["no_call"] * n_no_call + ["multihop"] * n_multihop + ["auto"] * (n - n_no_call - n_multihop)
    rng.shuffle(types)
    items_with_type = list(zip(items, types))

    results: list[dict] = []
    stats = {"total": 0, "skipped": 0, "generated": 0, "failed_api": 0}
    gate_fails: dict[str, int] = {}
    start_time = time.monotonic()

    pbar = tqdm(
        total=cfg.n_samples,
        desc="Generating",
        unit="trace",
        dynamic_ncols=True,
        bar_format=(
            "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"
            " [{elapsed}<{remaining}, {rate_fmt}]{postfix}"
        ),
    )

    def _update_bar(passed: bool) -> None:
        """Refresh tqdm bar after each completed attempt."""
        attempted = stats["total"]
        pass_rate = stats["generated"] / max(attempted, 1)
        elapsed = time.monotonic() - start_time

        # Time estimate: based on generated traces so far
        if stats["generated"] > 0:
            sec_per_trace = elapsed / stats["generated"]
            remaining_traces = cfg.n_samples - stats["generated"]
            eta_min = sec_per_trace * remaining_traces / 60
            eta_str = f"{eta_min:.0f}m" if eta_min < 60 else f"{eta_min/60:.1f}h"
        else:
            eta_str = "?"

        top_fail = (
            max(gate_fails, key=gate_fails.get) if gate_fails else "-"
        )

        pbar.set_postfix(
            pass_rate=f"{pass_rate:.0%}",
            attempt=attempted,
            cost=f"${cost_tracker.cost_usd:.3f}",
            proj=f"${cost_tracker.cost_usd / max(stats['generated'], 1) * cfg.n_samples:.2f}",
            eta=eta_str,
            top_fail=top_fail,
            refresh=False,
        )
        if passed:
            pbar.update(1)  # advance bar only on pass
        else:
            pbar.refresh()  # redraw postfix without advancing

    async def process_one(item: dict, trace_type: str, idx: int) -> None:
        key = (item["_source"], item["_id"])
        if key in already_done:
            stats["skipped"] += 1
            return

        async with sem:
            trace = await generate_trace(
                client, kg_tool, item, cfg, rate_limiter, trace_type, cost_tracker
            )

        stats["total"] += 1

        if trace is None:
            stats["failed_api"] += 1
            _update_bar(passed=False)
            return

        passed, reason = filter_trace(trace)

        if passed:
            results.append(trace)
            stats["generated"] += 1
        else:
            gate_fails[reason] = gate_fails.get(reason, 0) + 1

        _update_bar(passed=passed)

    tasks = [
        asyncio.create_task(process_one(item, trace_type, idx))
        for idx, (item, trace_type) in enumerate(items_with_type)
    ]
    await asyncio.gather(*tasks)

    pbar.close()

    elapsed_total = time.monotonic() - start_time
    tqdm.write(
        f"\nBatch complete in {elapsed_total/60:.1f}m — "
        f"{stats['generated']} generated / {stats['total']} attempted / "
        f"{stats['skipped']} skipped | API errors: {stats['failed_api']}"
    )
    tqdm.write(f"Gate failures: {gate_fails}")
    tqdm.write(f"FINAL COST — {cost_tracker.summary()}")
    return results


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_source_questions(cfg: GenConfig) -> list[dict]:
    """Load and merge MedQA surplus + MedMCQA test into flat list of dicts."""
    items: list[dict] = []

    # MedQA train surplus (indices GRPO_RESERVED_MEDQA .. end)
    medqa_path = DATA_PATHS["medqa_train"]
    if medqa_path.exists():
        ds = load_from_disk(str(medqa_path))["train"]
        for idx in range(GRPO_RESERVED_MEDQA, len(ds)):
            row = ds[idx]
            items.append({
                "question": row["question"],
                "options": row["options"],
                "answer_idx": row["answer_idx"],
                "metamap_phrases": row.get("metamap_phrases", []),
                "_source": "medqa_train",
                "_id": idx,
            })
        logger.info("Loaded %d MedQA surplus questions (idx %d..%d)",
                    len(items), GRPO_RESERVED_MEDQA, len(ds) - 1)
    else:
        logger.warning("MedQA path not found: %s", medqa_path)

    # NOTE: MedMCQA_4options_fixed is EXCLUDED — answer_idx is "D" for 100%
    # of questions (broken preprocessing), which causes severe D-bias.
    # MedQA surplus alone (8,130 questions) is sufficient for our 3,500 target.

    return items


def load_already_done(output_path: Path) -> set[tuple]:
    """Load (source, id) keys from existing output JSONL for resume."""
    done = set()
    if not output_path.exists():
        return done
    with open(output_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add((rec["source"], rec["question_id"]))
            except (json.JSONDecodeError, KeyError):
                pass
    logger.info("Resume: found %d already-done traces in %s", len(done), output_path)
    return done


# ---------------------------------------------------------------------------
# Test sample mode — verbose per-trace output
# ---------------------------------------------------------------------------

def _fmt_messages(messages: list[dict]) -> str:
    """Pretty-print messages for test mode."""
    lines = []
    for m in messages:
        role = m["role"].upper()
        content = m.get("content", "") or ""
        tcs = m.get("tool_calls", [])
        if tcs:
            tc_str = ", ".join(
                f'search("{json.loads(tc["function"]["arguments"]).get("query","?")}")'
                if isinstance(tc["function"]["arguments"], str)
                else f'search({tc["function"]["arguments"].get("query","?")})'
                for tc in tcs
            )
            lines.append(f"  [{role}] {content[:100]} → TOOL_CALL: {tc_str}")
        elif role == "TOOL":
            preview = content[:120].replace("\n", " ")
            lines.append(f"  [{role}] {preview}...")
        else:
            preview = content[:200].replace("\n", " ")
            lines.append(f"  [{role}] {preview}")
    return "\n".join(lines)


async def run_test_samples(cfg: GenConfig) -> None:
    """Generate --test-samples traces and print detailed per-trace output."""
    import random
    rng = random.Random(cfg.seed)

    logger.info("Loading KG tool from %s ...", cfg.data_dir)
    kg_tool = MedicalKnowledgeTool.load(cfg.data_dir)
    logger.info("KG loaded. Loading questions...")

    all_items = load_source_questions(cfg)
    rng.shuffle(all_items)
    items = all_items[: cfg.test_samples * 3]  # oversample to account for failures

    prov = PROVIDERS[cfg.provider]
    client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    rate_limiter = RateLimiter(cfg.rpm)
    cost_tracker = CostTracker(
        price_input=prov["price_input"],
        price_output=prov["price_output"],
        price_cached=prov["price_cached"],
    )

    types = ["no_call", "multihop", "auto", "auto", "auto"]  # mix for test
    gate_stats: dict[str, int] = {}
    generated = 0
    attempted = 0

    print("\n" + "=" * 70)
    print(f"TEST MODE — {cfg.test_samples} samples | model: {cfg.model} | provider: {cfg.provider}")
    print(f"Pricing: ${prov['price_input']}/1M in | ${prov['price_output']}/1M out | "
          f"${prov['price_cached']}/1M cached")
    print("=" * 70)

    for i, item in enumerate(items):
        if generated >= cfg.test_samples:
            break

        trace_type = types[i % len(types)]
        attempted += 1

        print(f"\n[{attempted}] Q({item['_source']}#{item['_id']}) type={trace_type}")
        print(f"  Q: {item['question'][:120]}...")

        # Snapshot cost before this trace to compute per-trace delta
        cost_before = cost_tracker.cost_usd
        in_before = cost_tracker.input_tokens
        out_before = cost_tracker.output_tokens

        trace = await generate_trace(
            client, kg_tool, item, cfg, rate_limiter, trace_type, cost_tracker
        )

        if trace is None:
            print("  → API ERROR (None returned)")
            continue

        # Per-trace cost delta
        delta_in = cost_tracker.input_tokens - in_before
        delta_out = cost_tracker.output_tokens - out_before
        delta_cost = cost_tracker.cost_usd - cost_before

        passed, reason = filter_trace(trace)
        n_calls = trace["num_tool_calls"]
        predicted = extract_answer(
            " ".join(m.get("content", "") or "" for m in trace["messages"] if m["role"] == "assistant")
        )

        print(f"  Answer: expected={item['answer_idx']} predicted={predicted} | "
              f"tool_calls={n_calls} | filter={'✓ PASS' if passed else f'✗ FAIL ({reason})'}")
        print(f"  Tokens: in={delta_in:,} out={delta_out:,} | cost=${delta_cost:.5f}")
        print(_fmt_messages(trace["messages"]))

        gate_stats[reason] = gate_stats.get(reason, 0) + 1

        if passed:
            generated += 1

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print(f"  attempted={attempted} | passed={generated} | "
          f"pass_rate={generated/max(attempted,1):.1%}")
    print(f"  gate_stats: {gate_stats}")
    print(f"  {cost_tracker.summary()}")
    if attempted > 0:
        per_trace = cost_tracker.cost_usd / attempted
        print(f"  avg_cost_per_trace=${per_trace:.5f} | "
              f"est_3500_traces=${per_trace * 3500:.2f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Full generation run
# ---------------------------------------------------------------------------

async def run_full_generation(cfg: GenConfig) -> None:
    """Generate full dataset with streaming JSONL output."""
    logger.info("Loading KG tool from %s ...", cfg.data_dir)
    kg_tool = MedicalKnowledgeTool.load(cfg.data_dir)
    logger.info("KG loaded.")

    all_items = load_source_questions(cfg)

    import random
    rng = random.Random(cfg.seed)
    rng.shuffle(all_items)

    # Take enough items to expect cfg.n_samples after filtering (~65% pass rate)
    # Oversample by ~4.5x to account for filter failures (~25% pass rate)
    target_raw = min(int(cfg.n_samples * 4.5), len(all_items))
    items = all_items[:target_raw]
    logger.info("Sampling %d raw questions (target %d final traces)", target_raw, cfg.n_samples)

    already_done = set()
    if cfg.resume:
        already_done = load_already_done(cfg.output)

    cfg.output.parent.mkdir(parents=True, exist_ok=True)

    results = await generate_batch(items, cfg, kg_tool, already_done)

    # Trim to exactly n_samples if we overshot
    if len(results) > cfg.n_samples:
        results = results[: cfg.n_samples]

    # Write to JSONL
    mode = "a" if cfg.resume else "w"
    written = 0
    with open(cfg.output, mode) as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    logger.info("Wrote %d traces to %s", written, cfg.output)

    # Final stats
    n_calls_dist: dict[int, int] = {}
    for rec in results:
        n = rec["num_tool_calls"]
        n_calls_dist[n] = n_calls_dist.get(n, 0) + 1

    logger.info("tool_call distribution: %s", dict(sorted(n_calls_dist.items())))
    logger.info(
        "  no-call: %.1f%% | 1-call: %.1f%% | 2+-call: %.1f%%",
        n_calls_dist.get(0, 0) / max(written, 1) * 100,
        n_calls_dist.get(1, 0) / max(written, 1) * 100,
        sum(v for k, v in n_calls_dist.items() if k >= 2) / max(written, 1) * 100,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 1.5 data generation via Groq/Cerebras API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--provider", choices=list(PROVIDERS), default="groq",
                   help="API provider")
    p.add_argument("--model", default=None,
                   help="Model name (default: provider default)")
    p.add_argument("--api-key", default=None,
                   help="API key (default: read from env var)")
    p.add_argument("--n-samples", type=int, default=3500,
                   help="Target number of final (filtered) traces")
    p.add_argument("--no-call-ratio", type=float, default=0.30,
                   help="Fraction of questions forced to no-tool-call mode")
    p.add_argument("--multihop-ratio", type=float, default=0.08,
                   help="Fraction of questions hinted toward multi-hop retrieval")
    p.add_argument("--concurrency", type=int, default=3,
                   help="Max concurrent API requests (1 = safe for free tier)")
    p.add_argument("--rpm", type=int, default=None,
                   help="Requests-per-minute limit (default: provider default)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=3072)
    p.add_argument("--max-tool-iter", type=int, default=3,
                   help="Max tool call iterations per trace")
    p.add_argument("--max-retries", type=int, default=1,
                   help="Max retries per API call on tool_use_failed error")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="Output JSONL file path")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                   help="Path to FAISS indices + medical_hg.json")
    p.add_argument("--resume", action="store_true",
                   help="Resume from existing output file (skip already-done)")
    p.add_argument("--test-samples", type=int, default=0,
                   help="Test mode: generate N samples with verbose output, no save")
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    prov = PROVIDERS[args.provider]

    # Load .env from script directory (shell exports take precedence)
    _load_dotenv()

    # Resolve API key: --api-key > shell env > .env
    api_key = args.api_key or os.environ.get(prov["env_key"], "")
    if not api_key:
        parser.error(
            f"API key required. Set ${prov['env_key']} in .env, shell env, or pass --api-key."
        )

    cfg = GenConfig(
        provider=args.provider,
        model=args.model or prov["default_model"],
        api_key=api_key,
        base_url=prov["base_url"],
        n_samples=args.n_samples,
        no_call_ratio=args.no_call_ratio,
        multihop_ratio=args.multihop_ratio,
        max_tool_iter=args.max_tool_iter,
        max_retries=args.max_retries,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        rpm=args.rpm or prov["default_rpm"],
        output=args.output,
        data_dir=args.data_dir,
        resume=args.resume,
        test_samples=args.test_samples,
        seed=args.seed,
    )

    logger.info("Provider: %s | Model: %s | RPM: %d | Concurrency: %d",
                cfg.provider, cfg.model, cfg.rpm, cfg.concurrency)

    if cfg.test_samples > 0:
        asyncio.run(run_test_samples(cfg))
    else:
        asyncio.run(run_full_generation(cfg))


if __name__ == "__main__":
    main()
