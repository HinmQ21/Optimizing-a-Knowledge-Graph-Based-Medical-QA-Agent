"""Model-family adapter: unified interface for Qwen2.5 and Llama-3.2-3B-Instruct.

Both families share the same high-level architecture:
  <think>...</think>  reasoning blocks
  <answer>...</answer> answer tags
  Tool calls in family-specific format
  Tool responses in family-specific role

Qwen  → <tool_call>{"name":…,"arguments":{…}}</tool_call>   role="user" (wrapped in <tool_response>)
Llama → {"name":…,"parameters":{…}}<|eot_id|>  (plain JSON, no python_tag prefix)
        role="tool" (template renders as <|start_header_id|>ipython<|end_header_id|>)
        Tool definition injected into user message via tools= param in apply_chat_template.

Usage (auto-detect family from model path):
    from scripts.utils.model_adapter import detect_family, normalize_special_tokens, ...
    family = detect_family(args.model_path)
    normalize_special_tokens(tokenizer, model, family, padding_side="left")
"""

import re
from typing import Literal

ModelFamily = Literal["qwen", "llama"]

# ---------------------------------------------------------------------------
# Tool-call TRL response schemas
# ---------------------------------------------------------------------------
# Qwen: <tool_call>{"name":…,"arguments":{…}}</tool_call><|im_end|>
QWEN_TOOL_SCHEMA: dict = {
    "x-regex": (
        r"^(?:<think>\n?(?:(?P<reasoning_content>.*?\S.*?)\n?|[\s]*)</think>\s*)?"
        r"(?P<content>.*?)(?:\n(?=<tool_call>))?(?=(?:<tool_call>|<\|im_end\|>|$))"
        r"(?P<tool_calls>(?:<tool_call>.+?</tool_call>\s*)+)?\s*(?:<\|im_end\|>|$)"
    ),
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string"},
        "reasoning_content": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "x-regex-iterator": r"<tool_call>\s*(.+?)\s*</tool_call>",
            "items": {
                "x-parser": "json",
                "x-parser-args": {"transform": "{type: 'function', function: @}"},
                "type": "object",
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {"type": "object", "additionalProperties": {}},
                        },
                    },
                },
            },
        },
    },
}

# Llama: {"name":…,"parameters":{…}}<|eot_id|>   (plain JSON, no python_tag prefix)
# Llama's native chat template emits `parameters`, but TRL's _validate_tool_calls
# and _tool_call_loop strictly require `arguments`. The JMESPath transform below
# renames parameters → arguments so the parsed structure is TRL-compatible.
LLAMA_TOOL_SCHEMA: dict = {
    "x-regex": (
        r"^(?:<think>\n?(?:(?P<reasoning_content>.*?\S.*?)\n?|[\s]*)</think>\s*)?"
        r"(?P<content>[^{<]*)"
        r"(?P<tool_calls>(?:\{(?:[^{}]|\{[^{}]*\})*\}\s*(?:<\|eot_id\|>|<\|eom_id\|>)\s*)+)?"
        r"\s*(?:<\|eot_id\|>|$)"
    ),
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string"},
        "reasoning_content": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "x-regex-iterator": r"(\{(?:[^{}]|\{[^{}]*\})*\})\s*(?:<\|eot_id\|>|<\|eom_id\|>)",
            "items": {
                "x-parser": "json",
                "x-parser-args": {
                    "transform": "{type: 'function', function: {name: name, arguments: parameters}}"
                },
                "type": "object",
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {"type": "object", "additionalProperties": {}},
                        },
                    },
                },
            },
        },
    },
}

# ---------------------------------------------------------------------------
# Family detection
# ---------------------------------------------------------------------------

def detect_family(model_path: str) -> ModelFamily:
    """Auto-detect model family from path string, falling back to config inspection."""
    path_lower = str(model_path).lower()
    if "llama" in path_lower:
        return "llama"
    if "qwen" in path_lower:
        return "qwen"
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = getattr(cfg, "model_type", "").lower()
        arch = (getattr(cfg, "architectures", None) or [""])[0].lower()
        if "llama" in model_type or "llama" in arch:
            return "llama"
        if "qwen" in model_type or "qwen" in arch:
            return "qwen"
    except Exception:
        pass
    return "qwen"  # safe default


# ---------------------------------------------------------------------------
# Special token normalisation
# ---------------------------------------------------------------------------

def normalize_special_tokens(
    tokenizer,
    model,
    family: ModelFamily,
    padding_side: str = "left",
) -> None:
    """Set EOS / PAD tokens and model config for the given family.

    Args:
        padding_side: "left" for generation/GRPO, "right" for SFT training.
    """
    if family == "llama":
        _normalize_llama(tokenizer, model)
    else:
        _normalize_qwen(tokenizer, model)

    tokenizer.padding_side = padding_side
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id


def normalize_tokenizer_only(
    tokenizer,
    family: ModelFamily,
    padding_side: str = "left",
) -> None:
    """Tokenizer-only variant of normalize_special_tokens.

    Use when no HF model object is available (e.g., when generation is performed
    by vLLM). Skips all model.config / generation_config side effects.
    """
    if family == "llama":
        _normalize_llama(tokenizer, None)
    else:
        _normalize_qwen(tokenizer, None)
    tokenizer.padding_side = padding_side


def _normalize_qwen(tokenizer, model) -> None:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "tokenizer.chat_template is missing. "
            "Ensure the model path contains a valid Qwen2.5 tokenizer."
        )
    vocab = tokenizer.get_vocab()
    if tokenizer.eos_token in {None, "", "<EOS_TOKEN>"}:
        if "<|im_end|>" in vocab:
            tokenizer.eos_token = "<|im_end|>"
        else:
            raise ValueError(
                f"Cannot resolve eos_token for Qwen: {tokenizer.eos_token!r}"
            )
    if tokenizer.pad_token in {None, "", "<PAD_TOKEN>", "<EOS_TOKEN>"}:
        tokenizer.pad_token = tokenizer.eos_token


def _normalize_llama(tokenizer, model) -> None:
    vocab = tokenizer.get_vocab()
    # Llama-3.2-Instruct EOS is <|eot_id|> (end-of-turn), not <|end_of_text|>
    if "<|eot_id|>" in vocab:
        tokenizer.eos_token = "<|eot_id|>"
    elif tokenizer.eos_token in {None, ""}:
        raise ValueError("Cannot resolve eos_token for Llama: <|eot_id|> not in vocab.")

    # Pad token: prefer <|finetune_right_pad_id|>, else fall back to EOS
    if tokenizer.pad_token_id is None:
        if "<|finetune_right_pad_id|>" in vocab:
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        else:
            tokenizer.pad_token = tokenizer.eos_token


# ---------------------------------------------------------------------------
# Loss-masking: find assistant token spans
# ---------------------------------------------------------------------------

def find_assistant_spans(
    token_ids: list[int],
    tokenizer,
    family: ModelFamily,
) -> list[tuple[int, int]]:
    """Return (start, end) token index spans for assistant content only.

    Returned spans cover the assistant's generated tokens including the
    closing EOS/turn-end token so the model learns to stop.
    System, user, and tool-response tokens are excluded (masked).
    """
    if family == "llama":
        return _spans_llama(token_ids, tokenizer)
    return _spans_qwen(token_ids, tokenizer)


def _spans_qwen(token_ids: list[int], tokenizer) -> list[tuple[int, int]]:
    """Qwen2.5 ChatML: <|im_start|>assistant\\n ... <|im_end|>"""
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    asst_marker = tokenizer.encode("assistant\n", add_special_tokens=False)

    spans = []
    i = 0
    while i < len(token_ids):
        if token_ids[i] == im_start:
            marker_end = i + 1 + len(asst_marker)
            if (marker_end <= len(token_ids)
                    and token_ids[i + 1: marker_end] == asst_marker):
                content_start = marker_end
                content_end = content_start
                while content_end < len(token_ids) and token_ids[content_end] != im_end:
                    content_end += 1
                if content_end < len(token_ids):
                    content_end += 1  # include <|im_end|>
                spans.append((content_start, content_end))
                i = content_end
                continue
        i += 1
    return spans


def _spans_llama(token_ids: list[int], tokenizer) -> list[tuple[int, int]]:
    """Llama-3.2: <|start_header_id|>assistant<|end_header_id|>\\n\\n ... <|eot_id|>"""
    start_hdr = tokenizer.convert_tokens_to_ids("<|start_header_id|>")  # 128006
    end_hdr = tokenizer.convert_tokens_to_ids("<|end_header_id|>")      # 128007
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")                  # 128009
    # "assistant" is a single token [78191] in Llama-3.2 vocab
    asst_ids = tokenizer.encode("assistant", add_special_tokens=False)
    # \n\n is a single token [271] in Llama-3.2 vocab
    newline_ids = set(tokenizer.encode("\n\n", add_special_tokens=False)
                      + tokenizer.encode("\n", add_special_tokens=False))

    spans = []
    i = 0
    while i < len(token_ids):
        if token_ids[i] == start_hdr:
            role_end = i + 1 + len(asst_ids)
            if (role_end < len(token_ids)
                    and token_ids[i + 1: role_end] == asst_ids
                    and token_ids[role_end] == end_hdr):
                # skip <|end_header_id|> then any \n tokens
                content_start = role_end + 1
                while (content_start < len(token_ids)
                       and token_ids[content_start] in newline_ids):
                    content_start += 1
                # find closing <|eot_id|>
                content_end = content_start
                while content_end < len(token_ids) and token_ids[content_end] != eot:
                    content_end += 1
                if content_end < len(token_ids):
                    content_end += 1  # include <|eot_id|>
                spans.append((content_start, content_end))
                i = content_end
                continue
        i += 1
    return spans


# ---------------------------------------------------------------------------
# Tool-call regex
# ---------------------------------------------------------------------------

_QWEN_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# Llama native: plain JSON terminated by <|eot_id|> or <|eom_id|>; one brace-nesting level.
_LLAMA_TOOL_RE = re.compile(
    r"(\{(?:[^{}]|\{[^{}]*\})*\})\s*(?:<\|eot_id\|>|<\|eom_id\|>)",
    re.DOTALL,
)


def get_tool_call_regex(family: ModelFamily) -> re.Pattern:
    return _LLAMA_TOOL_RE if family == "llama" else _QWEN_TOOL_RE


# ---------------------------------------------------------------------------
# Output cleanup after generation
# ---------------------------------------------------------------------------

_QWEN_STRIP = {"<|im_start|>", "<|im_end|>", "<|endoftext|>"}
_LLAMA_STRIP = {
    "<|begin_of_text|>", "<|eot_id|>", "<|end_of_text|>",
    "<|start_header_id|>", "<|end_header_id|>", "<|eom_id|>",
}


def strip_generation_artifacts(text: str, family: ModelFamily) -> str:
    """Remove model-specific control tokens from decoded generation output."""
    tokens = _LLAMA_STRIP if family == "llama" else _QWEN_STRIP
    for tok in tokens:
        text = text.replace(tok, "")
    return text.strip()


def get_strip_tokens(family: ModelFamily) -> set[str]:
    """Token set for sft_eval_v2 _decode_generated()."""
    return _LLAMA_STRIP if family == "llama" else _QWEN_STRIP


# ---------------------------------------------------------------------------
# Tool response injection
# ---------------------------------------------------------------------------

def get_tool_response_message(result: str, family: ModelFamily) -> dict:
    """Build the message dict for injecting a tool result into conversation history.

    Qwen:  role="user" with <tool_response> wrapper (existing pipeline)
    Llama: role="tool" — Llama's chat template renders this as ipython header
    """
    if family == "llama":
        return {"role": "tool", "content": result}
    return {"role": "user", "content": f"<tool_response>\n{result}\n</tool_response>"}


# ---------------------------------------------------------------------------
# TRL response schema
# ---------------------------------------------------------------------------

def get_trl_response_schema(family: ModelFamily) -> dict:
    """Return the TRL response schema for the given model family."""
    if family == "llama":
        return LLAMA_TOOL_SCHEMA
    # Qwen: use the canonical qwen3_schema from TRL (identical to QWEN_TOOL_SCHEMA above)
    from trl.chat_template_utils import qwen3_schema
    return qwen3_schema


# ---------------------------------------------------------------------------
# Tool definition for apply_chat_template (Llama only)
# ---------------------------------------------------------------------------

MEDICAL_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search_medical_knowledge",
        "description": (
            "Search the medical knowledge base for relevant clinical information "
            "about diseases, drugs, symptoms, and treatments."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Medical query to search for.",
                }
            },
            "required": ["query"],
        },
    },
}


def get_tools_for_template(family: ModelFamily):
    """Return tools list for apply_chat_template, or None for Qwen.

    Llama's chat template injects the tool definition into the user message when
    tools= is passed. Qwen handles its own tool schema separately.
    """
    return [MEDICAL_TOOL_DEF] if family == "llama" else None


def get_eos_for_generation(family: ModelFamily, tokenizer) -> list[int] | int:
    """Return eos_token_id(s) for model.generate().

    Llama needs both <|eot_id|> (128009, normal turn end) and <|eom_id|>
    (128008, tool-call end) to stop generation at the right boundary.
    Without <|eom_id|> in the list, generation continues past the tool call.
    """
    if family == "llama":
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")  # 128009
        eom = tokenizer.convert_tokens_to_ids("<|eom_id|>")  # 128008
        return [eot, eom]
    return tokenizer.eos_token_id


# ---------------------------------------------------------------------------
# Model loading kwargs
# ---------------------------------------------------------------------------

def get_model_load_kwargs(family: ModelFamily) -> dict:
    """Return extra kwargs for AutoModelForCausalLM.from_pretrained."""
    # trust_remote_code not needed for Llama (official weights); harmless but
    # keeping it False for clarity. Qwen2.5 still needs it.
    return {"trust_remote_code": family == "qwen"}


def get_tokenizer_load_kwargs(family: ModelFamily) -> dict:
    return {"trust_remote_code": family == "qwen"}
