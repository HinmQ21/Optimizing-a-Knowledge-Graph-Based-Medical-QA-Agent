"""Convert stage1_5_sft_v2.jsonl to Llama-native tool-calling format.

The v2 data has multi-turn traces where the first assistant turn contains a
<think> reasoning block but NO explicit tool call JSON. Llama's native format
requires the assistant to emit a plain JSON tool call, so we transform each
tool-using trace as follows:

  BEFORE (v2 / Qwen-style):
    [assistant] <think>...reasoning...</think>
    [tool]      search result
    [assistant] <think>final</think><answer>X</answer>

  AFTER (Llama native):
    [assistant] {"name": "search_medical_knowledge", "parameters": {"query": "..."}}
    [tool]      search result
    [assistant] <think>final</think><answer>X</answer>

The query is derived from the user question (text before "Options:").
No-tool traces (no role=tool message) are passed through unchanged.

Usage:
    cd /home/vcsai/minhlbq/baseline
    ./training_venv312/bin/python -m scripts.stage1_5.convert_data_llama \
        --input  data/stage1_5_sft_v2.jsonl \
        --output data/stage1_5_sft_llama.jsonl
"""

import argparse
import json
import re


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def extract_query(messages: list[dict]) -> str:
    """Extract a search query from the conversation.

    Priority:
      1. First meaningful line from the first assistant <think> block.
      2. First 150 chars of the user question (before Options:).
    """
    # Try to mine key terms from the first <think> block
    for m in messages:
        if m["role"] == "assistant":
            think_match = _THINK_RE.search(m.get("content", ""))
            if think_match:
                think_text = think_match.group(1).strip()
                # Take first non-empty line; strip "Scenario:", "Key terms:" etc.
                for line in think_text.splitlines():
                    line = re.sub(r"^(Scenario|Key terms|Initial assessment)\s*:\s*", "", line.strip())
                    if len(line) > 15:
                        return line[:150]
            break

    # Fall back to the user question
    for m in messages:
        if m["role"] == "user":
            user_text = m.get("content", "")
            question = user_text.split("\n\nOptions:")[0].strip()
            return question[:150]

    return "medical knowledge"


def convert_trace(sample: dict) -> dict:
    """Convert one trace to Llama-native tool-calling format."""
    messages = sample["messages"]

    # Find first role=tool index
    tool_idx = next((i for i, m in enumerate(messages) if m["role"] == "tool"), None)
    if tool_idx is None:
        # No tool call — pass through unchanged
        return sample

    # Find the assistant turn immediately before the tool response
    asst_idx = tool_idx - 1
    if asst_idx < 0 or messages[asst_idx]["role"] != "assistant":
        return sample

    query = extract_query(messages)
    tool_call_json = json.dumps({
        "name": "search_medical_knowledge",
        "parameters": {"query": query},
    })

    # Replace the assistant turn with the plain JSON tool call
    new_messages = list(messages)
    new_messages[asst_idx] = {"role": "assistant", "content": tool_call_json}

    return {**sample, "messages": new_messages}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="data/stage1_5_sft_v2.jsonl")
    p.add_argument("--output", default="data/stage1_5_sft_llama.jsonl")
    args = p.parse_args()

    n_total = n_converted = n_passthrough = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            has_tool = any(m["role"] == "tool" for m in sample["messages"])
            converted = convert_trace(sample)
            fout.write(json.dumps(converted) + "\n")
            n_total += 1
            if has_tool:
                n_converted += 1
            else:
                n_passthrough += 1

    print(f"Total:       {n_total}")
    print(f"Converted:   {n_converted}  (tool-using → Llama native JSON)")
    print(f"Pass-through:{n_passthrough}  (no-tool, unchanged)")
    print(f"Output:      {args.output}")


if __name__ == "__main__":
    main()
