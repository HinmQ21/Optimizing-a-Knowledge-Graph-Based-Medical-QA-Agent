"""Dataset formatting for GRPO medical training.

Formats MedQA / MedMCQA into TRL GRPOTrainer prompt format.
The 'answer' column is preserved so TRL can pass it to answer_reward().
"""

from datasets import Dataset, load_from_disk

SYSTEM_PROMPT = (
    "You are a medical reasoning assistant with access to a "
    "search_medical_knowledge tool.\n\n"
    "Structure your response in this order:\n"
    "1. <think>Your initial reasoning about the question</think>\n"
    "2. (Optional) If you need to verify a medical fact, call "
    "search_medical_knowledge, then add another <think>...</think> "
    "incorporating the result.\n"
    "3. <answer>Your final answer</answer>\n\n"
    "IMPORTANT: In <answer> tags, write ONLY the option letter (e.g. A) "
    "or a short answer, NOT an explanation."
)


def format_dataset(examples: dict) -> dict:
    """Format dataset examples into TRL GRPOTrainer prompt format.

    Returns a dict with 'prompt' key (list[list[dict]]) and preserves
    all other columns (e.g. 'answer') for reward function kwargs.
    """
    prompts = []
    for question, options in zip(examples["question"], examples["options"]):
        if isinstance(options, dict):
            opt_text = "\n".join(f"{k}. {v}" for k, v in options.items())
            user_content = f"{question}\n\nOptions:\n{opt_text}"
        else:
            user_content = question

        prompts.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ])
    return {"prompt": prompts}


def load_medqa(data_path: str, max_samples: int | None = None) -> Dataset:
    """Load MedQA dataset and apply prompt formatting.

    Args:
        data_path: Path to dataset directory (load_from_disk format).
        max_samples: Optional cap on number of training examples.

    Returns:
        Dataset with 'prompt' column (list[dict]) and 'answer' column.
    """
    ds = load_from_disk(data_path)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    keep = {"answer", "answer_idx"} & set(ds.column_names)
    ds = ds.map(format_dataset, batched=True, load_from_cache_file=False,
                remove_columns=[
        c for c in ds.column_names if c not in keep
    ])
    return ds
