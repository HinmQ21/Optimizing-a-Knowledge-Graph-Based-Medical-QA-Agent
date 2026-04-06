"""Add missing 'answer' text column to MedMCQA_4options dataset.

MedMCQA has answer_idx (letter A/B/C/D) but no answer text column.
answer_reward() needs both. This script maps answer_idx → options[answer_idx].

Usage:
    python fix_medmcqa_answer.py
"""

from pathlib import Path
from datasets import load_from_disk, DatasetDict

BASE_DIR = Path(__file__).resolve().parents[2]
SRC = BASE_DIR / "dataset/MedMCQA_4options"
DST = BASE_DIR / "dataset/MedMCQA_4options_fixed"


def add_answer(example):
    idx = example["answer_idx"]  # already a letter: A/B/C/D
    example["answer"] = example["options"].get(idx, "")
    return example


def main():
    print(f"Loading from {SRC} ...")
    ds = load_from_disk(str(SRC))

    # Handle both DatasetDict and Dataset
    if isinstance(ds, DatasetDict):
        fixed = DatasetDict({
            split: ds[split].map(add_answer, load_from_cache_file=False)
            for split in ds
        })
    else:
        fixed = ds.map(add_answer, load_from_cache_file=False)

    # Verify
    sample = fixed[0] if not isinstance(fixed, DatasetDict) else fixed["train"][0]
    print(f"Sample answer_idx={sample['answer_idx']}  answer='{sample['answer']}'")

    # Check no empty answers
    if isinstance(fixed, DatasetDict):
        for split in fixed:
            n_empty = sum(1 for ex in fixed[split] if not ex["answer"])
            print(f"  [{split}] empty answers: {n_empty}/{len(fixed[split])}")
    else:
        n_empty = sum(1 for ex in fixed if not ex["answer"])
        print(f"  Empty answers: {n_empty}/{len(fixed)}")

    print(f"\nSaving to {DST} ...")
    fixed.save_to_disk(str(DST))
    print("Done.")


if __name__ == "__main__":
    main()
