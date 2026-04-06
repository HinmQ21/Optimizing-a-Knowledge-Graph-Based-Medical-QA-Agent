"""Prepare retrieval evaluation set from MedQA.

Samples N questions from MedQA train set and saves them to a JSON file.
Each entry has: question, answer (text), answer_idx (letter), options.

Usage:
    python prepare_retrieval_eval.py [--n_samples 200] [--seed 42]
                                     [--output eval/retrieval_eval_200.json]
"""

import argparse
import json
import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]  # baseline/
sys.path.insert(0, str(BASE_DIR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset_path",
        default=str(BASE_DIR / "dataset/MedQA/train"),
    )
    parser.add_argument(
        "--output",
        default=str(BASE_DIR / "eval/retrieval_eval_200.json"),
    )
    args = parser.parse_args()

    from datasets import load_from_disk

    print(f"Loading MedQA from {args.dataset_path} ...")
    ds = load_from_disk(args.dataset_path)
    print(f"  Total samples: {len(ds)}")

    random.seed(args.seed)
    n = min(args.n_samples, len(ds))
    indices = random.sample(range(len(ds)), n)
    indices.sort()

    samples = []
    for idx in indices:
        row = ds[idx]
        samples.append(
            {
                "idx": idx,
                "question": row["question"],
                "answer": row["answer"],
                "answer_idx": row["answer_idx"],
                "options": dict(row["options"]),
            }
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {"n_samples": len(samples), "seed": args.seed, "samples": samples},
            f,
            indent=2,
        )
    print(f"Saved {len(samples)} samples → {args.output}")

    # Quick stats
    answers = [s["answer"] for s in samples]
    unique = len(set(a.lower() for a in answers))
    print(f"  Unique answers: {unique}/{len(samples)}")
    avg_q_len = sum(len(s["question"].split()) for s in samples) / len(samples)
    print(f"  Avg question length: {avg_q_len:.1f} words")


if __name__ == "__main__":
    main()
