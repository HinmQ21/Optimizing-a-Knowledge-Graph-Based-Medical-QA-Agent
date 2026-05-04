"""Step 6 (optional): Build priority entity set from benchmark questions.

Priority entities bias path sampling (Step 2C) toward entities that appear
in evaluation benchmarks, improving retrieval coverage on test questions.
"""

import argparse
import re

from datasets import load_from_disk
from rapidfuzz import process as fuzz_process


def build_priority_entities(
    benchmark_dirs: list[str],
    kg_entity_names: set[str],
    fuzzy_threshold: int = 85,
) -> set[str]:
    """Scan benchmark questions -> extract medical terms -> fuzzy match with KG entities.

    No NER required. Fuzzy match compensates for mismatches like
    'type 2 diabetes' vs 'Type 2 Diabetes Mellitus'.
    """
    candidate_terms: set[str] = set()
    for bench_dir in benchmark_dirs:
        print(f"  Scanning {bench_dir}...")
        ds = load_from_disk(bench_dir)
        for example in ds:
            text = " ".join(
                [
                    example.get('question', ''),
                    example.get('answer', ''),
                    *example.get('options', {}).values(),
                ]
            )
            # Title Case noun phrases and ALL CAPS abbreviations
            candidate_terms.update(
                re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            )
            candidate_terms.update(re.findall(r'\b[A-Z]{2,}\b', text))

    print(f"  Extracted {len(candidate_terms):,} candidate terms")

    # Exact match (case-insensitive)
    kg_lower = {e.lower(): e for e in kg_entity_names}
    priority = {kg_lower[t.lower()] for t in candidate_terms if t.lower() in kg_lower}
    print(f"  Exact matches: {len(priority):,}")

    # Fuzzy match for the rest — batch via cdist (parallel, much faster than loop)
    unmatched = [t for t in candidate_terms if t.lower() not in kg_lower]
    # Skip very long terms (>6 words); KG entity names are almost always ≤5 words
    unmatched = [t for t in unmatched if len(t.split()) <= 6]
    kg_list = list(kg_entity_names)
    fuzzy_hits = 0
    if unmatched:
        import numpy as np
        from rapidfuzz import fuzz
        # cdist: shape (len(unmatched), len(kg_list)), workers=-1 uses all CPU cores
        scores = fuzz_process.cdist(
            unmatched, kg_list,
            scorer=fuzz.WRatio,
            score_cutoff=fuzzy_threshold,
            workers=-1,
        )
        # For each query term, pick the best-matching KG entity (if any passed cutoff)
        best_idx = np.argmax(scores, axis=1)
        best_scores = scores[np.arange(len(unmatched)), best_idx]
        for i, (score, kg_idx) in enumerate(zip(best_scores, best_idx)):
            if score >= fuzzy_threshold:
                priority.add(kg_list[kg_idx])
                fuzzy_hits += 1
    print(f"  Fuzzy matches: {fuzzy_hits:,}")
    print(f"  Total priority entities: {len(priority):,}")

    return priority


if __name__ == '__main__':
    import pandas as pd

    parser = argparse.ArgumentParser(description='Build priority entities from benchmarks')
    parser.add_argument(
        '--benchmarks',
        nargs='+',
        default=[
            'dataset/MedQA/train',
            'dataset/MedMCQA_4options/train',
            'dataset/PubMedQA/train',
        ],
    )
    parser.add_argument('--kg-parquet', default='data/filtered_kg.parquet')
    parser.add_argument('--fuzzy-threshold', type=int, default=85)
    args = parser.parse_args()

    kg = pd.read_parquet(args.kg_parquet)
    entity_names = set(kg['x_name'].unique()) | set(kg['y_name'].unique())
    priority = build_priority_entities(
        args.benchmarks, entity_names, args.fuzzy_threshold
    )
    print(f"\nDone. {len(priority):,} priority entities.")
