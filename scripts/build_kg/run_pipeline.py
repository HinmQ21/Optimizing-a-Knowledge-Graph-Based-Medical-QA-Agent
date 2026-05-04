"""Orchestrator: Run the full KG pipeline end-to-end.

Usage:
    cd baseline/
    python -m scripts.build_kg.run_pipeline

    # Or with custom paths:
    python -m scripts.build_kg.run_pipeline --kg-path PrimeKG/kg.csv --data-dir data/

    # Skip priority entities (faster):
    python -m scripts.build_kg.run_pipeline --skip-priority
"""

import argparse
import time

import pandas as pd

from .aggregate import aggregate_all
from .embed import build_index
from .feature_hedges import build_feature_hedges
from .filter import filter_kg
from .priority import build_priority_entities
from .store import build_hypergraph, save_hypergraph
from .verbalize import MedicalTemplateEngine


def run_pipeline(
    kg_path: str = "PrimeKG/kg.csv",
    data_dir: str = "data",
    skip_priority: bool = False,
    benchmark_dirs: list[str] | None = None,
    path_limit: int = 5000,
    embed_model: str = "abhinand/MedEmbed-large-v0.1",
    embed_batch_size: int = 512,
    embed_device: str | None = None,
    disease_features_path: str = "PrimeKG/disease_features.csv",
    drug_features_path: str = "PrimeKG/drug_features.csv",
    skip_features: bool = False,
):
    t0 = time.time()

    # --- Step 1: Filter ---
    print("=" * 60)
    print("STEP 1: Filter PrimeKG")
    print("=" * 60)
    parquet_path = f"{data_dir}/filtered_kg.parquet"
    filtered_kg = filter_kg(kg_path, parquet_path)
    print()

    # --- Step 2 (optional): Priority entities ---
    priority_entities = None
    if not skip_priority:
        print("=" * 60)
        print("STEP 2 (pre): Build priority entities from benchmarks")
        print("=" * 60)
        if benchmark_dirs is None:
            benchmark_dirs = [
                "dataset/MedQA/train",
                "dataset/MedMCQA_4options/train",
                "dataset/PubMedQA/train",
            ]
        entity_names = set(filtered_kg['x_name'].unique()) | set(
            filtered_kg['y_name'].unique()
        )
        try:
            priority_entities = build_priority_entities(
                benchmark_dirs, entity_names
            )
        except Exception as e:
            print(f"  Warning: priority entities failed ({e}), continuing without.")
        print()

    # --- Step 2: Aggregate ---
    print("=" * 60)
    print("STEP 2: Aggregate into hyperedges")
    print("=" * 60)
    all_hedges = aggregate_all(filtered_kg, priority_entities, path_limit)
    print()

    # --- Step 2.5: Feature-profile hyperedges ---
    if not skip_features:
        print("=" * 60)
        print("STEP 2.5: Build feature-profile hyperedges")
        print("=" * 60)
        feature_hedges = build_feature_hedges(
            filtered_kg=filtered_kg,
            disease_features_path=disease_features_path,
            drug_features_path=drug_features_path,
        )
        all_hedges.extend(feature_hedges)
        print(f"  Relation-based hedges: {len(all_hedges) - len(feature_hedges):,}")
        print(f"  Feature hedges:        {len(feature_hedges):,}")
        print(f"  Total:                 {len(all_hedges):,}")
        print()

    # --- Step 3+4: Verbalize + Store ---
    print("=" * 60)
    print("STEP 3+4: Verbalize + Build + Save hypergraph")
    print("=" * 60)
    engine = MedicalTemplateEngine()

    # Build name→node_index map from filtered_kg for entity schema preservation
    name_to_idx: dict[str, int] = {}
    for col_name, col_idx in [('x_name', 'x_index'), ('y_name', 'y_index')]:
        for name, idx in zip(filtered_kg[col_name], filtered_kg[col_idx]):
            if name not in name_to_idx:
                name_to_idx[name] = int(idx)

    hg = build_hypergraph(all_hedges, engine, name_to_idx=name_to_idx)
    hg_path = f"{data_dir}/medical_hg.json"
    save_hypergraph(hg, hg_path)
    print()

    # --- Step 5: Embed + Index ---
    print("=" * 60)
    print("STEP 5: Embed + Build FAISS indices")
    print("=" * 60)
    build_index(
        hg_path,
        data_dir,
        embed_model,
        embed_batch_size,
        embed_device,
    )
    print()

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"  Entities:   {len(hg.entities):,}")
    print(f"  Hyperedges: {len(hg.hyperedges):,}")
    print(f"  Output dir: {data_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run full KG pipeline')
    parser.add_argument('--kg-path', default='PrimeKG/kg.csv')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--skip-priority', action='store_true')
    parser.add_argument(
        '--benchmarks',
        nargs='+',
        default=None,
        help='Benchmark dirs for priority entities',
    )
    parser.add_argument('--path-limit', type=int, default=5000)
    parser.add_argument('--embed-model', default='abhinand/MedEmbed-large-v0.1')
    parser.add_argument('--embed-batch-size', type=int, default=256)
    parser.add_argument('--embed-device', default=None)
    parser.add_argument('--disease-features', default='PrimeKG/disease_features.csv',
                        help='Path to disease_features.csv')
    parser.add_argument('--drug-features', default='PrimeKG/drug_features.csv',
                        help='Path to drug_features.csv')
    parser.add_argument('--skip-features', action='store_true',
                        help='Skip feature-profile hyperedges (relation-based only)')
    args = parser.parse_args()

    run_pipeline(
        kg_path=args.kg_path,
        data_dir=args.data_dir,
        skip_priority=args.skip_priority,
        benchmark_dirs=args.benchmarks,
        path_limit=args.path_limit,
        embed_model=args.embed_model,
        embed_batch_size=args.embed_batch_size,
        embed_device=args.embed_device,
        disease_features_path=args.disease_features,
        drug_features_path=args.drug_features,
        skip_features=args.skip_features,
    )
