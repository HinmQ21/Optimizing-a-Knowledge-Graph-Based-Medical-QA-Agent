"""Step 2: Aggregate filtered edges into hyperedges (3 strategies)."""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

# --- 2A: Neighbor Aggregation configs ---
NEIGHBOR_AGG_CONFIGS = [
    # Disease-anchored (nb = phenotypes/drugs/proteins/diseases/exposures)
    ('disease', 'disease_phenotype_positive', 8),
    ('disease', 'disease_phenotype_negative', 5),
    ('disease', 'disease_protein', 5),
    ('disease', 'disease_disease', 5),
    ('disease', 'indication', 4),          # anchor=disease → nb=drugs ✓
    ('disease', 'contraindication', 4),    # anchor=disease → nb=drugs ✓
    ('disease', 'exposure_disease', 5),    # anchor=disease → nb=exposures ✓
    # Drug-anchored (nb = diseases/proteins/effects)
    ('drug', 'indication', 4),             # anchor=drug → nb=diseases (SINGLE_BY_ANCHOR)
    ('drug', 'contraindication', 4),       # anchor=drug → nb=diseases (SINGLE_BY_ANCHOR)
    ('drug', 'off-label use', 4),          # anchor=drug → nb=diseases (SINGLE_BY_ANCHOR)
    ('drug', 'drug_protein', 4),
    ('drug', 'drug_effect', 5),
    ('drug', 'drug_drug', 4),
    # Other entity types
    ('cellular_component', 'cellcomp_protein', 5),
    ('pathway', 'pathway_protein', 6),
    ('biological_process', 'bioprocess_protein', 5),
    ('molecular_function', 'molfunc_protein', 4),
    ('effect/phenotype', 'phenotype_protein', 4),
    ('exposure', 'exposure_protein', 4),
    ('exposure', 'exposure_bioprocess', 4),
    ('exposure', 'exposure_disease', 4),   # anchor=exposure → nb=diseases (SINGLE_BY_ANCHOR)
]

# --- 2C: Path patterns ---
PATH_PATTERNS = [
    {
        'name': 'symptom_disease_drug',
        'h1': ('effect/phenotype', 'disease_phenotype_positive', 'disease'),
        'h2': ('disease', 'indication', 'drug'),
    },
    {
        'name': 'disease_protein_drug',
        'h1': ('disease', 'disease_protein', 'gene/protein'),
        'h2': ('gene/protein', 'drug_protein', 'drug'),
    },
    {
        'name': 'drug_protein_pathway',
        'h1': ('drug', 'drug_protein', 'gene/protein'),
        'h2': ('gene/protein', 'pathway_protein', 'pathway'),
    },
    {
        'name': 'exposure_disease_phenotype',
        'h1': ('exposure', 'exposure_disease', 'disease'),
        'h2': ('disease', 'disease_phenotype_positive', 'effect/phenotype'),
    },
    {
        'name': 'comorbidity_drug',
        'h1': ('disease', 'disease_disease', 'disease'),
        'h2': ('disease', 'indication', 'drug'),
    },
]


def neighbor_aggregate(
    filtered_kg: pd.DataFrame, anchor_type: str, relation: str, max_n: int
) -> list[dict]:
    """2A: Group neighbors by anchor entity and relation into hyperedges."""
    results = []
    mask = (filtered_kg['x_type'] == anchor_type) & (
        filtered_kg['relation'] == relation
    )
    subset = filtered_kg[mask]
    for anchor_name, group in subset.groupby('x_name'):
        neighbors = group['y_name'].tolist()[:max_n]
        if len(neighbors) < 2:
            continue
        results.append(
            {
                'type': 'neighbor_agg',
                'anchor': anchor_name,
                'anchor_type': anchor_type,
                'neighbors': neighbors,
                'neighbor_types': group['y_type'].tolist()[:max_n],
                'relation': relation,
            }
        )
    return results


def _build_neighbor_lookup(
    filtered_kg: pd.DataFrame,
) -> dict[tuple[str, str], list[str]]:
    """Pre-build (x_name, relation) -> [y_name, ...] lookup dict.

    O(n) scan once instead of O(n) per entity per relation.
    """
    lookup: dict[tuple[str, str], list[str]] = {}
    for x_name, relation, y_name in zip(
        filtered_kg['x_name'], filtered_kg['relation'], filtered_kg['y_name']
    ):
        lookup.setdefault((x_name, relation), []).append(y_name)
    return lookup


# Composite configs
DISEASE_COMPOSITE_RELS = [
    ('disease_phenotype_positive', 'presents with', 4),
    ('disease_phenotype_negative', 'notably without', 3),
    ('indication', 'treated with', 3),
    ('contraindication', 'contraindicated with', 2),
    ('exposure_disease', 'risk factors include', 3),
    ('disease_disease', 'co-occurs with', 2),
]

DRUG_COMPOSITE_RELS = [
    ('indication', 'indicated for', 3),
    ('drug_protein', 'targets', 3),
    ('drug_effect', 'effects include', 4),
    ('contraindication', 'contraindicated in', 3),
]


def _composite_from_lookup(
    entity_name: str,
    entity_type: str,
    rel_configs: list[tuple[str, str, int]],
    lookup: dict[tuple[str, str], list[str]],
) -> Optional[dict]:
    """Build a composite hyperedge using pre-built lookup dict."""
    parts, entities = [], [entity_name]
    for rel, key, n in rel_configs:
        nb = lookup.get((entity_name, rel), [])[:n]
        if nb:
            entities.extend(nb)
            parts.append((key, nb))
    if len(parts) < 2:
        return None
    return {
        'type': 'composite',
        'anchor': entity_name,
        'anchor_type': entity_type,
        'entities': entities,
        'parts': parts,
    }


def extract_paths(
    pattern: dict,
    filtered_kg: pd.DataFrame,
    priority_entities: Optional[set] = None,
    limit: int = 5000,
) -> list[dict]:
    """2C: Extract 2-hop paths following a clinical reasoning pattern."""
    h1_rel, h2_rel = pattern['h1'][1], pattern['h2'][1]
    hop1 = filtered_kg[filtered_kg['relation'] == h1_rel]
    hop2 = filtered_kg[filtered_kg['relation'] == h2_rel]
    paths = hop1.merge(
        hop2, left_on='y_index', right_on='x_index', suffixes=('_h1', '_h2')
    )

    if priority_entities and len(paths) > limit:
        pri = paths[
            paths['x_name_h1'].isin(priority_entities)
            | paths['y_name_h1'].isin(priority_entities)
            | paths['y_name_h2'].isin(priority_entities)
        ]
        rest = paths[~paths.index.isin(pri.index)]
        if len(pri) >= limit:
            # Priority set alone exceeds limit — sample within priority paths
            paths = pri.sample(limit, random_state=42)
        else:
            paths = pd.concat(
                [pri, rest.sample(min(max(0, limit - len(pri)), len(rest)), random_state=42)]
            )
    elif len(paths) > limit:
        paths = paths.sample(limit, random_state=42)

    return [
        {
            'type': 'path',
            'path_pattern': pattern['name'],
            'entities': [r['x_name_h1'], r['y_name_h1'], r['y_name_h2']],
            'entity_types': [pattern['h1'][0], pattern['h1'][2], pattern['h2'][2]],
            'relations': [h1_rel, h2_rel],
        }
        for _, r in paths.iterrows()
    ]


def aggregate_all(
    filtered_kg: pd.DataFrame,
    priority_entities: Optional[set] = None,
    path_limit: int = 5000,
) -> list[dict]:
    """Run all 3 aggregation strategies and return combined hyperedges."""
    all_hedges = []

    # 2A: Neighbor aggregation
    print("Running neighbor aggregation...")
    for anchor_type, relation, max_n in NEIGHBOR_AGG_CONFIGS:
        hedges = neighbor_aggregate(filtered_kg, anchor_type, relation, max_n)
        all_hedges.extend(hedges)
    print(f"  Neighbor hyperedges: {len(all_hedges):,}")

    # 2B: Cross-type composite (using pre-built lookup for speed)
    print("Running cross-type composite...")
    n_before = len(all_hedges)
    lookup = _build_neighbor_lookup(filtered_kg)
    print("  Built neighbor lookup dict")

    diseases = filtered_kg[filtered_kg['x_type'] == 'disease']['x_name'].unique()
    for d in diseases:
        he = _composite_from_lookup(d, 'disease', DISEASE_COMPOSITE_RELS, lookup)
        if he:
            all_hedges.append(he)

    drugs = filtered_kg[filtered_kg['x_type'] == 'drug']['x_name'].unique()
    for d in drugs:
        he = _composite_from_lookup(d, 'drug', DRUG_COMPOSITE_RELS, lookup)
        if he:
            all_hedges.append(he)
    print(f"  Composite hyperedges: {len(all_hedges) - n_before:,}")

    # 2C: Path aggregation
    print("Running path aggregation...")
    n_before = len(all_hedges)
    for pattern in PATH_PATTERNS:
        hedges = extract_paths(pattern, filtered_kg, priority_entities, path_limit)
        all_hedges.extend(hedges)
        print(f"    {pattern['name']}: {len(hedges):,} paths")
    print(f"  Path hyperedges: {len(all_hedges) - n_before:,}")

    print(f"Total hyperedges: {len(all_hedges):,}")
    return all_hedges


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate filtered KG into hyperedges')
    parser.add_argument('--input', default='data/filtered_kg.parquet')
    parser.add_argument('--path-limit', type=int, default=5000)
    args = parser.parse_args()

    filtered_kg = pd.read_parquet(args.input)
    hedges = aggregate_all(filtered_kg, path_limit=args.path_limit)
    print(f"\nDone. {len(hedges):,} hyperedges generated.")
