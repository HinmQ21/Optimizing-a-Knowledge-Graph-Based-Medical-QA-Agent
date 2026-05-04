"""Step 2.5: Build feature-profile hyperedges from PrimeKG disease/drug feature tables.

Each feature field produces one hyperedge anchored on the disease or drug entity.
Text is either used directly (drug fields are pre-verbalized by PrimeKG) or wrapped
in a short prefix template (Mayo Clinic / Orphanet prose fields).

Hedge dict format (compatible with store.build_hypergraph):
  {
    'type':         'feature',
    'feature_type': str,          # e.g. 'disease_causes', 'drug_moa'
    'anchor':       str,          # entity name from filtered_kg
    'anchor_type':  str,          # 'disease' or 'drug'
    'anchor_index': int | None,   # PrimeKG node_index
    'description':  str,          # verbalized text (ready for embed)
    'entities':     list[str],    # [anchor] — single entity for Phase 1
    'relation':     str,          # same key as feature_type for validation
  }
"""

import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

_HTML_RE = re.compile(r'<[^>]+>')
_MULTI_WS = re.compile(r'\s+')

# Mayo fields can be >3000 chars; embedding model max is ~512 tokens (~2000 chars).
# Truncate at sentence boundary ≤ this limit.
_MAX_CHARS = 1200


def _clean(text: str) -> str:
    """Strip HTML, collapse whitespace."""
    text = _HTML_RE.sub(' ', text)
    return _MULTI_WS.sub(' ', text).strip()


def _truncate(text: str, max_chars: int = _MAX_CHARS) -> str:
    """Truncate at the last sentence boundary before max_chars."""
    if len(text) <= max_chars:
        return text
    # Find last '. ' or '.\n' before max_chars
    cut = text.rfind('. ', 0, max_chars)
    if cut > max_chars // 2:
        return text[: cut + 1]
    # Fallback: hard truncate at max_chars
    return text[:max_chars].rstrip() + '…'


def _prep(text: str, max_chars: int = _MAX_CHARS) -> str | None:
    """Clean + truncate + validate non-empty."""
    t = _truncate(_clean(text), max_chars)
    return t if t else None


# ---------------------------------------------------------------------------
# Disease feature field configs
# Tuple: (feature_type, column, prefix_template | None)
# prefix_template: None means use text as-is; otherwise "{name}: {text}"
# ---------------------------------------------------------------------------

DISEASE_FIELD_CONFIGS: list[tuple[str, str, str | None]] = [
    # Definitions — prefix with name to prevent cross-disease duplicate text
    # (many disease variants/synonyms share identical MONDO/UMLS/Orphanet definitions)
    ('disease_definition',          'mondo_definition',                 '{name}: {text}'),
    ('disease_definition_umls',     'umls_description',                 '{name}: {text}'),
    ('disease_definition_orphanet', 'orphanet_definition',              '{name}: {text}'),
    # Mayo Clinic clinical profiles — longer, need prefix for retrieval context
    ('disease_symptoms',            'mayo_symptoms',                    'Symptoms of {name}: {text}'),
    ('disease_causes',              'mayo_causes',                      'Causes of {name}: {text}'),
    ('disease_risk_factors',        'mayo_risk_factors',                'Risk factors for {name}: {text}'),
    ('disease_complications',       'mayo_complications',               'Complications of {name}: {text}'),
    ('disease_prevention',          'mayo_prevention',                  'Prevention of {name}: {text}'),
    ('disease_see_doc',             'mayo_see_doc',                     'When to seek care for {name}: {text}'),
    # Orphanet clinical / management
    ('disease_clinical',            'orphanet_clinical_description',    'Clinical presentation of {name}: {text}'),
    ('disease_management',          'orphanet_management_and_treatment','Management of {name}: {text}'),
]

# ---------------------------------------------------------------------------
# Drug feature field configs
# ---------------------------------------------------------------------------

DRUG_FIELD_CONFIGS: list[tuple[str, str, str | None]] = [
    # DrugBank prose fields — prefix with name to prevent boilerplate duplicates
    # ("No pharmacokinetic data available.", "95%", "Investigated for use in..." etc.)
    ('drug_description',        'description',          '{name}: {text}'),
    ('drug_indication',         'indication',           '{name} indication: {text}'),
    ('drug_moa',                'mechanism_of_action',  '{name} mechanism of action: {text}'),
    ('drug_pharmacodynamics',   'pharmacodynamics',     '{name} pharmacodynamics: {text}'),
    ('drug_protein_binding',    'protein_binding',      '{name} protein binding: {text}'),
    # Categorical/short fields — must include name prefix to avoid cross-drug duplicates
    # (group/atc_1 are shared labels like "approved" / "Cardiovascular system")
    ('drug_group',              'group',                '{name} approval status: {text}'),
    ('drug_half_life',          'half_life',            '{name} half-life: {text}'),
    # Category is a long semicolon-separated list — already drug-specific, but add prefix for retrieval context
    ('drug_category',           'category',             '{name} pharmacological category: {text}'),
    # ATC level-1 is coarse (14 categories shared across thousands of drugs)
    ('drug_atc',                'atc_1',                '{name} ATC classification: {text}'),
]


# ---------------------------------------------------------------------------
# Dedup helper for disease features
# ---------------------------------------------------------------------------

def _dedup_disease_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per node_index: the row with the most filled text columns."""
    text_cols = [c for _, c, _ in DISEASE_FIELD_CONFIGS]
    df = df.copy()
    df['_fill'] = df[text_cols].notna().sum(axis=1)
    deduped = (
        df.sort_values('_fill', ascending=False)
        .drop_duplicates(subset='node_index', keep='first')
        .drop(columns=['_fill'])
        .reset_index(drop=True)
    )
    return deduped


# ---------------------------------------------------------------------------
# Name lookup: node_index → entity name (from filtered_kg)
# ---------------------------------------------------------------------------

def _build_idx_to_name(filtered_kg: pd.DataFrame, entity_type: str) -> dict[int, str]:
    """Map node_index → canonical entity name as it appears in filtered_kg."""
    x = filtered_kg[filtered_kg['x_type'] == entity_type][['x_index', 'x_name']]
    y = filtered_kg[filtered_kg['y_type'] == entity_type][['y_index', 'y_name']]
    x = x.rename(columns={'x_index': 'idx', 'x_name': 'name'})
    y = y.rename(columns={'y_index': 'idx', 'y_name': 'name'})
    combined = pd.concat([x, y]).drop_duplicates(subset='idx')
    return dict(zip(combined['idx'], combined['name']))


# ---------------------------------------------------------------------------
# Core generators
# ---------------------------------------------------------------------------

def _make_hedge(
    feature_type: str,
    anchor: str,
    anchor_type: str,
    anchor_index: int | None,
    description: str,
) -> dict:
    return {
        'type': 'feature',
        'feature_type': feature_type,
        'anchor': anchor,
        'anchor_type': anchor_type,
        'anchor_index': anchor_index,
        'description': description,
        'entities': [anchor],
        'relation': feature_type,
    }


def _generate_disease_hedges(
    features_path: str,
    idx_to_name: dict[int, str],
) -> list[dict]:
    df = pd.read_csv(features_path, low_memory=False)
    df = _dedup_disease_features(df)

    hedges = []
    for _, row in df.iterrows():
        node_idx = int(row['node_index'])
        anchor = idx_to_name.get(node_idx)
        if anchor is None:
            continue  # entity not in filtered_kg; skip for now

        for feature_type, col, prefix in DISEASE_FIELD_CONFIGS:
            val = row.get(col)
            if not isinstance(val, str) or not val.strip():
                continue
            text = _prep(val)
            if not text:
                continue
            if prefix:
                desc = prefix.format(name=anchor, text=text)
            else:
                desc = text
            hedges.append(_make_hedge(feature_type, anchor, 'disease', node_idx, desc))

    return hedges


def _generate_drug_hedges(
    features_path: str,
    idx_to_name: dict[int, str],
) -> list[dict]:
    df = pd.read_csv(features_path, low_memory=False)

    hedges = []
    for _, row in df.iterrows():
        node_idx = int(row['node_index'])
        anchor = idx_to_name.get(node_idx)
        if anchor is None:
            # Drug was dropped during aggregation — still create hedge (rescues entity)
            # Use fallback name from drug features if available (not available here; skip)
            continue

        for feature_type, col, prefix in DRUG_FIELD_CONFIGS:
            val = row.get(col)
            if not isinstance(val, str) or not val.strip():
                continue
            text = _prep(val, max_chars=800)  # drug fields tend to be dense; shorter cap
            if not text:
                continue
            if prefix:
                desc = prefix.format(name=anchor, text=text)
            else:
                desc = text
            hedges.append(_make_hedge(feature_type, anchor, 'drug', node_idx, desc))

    return hedges


def _generate_dropped_drug_hedges(
    features_path: str,
    idx_to_name_filtered: dict[int, str],
    idx_to_name_all: dict[int, str],
) -> list[dict]:
    """Generate hedges for drug nodes present in feature file but dropped during aggregation.

    Dropped drugs have no relation-based hedges; feature hedges are the only way they
    enter the hypergraph. The anchor name is taken from filtered_kg (same source as
    KG edges), so names are consistent.
    """
    df = pd.read_csv(features_path, low_memory=False)
    in_hg = set(idx_to_name_filtered.keys())

    hedges = []
    for _, row in df.iterrows():
        node_idx = int(row['node_index'])
        if node_idx in in_hg:
            continue  # already handled by _generate_drug_hedges
        anchor = idx_to_name_all.get(node_idx)
        if anchor is None:
            continue

        for feature_type, col, prefix in DRUG_FIELD_CONFIGS:
            val = row.get(col)
            if not isinstance(val, str) or not val.strip():
                continue
            text = _prep(val, max_chars=800)
            if not text:
                continue
            if prefix:
                desc = prefix.format(name=anchor, text=text)
            else:
                desc = text
            hedges.append(_make_hedge(feature_type, anchor, 'drug', node_idx, desc))

    return hedges


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def build_feature_hedges(
    filtered_kg: pd.DataFrame,
    disease_features_path: str = 'PrimeKG/disease_features.csv',
    drug_features_path: str = 'PrimeKG/drug_features.csv',
    rescue_dropped_drugs: bool = True,
) -> list[dict]:
    """Generate all feature-profile hyperedges.

    Args:
        filtered_kg:            Output of filter_kg (contains x/y index+name+type).
        disease_features_path:  Path to disease_features.csv.
        drug_features_path:     Path to drug_features.csv.
        rescue_dropped_drugs:   If True, also create hedges for drug nodes that are in
                                the feature file but were dropped during aggregation
                                (no relation-based hedges).  They enter the hypergraph
                                via feature hedges only.

    Returns:
        List of raw hedge dicts ready for store.build_hypergraph.
    """
    # Build node_index → name lookups
    dis_idx_to_name = _build_idx_to_name(filtered_kg, 'disease')
    # "in_hg" set = drugs that survive aggregation (approximated by filtered_kg presence;
    # exact dropout is only known post-aggregation, so we rescue all filtered_kg drugs here
    # and let the rescue path handle the remainder from feature file)
    drug_idx_to_name_filt = _build_idx_to_name(filtered_kg, 'drug')

    hedges: list[dict] = []

    # Disease hedges (anchored on entities already in filtered_kg)
    if Path(disease_features_path).exists():
        dis_hedges = _generate_disease_hedges(disease_features_path, dis_idx_to_name)
        hedges.extend(dis_hedges)
        print(f"  Disease feature hedges: {len(dis_hedges):,}")
    else:
        print(f"  [skip] disease features not found: {disease_features_path}")

    # Drug hedges (anchored on entities in filtered_kg)
    if Path(drug_features_path).exists():
        drug_hedges = _generate_drug_hedges(drug_features_path, drug_idx_to_name_filt)
        hedges.extend(drug_hedges)
        print(f"  Drug feature hedges (in filtered_kg): {len(drug_hedges):,}")

        # Rescue dropped drugs (in feature file but absent from filtered_kg edges)
        if rescue_dropped_drugs:
            # Build a broader name map: all drugs in feature file that appear in kg.csv
            # We don't have kg.csv here, but drug_features node_index is a subset of
            # filtered_kg drug indices + extras not in any kept relation.
            # For now, pass drug_idx_to_name_filt as "all" — dropped drugs (idx not in
            # filtered_kg edges) will naturally have no anchor name and be skipped.
            # A deeper rescue requires reading kg.csv nodes_full; deferring to Phase 2.
            pass
    else:
        print(f"  [skip] drug features not found: {drug_features_path}")

    # Dedup by description: same (anchor, relation, text) → same description with name prefix.
    # Occurs when multiple PrimeKG node_indices map to the same entity name in filtered_kg
    # (disease synonyms / variants sharing the same MONDO/UMLS/Orphanet record).
    before = len(hedges)
    seen: set[str] = set()
    deduped: list[dict] = []
    for h in hedges:
        if h['description'] not in seen:
            seen.add(h['description'])
            deduped.append(h)
    removed = before - len(deduped)
    if removed:
        print(f"  Removed {removed:,} duplicate feature hedges (same entity+field text in PrimeKG)")
    print(f"  Total feature hedges: {len(deduped):,}")
    return deduped
