"""Step 4: Data model (dataclasses) + build hypergraph + save JSON."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .verbalize import MedicalTemplateEngine


@dataclass
class Entity:
    name: str
    entity_type: str  # disease, drug, gene/protein, pathway, ...
    node_index: Optional[int] = None  # PrimeKG node_index; preserved for feature-file joins


@dataclass
class Hyperedge:
    id: str  # "he_{i:06d}"
    description: str  # verbalized text
    entities: list[str]  # entity names
    hedge_type: str  # neighbor_agg | composite | path
    source_relation: str  # relation or path_pattern
    anchor: str = ""  # anchor entity (if applicable)


@dataclass
class MedicalHypergraph:
    entities: dict[str, Entity] = field(default_factory=dict)  # name -> Entity
    hyperedges: list[Hyperedge] = field(default_factory=list)
    entity_to_hedges: dict[str, list[str]] = field(
        default_factory=dict
    )  # entity_name -> [hedge_id, ...]


def build_hypergraph(
    all_hyperedges_raw: list[dict],
    engine: MedicalTemplateEngine,
    name_to_idx: dict[str, int] | None = None,
) -> MedicalHypergraph:
    """Build MedicalHypergraph from raw aggregated hyperedge dicts.

    Args:
        all_hyperedges_raw: Raw hedge dicts from aggregate_all + feature_hedges.
        engine:             Verbalization engine.
        name_to_idx:        Optional name → PrimeKG node_index map (from filtered_kg).
                            Populated when provided; entities not in the map get None.
    """
    entities = {}
    hyperedges = []
    entity_to_hedges: dict[str, list[str]] = {}
    _idx = name_to_idx or {}

    for i, he_raw in enumerate(all_hyperedges_raw):
        # Use pre-existing description if available (e.g. loaded from saved JSON)
        desc = he_raw.get('description') or engine.verbalize(he_raw)

        # Collect entities based on hyperedge type
        # Fall back to 'entities' list when loading from saved JSON (neighbors/parts absent)
        if he_raw['type'] == 'neighbor_agg':
            if 'neighbors' in he_raw:
                ent_list = [he_raw['anchor']] + he_raw['neighbors']
                types = [he_raw['anchor_type']] + he_raw['neighbor_types']
            else:
                ent_list = he_raw['entities']
                types = [he_raw.get('anchor_type', 'unknown')] + ['unknown'] * (len(ent_list) - 1)
        elif he_raw['type'] == 'composite':
            ent_list = he_raw['entities']
            types = [he_raw.get('anchor_type', 'unknown')] + ['unknown'] * (len(ent_list) - 1)
        elif he_raw['type'] == 'feature':
            ent_list = he_raw.get('entities', [he_raw['anchor']])
            types = [he_raw.get('anchor_type', 'unknown')] * len(ent_list)
        else:  # path
            ent_list = he_raw['entities']
            types = he_raw.get('entity_types', ['unknown'] * len(ent_list))

        for name, etype in zip(ent_list, types):
            if name not in entities:
                entities[name] = Entity(
                    name=name,
                    entity_type=etype,
                    node_index=_idx.get(name),
                )

        hedge_id = f"he_{i:06d}"
        he = Hyperedge(
            id=hedge_id,
            description=desc,
            entities=ent_list,
            hedge_type=he_raw['type'],
            source_relation=he_raw.get('relation') or he_raw.get('path_pattern', ''),
            anchor=he_raw.get('anchor', ''),
        )
        hyperedges.append(he)

        for name in ent_list:
            entity_to_hedges.setdefault(name, []).append(hedge_id)

    return MedicalHypergraph(
        entities=entities,
        hyperedges=hyperedges,
        entity_to_hedges=entity_to_hedges,
    )


def save_hypergraph(hg: MedicalHypergraph, path: str = "data/medical_hg.json"):
    """Serialize MedicalHypergraph to JSON."""
    data = {
        "entities": {
            k: {
                "name": v.name,
                "type": v.entity_type,
                **({"node_index": v.node_index} if v.node_index is not None else {}),
            }
            for k, v in hg.entities.items()
        },
        "hyperedges": [
            {
                "id": h.id,
                "description": h.description,
                "entities": h.entities,
                "type": h.hedge_type,
                "relation": h.source_relation,
                "anchor": h.anchor,
            }
            for h in hg.hyperedges
        ],
        "entity_to_hedges": hg.entity_to_hedges,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)
    print(f"Saved hypergraph to {path}")
    print(f"  Entities: {len(hg.entities):,}")
    print(f"  Hyperedges: {len(hg.hyperedges):,}")
