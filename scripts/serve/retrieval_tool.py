"""Step 6A: Medical Knowledge Retrieval Tool for TRL GRPOTrainer.

Singleton pattern — load() once, then use search_medical_knowledge() as a
TRL tool callable. Encoder runs on CPU to avoid stealing VRAM from training.

Retrieval logic:
    Query
      ├── 1. Encode query (MedEmbed-large, CPU)
      ├── 2. FAISS search index_hyperedge → top-k hedge descriptions (primary)
      ├── 3. FAISS search index_entity → top-3 entity names (expand)
      │       └── Lookup entity_to_hedges → add related hedges
      └── 4. Deduplicate + return top-k descriptions
"""

import functools
import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class MedicalKnowledgeTool:
    """Singleton loader for medical hypergraph retrieval."""

    _instance = None
    _STOPWORDS = {
        'a',
        'an',
        'and',
        'are',
        'as',
        'at',
        'be',
        'by',
        'does',
        'for',
        'from',
        'how',
        'in',
        'is',
        'of',
        'on',
        'or',
        'the',
        'to',
        'what',
        'which',
        'with',
    }

    @classmethod
    def load(cls, data_dir: str = "data/") -> "MedicalKnowledgeTool":
        if cls._instance is not None:
            return cls._instance

        inst = cls()
        inst.data_dir = data_dir

        # Encoder on CUDA — safe for eval; use device='cpu' during GRPO training
        # to avoid competing with the training model for VRAM
        inst.encoder = SentenceTransformer(
            'abhinand/MedEmbed-large-v0.1', device='cuda'
        )

        # FAISS indices
        inst.idx_he = faiss.read_index(f"{data_dir}/index_hyperedge.bin")
        inst.idx_ent = faiss.read_index(f"{data_dir}/index_entity.bin")

        # ID mappings
        inst.hedge_ids = np.load(
            f"{data_dir}/hedge_ids.npy", allow_pickle=True
        ).tolist()
        inst.ent_names = np.load(
            f"{data_dir}/entity_names.npy", allow_pickle=True
        ).tolist()

        # Hypergraph data
        with open(f"{data_dir}/medical_hg.json") as f:
            hg = json.load(f)
        inst.hedge_meta = {
            h['id']: {
                'description': h['description'],
                'relation': h.get('relation', ''),
                'type': h.get('type', ''),
                'anchor': h.get('anchor', ''),
                'entities': h.get('entities', []),
            }
            for h in hg['hyperedges']
        }
        inst.hedge_by_id = {
            hid: meta['description'] for hid, meta in inst.hedge_meta.items()
        }
        inst.entity_to_hedges = hg['entity_to_hedges']
        inst.entity_type_by_name = {
            name: meta.get('type', '') for name, meta in hg['entities'].items()
        }
        inst.hedge_token_sets = {
            hid: set(
                inst._tokenize(
                    " ".join(
                        [
                            meta['description'],
                            meta['relation'],
                            meta['type'],
                            meta['anchor'],
                            *meta['entities'],
                        ]
                    )
                )
            )
            for hid, meta in inst.hedge_meta.items()
        }
        inst.entity_token_sets = {
            name: set(inst._tokenize(name)) for name in hg['entities']
        }

        cls._instance = inst
        return inst

    @classmethod
    def reset(cls):
        """Reset singleton (useful for testing)."""
        cls._instance = None

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return [t for t in tokens if t not in cls._STOPWORDS]

    @staticmethod
    def _rrf(rank: int | None, k: int = 10) -> float:
        if rank is None:
            return 0.0
        return 1.0 / (k + rank + 1)

    def _lexical_score(self, query_tokens: set[str], hid: str) -> float:
        if not query_tokens:
            return 0.0
        hedge_tokens = self.hedge_token_sets.get(hid, set())
        if not hedge_tokens:
            return 0.0
        overlap = len(query_tokens & hedge_tokens)
        return overlap / max(1, min(len(query_tokens), 6))

    def _entity_match_score(self, query_tokens: set[str], meta: dict) -> float:
        if not query_tokens:
            return 0.0

        best = 0.0
        for name in [meta.get('anchor', ''), *meta.get('entities', [])]:
            name_tokens = self.entity_token_sets.get(name)
            if not name_tokens:
                name_tokens = set(self._tokenize(name))
            if not name_tokens:
                continue
            best = max(best, len(query_tokens & name_tokens) / len(name_tokens))
        return best

    def _expansion_priority(
        self,
        query_tokens: set[str],
        entity_name: str,
        entity_sim: float,
        hid: str,
    ) -> float:
        meta = self.hedge_meta[hid]
        lexical = self._lexical_score(query_tokens, hid)
        entity_match = self._entity_match_score(query_tokens, meta)
        anchor_bonus = 1.0 if meta.get('anchor') == entity_name else 0.35

        return (
            0.60 * entity_sim
            + 0.20 * lexical
            + 0.10 * entity_match
            + 0.10 * anchor_bonus
        )

    def retrieve_v0(self, query: str, top_k: int = 5) -> list[str]:
        """Original baseline retrieval: hyperedge search + shallow entity expansion."""
        q_emb = self.encoder.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        _, he_ids = self.idx_he.search(q_emb, top_k)
        results: dict[str, str] = {}
        for i in he_ids[0]:
            if i < len(self.hedge_ids):
                hid = self.hedge_ids[i]
                results[hid] = self.hedge_by_id[hid]

        _, ent_ids = self.idx_ent.search(q_emb, 3)
        for i in ent_ids[0]:
            if i >= len(self.ent_names):
                continue
            for hid in self.entity_to_hedges.get(
                self.ent_names[i], []
            )[:2]:
                if hid not in results:
                    results[hid] = self.hedge_by_id[hid]
                if len(results) >= top_k:
                    break
            if len(results) >= top_k:
                break

        return list(results.values())[:top_k]

    def retrieve_v1(self, query: str, top_k: int = 5) -> list[str]:
        """Dual retrieval + lightweight fusion without intent-specific rules."""
        q_emb = self.encoder.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        query_tokens = set(self._tokenize(query))

        he_k = max(40, top_k * 8)
        ent_k = max(12, top_k * 4)
        per_entity_limit = 8

        he_scores, he_ids = self.idx_he.search(q_emb, he_k)
        ent_scores, ent_ids = self.idx_ent.search(q_emb, ent_k)

        candidates: dict[str, dict] = {}

        for rank, (score, idx) in enumerate(zip(he_scores[0], he_ids[0])):
            if idx < 0 or idx >= len(self.hedge_ids):
                continue
            hid = self.hedge_ids[idx]
            meta = self.hedge_meta.get(hid)
            if meta is None:
                continue
            candidates.setdefault(
                hid,
                {
                    'he_rank': None,
                    'he_sim': 0.0,
                    'ent_rank': None,
                    'ent_sim': 0.0,
                    'exp_rank': None,
                    'exp_score': 0.0,
                    'matched_entities': set(),
                },
            )
            candidates[hid]['he_rank'] = rank
            candidates[hid]['he_sim'] = max(candidates[hid]['he_sim'], float(score))

        for ent_rank, (ent_score, idx) in enumerate(zip(ent_scores[0], ent_ids[0])):
            if idx < 0 or idx >= len(self.ent_names):
                continue

            entity_name = self.ent_names[idx]
            related_hids = self.entity_to_hedges.get(entity_name, [])
            if not related_hids:
                continue

            expanded = []
            for hid in related_hids:
                if hid not in self.hedge_meta:
                    continue
                expanded.append(
                    (
                        self._expansion_priority(
                            query_tokens, entity_name, float(ent_score), hid
                        ),
                        hid,
                    )
                )

            expanded.sort(key=lambda x: x[0], reverse=True)
            for exp_rank, (exp_score, hid) in enumerate(expanded[:per_entity_limit]):
                candidates.setdefault(
                    hid,
                    {
                        'he_rank': None,
                        'he_sim': 0.0,
                        'ent_rank': None,
                        'ent_sim': 0.0,
                        'exp_rank': None,
                        'exp_score': 0.0,
                        'matched_entities': set(),
                    },
                )
                cand = candidates[hid]
                cand['ent_rank'] = (
                    ent_rank
                    if cand['ent_rank'] is None
                    else min(cand['ent_rank'], ent_rank)
                )
                cand['ent_sim'] = max(cand['ent_sim'], float(ent_score))
                cand['exp_rank'] = (
                    exp_rank
                    if cand['exp_rank'] is None
                    else min(cand['exp_rank'], exp_rank)
                )
                cand['exp_score'] = max(cand['exp_score'], float(exp_score))
                cand['matched_entities'].add(entity_name)

        ranked: list[tuple[float, str]] = []
        for hid, cand in candidates.items():
            meta = self.hedge_meta[hid]
            lexical = self._lexical_score(query_tokens, hid)
            entity_match = self._entity_match_score(query_tokens, meta)

            final_score = (
                0.41 * cand['he_sim']
                + 0.14 * cand['ent_sim']
                + 0.14 * self._rrf(cand['he_rank'])
                + 0.10 * self._rrf(cand['ent_rank'])
                + 0.08 * self._rrf(cand['exp_rank'])
                + 0.08 * min(1.0, cand['exp_score'])
                + 0.10 * lexical
                + 0.05 * entity_match
            )
            ranked.append((final_score, hid))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [self.hedge_by_id[hid] for _, hid in ranked[:top_k]]

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve top-k relevant hyperedge descriptions for a query."""
        return self.retrieve_v1(query, top_k)


# ---------------------------------------------------------------------------
# Tool function for TRL GRPOTrainer (module-level, loaded lazily)
# ---------------------------------------------------------------------------

_tool: MedicalKnowledgeTool | None = None


def _get_tool() -> MedicalKnowledgeTool:
    global _tool
    if _tool is None:
        _tool = MedicalKnowledgeTool.load()
    return _tool


@functools.lru_cache(maxsize=512)
def _retrieve_cached(query: str) -> str:
    results = _get_tool().retrieve(query, top_k=5)
    if not results:
        return "No relevant knowledge found."
    return "\n".join(f"- {r}" for r in results)


def search_medical_knowledge(query: str) -> str:
    """Search the medical knowledge graph for relevant clinical facts.

    Use this tool to retrieve information about diseases, drugs,
    symptoms, proteins, pathways, and their relationships.

    Args:
        query: A medical search query describing what you need to know.

    Returns:
        Relevant medical knowledge facts from the knowledge graph.
    """
    return _retrieve_cached(query)


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    tool = MedicalKnowledgeTool.load()
    queries = sys.argv[1:] or [
        "What drugs treat Type 2 Diabetes?",
        "Side effects of metformin",
        "Symptoms of rheumatoid arthritis",
    ]
    for q in queries:
        print(f"\nQ: {q}")
        print(search_medical_knowledge(q))
