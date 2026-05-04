"""Medical Hypergraph Validation — multi-dimensional quality assessment.

Usage:
    cd baseline/
    ./stage2_venv312/bin/python -m scripts.build_kg.validate_kg
    ./stage2_venv312/bin/python -m scripts.build_kg.validate_kg --only-intrinsic
    ./stage2_venv312/bin/python -m scripts.build_kg.validate_kg --n-samples 300

See docs/kg_validation.md for methodology.
"""

import argparse
import ast
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 1. Intrinsic Quality
# ---------------------------------------------------------------------------

def validate_intrinsic(hg: dict) -> dict:
    """Validate hypergraph structural quality (no model needed)."""
    entities = hg['entities']
    hedges = hg['hyperedges']
    e2h = hg['entity_to_hedges']

    results = {}

    # 1a. Basic stats
    ent_per_hedge = [len(h['entities']) for h in hedges]
    results['entity_count'] = len(entities)
    results['hyperedge_count'] = len(hedges)
    results['avg_entities_per_hedge'] = float(np.mean(ent_per_hedge))
    results['min_entities_per_hedge'] = min(ent_per_hedge)
    results['max_entities_per_hedge'] = max(ent_per_hedge)
    results['entity_types'] = dict(Counter(e['type'] for e in entities.values()))
    results['hedge_types'] = dict(Counter(h['type'] for h in hedges))

    # 1b. Connectivity
    hedge_counts = [len(v) for v in e2h.values()]
    results['avg_hedges_per_entity'] = float(np.mean(hedge_counts))
    results['median_hedges_per_entity'] = float(np.median(hedge_counts))
    results['orphan_entities'] = sum(1 for e in entities if e not in e2h)
    results['low_connectivity_pct'] = sum(1 for c in hedge_counts if c == 1) / len(entities)

    # 1c. Verbalization integrity
    placeholder_re = re.compile(r'\{(anchor|nb|e[0-9])\}')
    results['placeholder_leaks'] = sum(1 for h in hedges if placeholder_re.search(h['description']))
    results['empty_descriptions'] = sum(1 for h in hedges if not h['description'].strip())
    results['empty_entity_names'] = sum(1 for h in hedges if '' in h['entities'])

    desc_counts = Counter(h['description'] for h in hedges)
    n_dup = sum(c for c in desc_counts.values() if c > 1)
    results['duplicate_descriptions'] = n_dup
    results['duplicate_pct'] = n_dup / len(hedges)

    composites = [h for h in hedges if h['type'] == 'composite']
    if composites:
        results['composite_no_period'] = sum(
            1 for h in composites if not h['description'].rstrip().endswith('.')
        )
    else:
        results['composite_no_period'] = 0

    desc_lens = [len(h['description'].split()) for h in hedges]
    results['desc_len_mean'] = float(np.mean(desc_lens))
    results['desc_len_median'] = float(np.median(desc_lens))
    results['desc_len_min'] = min(desc_lens)
    results['desc_len_p5'] = float(np.percentile(desc_lens, 5))
    results['desc_len_p95'] = float(np.percentile(desc_lens, 95))

    # 1d. Relation coverage
    relations = Counter(h['relation'] for h in hedges)
    results['relations_covered'] = dict(relations)

    # 1e. Feature hedge coverage (if present)
    feature_hedges = [h for h in hedges if h.get('type') == 'feature']
    if feature_hedges:
        results['feature_hedge_count'] = len(feature_hedges)
        results['feature_types'] = dict(Counter(h.get('relation', h.get('feature_type', '')) for h in feature_hedges))
        # Entities that exist ONLY in feature hedges (not in any relation-based hedge)
        relation_entities: set[str] = set()
        for h in hedges:
            if h.get('type') != 'feature':
                for ent in h.get('entities', []):
                    relation_entities.add(ent)
        feature_only_entities: set[str] = set()
        for h in feature_hedges:
            for ent in h.get('entities', []):
                if ent not in relation_entities:
                    feature_only_entities.add(ent)
        results['feature_only_entities'] = len(feature_only_entities)

    return results


def print_intrinsic(r: dict):
    print('=' * 60)
    print('1. INTRINSIC QUALITY')
    print('=' * 60)

    print(f'  Entities:            {r["entity_count"]:,}')
    print(f'  Hyperedges:          {r["hyperedge_count"]:,}')
    print(f'  Avg entities/hedge:  {r["avg_entities_per_hedge"]:.1f}')
    print(f'  Entity types:        {len(r["entity_types"])}')
    print()

    print(f'  Avg hedges/entity:   {r["avg_hedges_per_entity"]:.1f}')
    print(f'  Median hedges/entity:{r["median_hedges_per_entity"]:.0f}')
    print(f'  Orphan entities:     {r["orphan_entities"]}', _pass(r['orphan_entities'] == 0))
    print(f'  Low-connectivity:    {r["low_connectivity_pct"]:.0%}', _pass(r['low_connectivity_pct'] < 0.5))
    print()

    print(f'  Placeholder leaks:   {r["placeholder_leaks"]}', _pass(r['placeholder_leaks'] == 0))
    print(f'  Empty descriptions:  {r["empty_descriptions"]}', _pass(r['empty_descriptions'] == 0))
    print(f'  Empty entity names:  {r["empty_entity_names"]}', _pass(r['empty_entity_names'] == 0))
    print(f'  Duplicate descs:     {r["duplicate_descriptions"]} ({r["duplicate_pct"]:.2%})',
          _pass(r['duplicate_pct'] < 0.001))
    print(f'  Composite no period: {r["composite_no_period"]}', _pass(r['composite_no_period'] == 0))
    print(f'  Desc length median:  {r["desc_len_median"]:.0f} words')
    print()

    if 'feature_hedge_count' in r:
        print(f'  Feature hedges:      {r["feature_hedge_count"]:,}')
        print(f'  Feature-only entities (rescued): {r.get("feature_only_entities", 0):,}')
        print(f'  Feature types ({len(r["feature_types"])}):')
        for ft, c in sorted(r['feature_types'].items(), key=lambda x: -x[1]):
            print(f'    {ft:<35}: {c:,}')
        print()


# ---------------------------------------------------------------------------
# 2. Embedding Quality
# ---------------------------------------------------------------------------

def validate_embedding(hg: dict, index_dir: str, n_samples: int = 200) -> dict:
    """Validate embedding self-consistency (needs model + FAISS)."""
    import faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('abhinand/MedEmbed-large-v0.1')
    idx = faiss.read_index(f'{index_dir}/index_hyperedge.bin')
    hedge_ids = np.load(f'{index_dir}/hedge_ids.npy', allow_pickle=True).tolist()

    results = {}

    # 2a. Anchor-Description coherence
    pairs = []
    for h in random.sample(hg['hyperedges'], min(n_samples, len(hg['hyperedges']))):
        if h.get('anchor'):
            pairs.append((h['anchor'], h['description']))
    if pairs:
        emb_a = model.encode([p[0] for p in pairs], normalize_embeddings=True)
        emb_d = model.encode([p[1] for p in pairs], normalize_embeddings=True)
        cos = np.sum(emb_a * emb_d, axis=1)
        results['anchor_desc_mean'] = float(cos.mean())
        results['anchor_desc_p10'] = float(np.percentile(cos, 10))
        results['anchor_desc_min'] = float(cos.min())
        results['anchor_desc_n'] = len(pairs)

    # 2b. Entity self-recall@5
    test_entities = random.sample(
        list(hg['entity_to_hedges'].keys()), min(n_samples, len(hg['entity_to_hedges']))
    )
    hits = 0
    for ename in test_entities:
        gold = set(hg['entity_to_hedges'][ename])
        q_emb = model.encode([ename], normalize_embeddings=True).astype(np.float32)
        _, ids = idx.search(q_emb, 5)
        retrieved = {hedge_ids[i] for i in ids[0] if i < len(hedge_ids)}
        if gold & retrieved:
            hits += 1
    results['entity_self_recall_at5'] = hits / len(test_entities)
    results['entity_self_recall_n'] = len(test_entities)

    # 2c. Negative separation
    neg_queries = [
        f'{topic} treatment guidelines'
        for topic in [
            'xylophone', 'bicycle', 'abstract algebra', 'tennis racket',
            'cryptocurrency', 'quantum entanglement', 'oil painting',
            'medieval architecture', 'chess opening', 'jazz improvisation',
            'volcanic eruption', 'deep sea fishing', 'furniture design',
            'smartphone battery', 'train schedule', 'baking sourdough',
            'stock market', 'video game', 'photography tips', 'gardening',
        ]
    ] + [
        'How to cook pasta',
        'Best movies of 2024',
        'Python programming tutorial',
        'History of the Roman Empire',
        'Climate change policy',
    ]
    neg_scores = []
    for q in neg_queries:
        q_emb = model.encode([q], normalize_embeddings=True).astype(np.float32)
        scores, _ = idx.search(q_emb, 1)
        neg_scores.append(float(scores[0][0]))
    results['neg_mean_cosine'] = float(np.mean(neg_scores))
    results['neg_max_cosine'] = float(np.max(neg_scores))
    results['neg_frac_above_070'] = sum(1 for s in neg_scores if s > 0.7) / len(neg_scores)
    results['neg_n'] = len(neg_queries)

    return results, model, idx, hedge_ids


def print_embedding(r: dict):
    print('=' * 60)
    print('2. EMBEDDING QUALITY')
    print('=' * 60)

    if 'anchor_desc_mean' in r:
        print(f'  Anchor-Desc cosine (n={r["anchor_desc_n"]}):')
        print(f'    Mean: {r["anchor_desc_mean"]:.3f}', _pass(r['anchor_desc_mean'] >= 0.75))
        print(f'    P10:  {r["anchor_desc_p10"]:.3f}', _pass(r['anchor_desc_p10'] >= 0.65))
        print(f'    Min:  {r["anchor_desc_min"]:.3f}')
    print()

    print(f'  Entity self-recall@5 (n={r["entity_self_recall_n"]}):')
    print(f'    {r["entity_self_recall_at5"]:.1%}', _pass(r['entity_self_recall_at5'] >= 0.75))
    print()

    print(f'  Negative separation (n={r["neg_n"]}):')
    print(f'    Mean cosine: {r["neg_mean_cosine"]:.3f}', _pass(r['neg_mean_cosine'] < 0.68))
    print(f'    Max cosine:  {r["neg_max_cosine"]:.3f}', _pass(r['neg_max_cosine'] < 0.72))
    print(f'    Frac > 0.7:  {r["neg_frac_above_070"]:.0%}', _pass(r['neg_frac_above_070'] == 0))
    print()


# ---------------------------------------------------------------------------
# 3. Extrinsic — Retrieval on Medical QA Benchmarks
# ---------------------------------------------------------------------------

BENCHMARK_CONFIGS = {
    'MedQA': {
        'path': 'dataset/MedQA',
        'split': 'test',
        'get_answer': lambda ex: ex['answer'],
        'semantic_target': 0.70,
    },
    'MedMCQA': {
        'path': 'dataset/MedMCQA_4options',
        'split': 'test',
        'get_answer': lambda ex: ex['options'].get(ex['answer_idx'], ''),
        'semantic_target': 0.60,
    },
    'MedXpertQA': {
        'path': 'dataset/MedXpertQA_Text',
        'split': 'test',
        'get_answer': lambda ex: _parse_options(ex).get(ex.get('label', ''), ''),
        'semantic_target': 0.50,
    },
    'PubMedQA': {
        'path': 'dataset/PubMedQA',
        'split': 'train',
        'get_answer': lambda ex: ex.get('final_decision', ''),
        'semantic_target': 0.50,
    },
}


def _parse_options(ex):
    opts = ex.get('options', {})
    if isinstance(opts, str):
        try:
            opts = ast.literal_eval(opts)
        except Exception:
            opts = {}
    return opts


def validate_extrinsic(
    hg: dict,
    model,
    idx,
    hedge_ids: list,
    benchmarks: list[str] | None = None,
    n_samples: int = 500,
) -> dict:
    """Evaluate retrieval quality on medical QA benchmarks."""
    from datasets import load_from_disk

    hedge_by_id = {h['id']: h for h in hg['hyperedges']}
    entity_names_lower = {e['name'].lower() for e in hg['entities'].values()}

    results = {}

    configs = BENCHMARK_CONFIGS
    if benchmarks:
        # Map paths back to config names, or create ad-hoc configs
        configs = {}
        for b in benchmarks:
            for name, cfg in BENCHMARK_CONFIGS.items():
                if cfg['path'] in b or name.lower() in b.lower():
                    configs[name] = cfg
                    break

    for name, cfg in configs.items():
        try:
            ds = load_from_disk(cfg['path'])
            if isinstance(ds, dict) or hasattr(ds, 'keys'):
                ds = ds[cfg['split']]
        except Exception as e:
            print(f'  Skipping {name}: {e}')
            continue

        samples = list(ds)[:n_samples]
        top1_scores = []
        entity_recall_hits = 0
        kg_answerable = 0
        intra_sims = []

        for ex in samples:
            question = ex.get('question', '')
            answer = str(cfg['get_answer'](ex)).lower().strip()

            q_emb = model.encode([question], normalize_embeddings=True).astype(np.float32)

            # Top-k retrieval
            scores, ids = idx.search(q_emb, 10)
            top1_scores.append(float(scores[0][0]))

            # Entity recall (only for KG-answerable)
            answer_in_kg = any(answer in e for e in entity_names_lower) or any(
                e in answer for e in entity_names_lower if len(e) > 5
            )
            if answer_in_kg and answer:
                kg_answerable += 1
                for i in ids[0]:
                    if i >= len(hedge_ids):
                        continue
                    he = hedge_by_id.get(hedge_ids[i])
                    if he and any(
                        answer in ent.lower() or ent.lower() in answer
                        for ent in he['entities']
                    ):
                        entity_recall_hits += 1
                        break

            # Intra-top5 diversity
            top5_descs = []
            for i in ids[0][:5]:
                if i < len(hedge_ids):
                    he = hedge_by_id.get(hedge_ids[i])
                    if he:
                        top5_descs.append(he['description'])
            if len(top5_descs) >= 2:
                embs = model.encode(top5_descs, normalize_embeddings=True)
                sim_matrix = embs @ embs.T
                # Mean of upper triangle (excluding diagonal)
                n = len(top5_descs)
                upper = [sim_matrix[i][j] for i in range(n) for j in range(i + 1, n)]
                intra_sims.append(float(np.mean(upper)))

        scores_arr = np.array(top1_scores)
        bench_results = {
            'n_samples': len(samples),
            'semantic_relevance_070': float(np.mean(scores_arr > 0.7)),
            'semantic_target': cfg['semantic_target'],
            'top1_cosine_mean': float(scores_arr.mean()),
            'top1_cosine_median': float(np.median(scores_arr)),
            'top1_cosine_p25': float(np.percentile(scores_arr, 25)),
            'top1_cosine_p75': float(np.percentile(scores_arr, 75)),
            'kg_answerable': kg_answerable,
            'entity_recall_at10': entity_recall_hits / max(kg_answerable, 1),
            'intra_top5_sim_mean': float(np.mean(intra_sims)) if intra_sims else 0.0,
        }
        results[name] = bench_results

    # Random baseline (on first benchmark)
    if results:
        first_name = next(iter(results))
        first_cfg = configs[first_name]
        ds = load_from_disk(first_cfg['path'])
        if isinstance(ds, dict) or hasattr(ds, 'keys'):
            ds = ds[first_cfg['split']]
        all_descs = [h['description'] for h in hg['hyperedges']]
        rand_hits = 0
        samples = list(ds)[:n_samples]
        for ex in samples:
            answer = str(first_cfg['get_answer'](ex)).lower().strip()
            rand_descs = random.sample(all_descs, 10)
            if any(answer in d.lower() for d in rand_descs):
                rand_hits += 1
        results['_random_baseline'] = {
            'benchmark': first_name,
            'entity_recall_at10': rand_hits / len(samples),
        }

    return results


def print_extrinsic(r: dict):
    print('=' * 60)
    print('3. EXTRINSIC — Retrieval on Medical QA')
    print('=' * 60)

    for name, br in r.items():
        if name.startswith('_'):
            continue
        target = br['semantic_target']
        sr = br['semantic_relevance_070']
        print(f'\n  {name} (n={br["n_samples"]}):')
        print(f'    Semantic relevance@0.7: {sr:.1%}', _pass(sr >= target))
        print(f'    Top-1 cosine mean:      {br["top1_cosine_mean"]:.3f}')
        print(f'    Top-1 cosine [P25-P75]: [{br["top1_cosine_p25"]:.3f} - {br["top1_cosine_p75"]:.3f}]')
        print(f'    KG-answerable questions: {br["kg_answerable"]}')
        print(f'    Entity recall@10 (KG):  {br["entity_recall_at10"]:.1%}',
              _pass(br['entity_recall_at10'] >= 0.15) if br['kg_answerable'] > 0 else '')
        print(f'    Intra-top5 similarity:  {br["intra_top5_sim_mean"]:.3f}',
              _pass(br['intra_top5_sim_mean'] < 0.85))

    if '_random_baseline' in r:
        rb = r['_random_baseline']
        print(f'\n  Random baseline ({rb["benchmark"]}):')
        print(f'    Entity recall@10: {rb["entity_recall_at10"]:.1%}')
    print()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _pass(cond: bool) -> str:
    return '[PASS]' if cond else '[FAIL]'


def summarize(intrinsic: dict, embedding: dict | None, extrinsic: dict | None) -> dict:
    """Produce pass/fail summary."""
    checks = []

    checks.append(('Orphan entities == 0', intrinsic['orphan_entities'] == 0))
    checks.append(('Placeholder leaks == 0', intrinsic['placeholder_leaks'] == 0))
    checks.append(('Empty descriptions == 0', intrinsic['empty_descriptions'] == 0))
    checks.append(('Duplicate descs < 0.1%', intrinsic['duplicate_pct'] < 0.001))

    if embedding:
        checks.append(('Anchor-desc cosine >= 0.75', embedding.get('anchor_desc_mean', 0) >= 0.75))
        checks.append(('Entity self-recall@5 >= 75%', embedding.get('entity_self_recall_at5', 0) >= 0.75))
        checks.append(('Negative max cosine < 0.72', embedding.get('neg_max_cosine', 1) < 0.72))

    if extrinsic:
        for name, br in extrinsic.items():
            if name.startswith('_'):
                continue
            target = br['semantic_target']
            checks.append((
                f'{name} semantic@0.7 >= {target:.0%}',
                br['semantic_relevance_070'] >= target,
            ))

    return {
        'total': len(checks),
        'passed': sum(1 for _, ok in checks if ok),
        'failed': sum(1 for _, ok in checks if not ok),
        'details': [{'check': c, 'pass': ok} for c, ok in checks],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Validate Medical Hypergraph quality'
    )
    parser.add_argument('--hg-path', default='data/medical_hg.json')
    parser.add_argument('--index-dir', default='data/')
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--only-intrinsic', action='store_true')
    parser.add_argument('--benchmarks', nargs='+', default=None)
    parser.add_argument('--output', default='data/validation_report.json')
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    t0 = time.time()

    print(f'Loading hypergraph from {args.hg_path}...')
    with open(args.hg_path) as f:
        hg = json.load(f)
    print()

    # 1. Intrinsic
    intrinsic = validate_intrinsic(hg)
    print_intrinsic(intrinsic)

    embedding = None
    extrinsic = None

    if not args.only_intrinsic:
        # 2. Embedding
        embedding, model, idx, hedge_ids = validate_embedding(
            hg, args.index_dir, args.n_samples
        )
        print_embedding(embedding)

        # 3. Extrinsic
        extrinsic = validate_extrinsic(
            hg, model, idx, hedge_ids, args.benchmarks, args.n_samples
        )
        print_extrinsic(extrinsic)

    # Summary
    summary = summarize(intrinsic, embedding, extrinsic)
    print('=' * 60)
    print(f'SUMMARY: {summary["passed"]}/{summary["total"]} checks passed')
    print('=' * 60)
    for item in summary['details']:
        status = '[PASS]' if item['pass'] else '[FAIL]'
        print(f'  {status} {item["check"]}')

    elapsed = time.time() - t0
    print(f'\nCompleted in {elapsed:.0f}s')

    # Save report
    report = {
        'intrinsic': intrinsic,
        'embedding': embedding,
        'extrinsic': extrinsic,
        'summary': summary,
        'elapsed_seconds': elapsed,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'Report saved to {args.output}')


if __name__ == '__main__':
    main()
