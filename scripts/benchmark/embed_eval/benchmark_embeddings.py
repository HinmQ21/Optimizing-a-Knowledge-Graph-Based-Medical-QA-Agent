"""Benchmark multiple embedding models on medical knowledge retrieval.

Evaluates: hit@5, hit@10, hit@20, MRR@20, encoding speed
Caches encoded KG embeddings to disk to avoid re-encoding.

Usage:
    python benchmark_embeddings.py [--models all] [--eval_set eval/retrieval_eval_200.json]
    python benchmark_embeddings.py --models medembed,biolord,e5large
    python benchmark_embeddings.py --list_models
    python benchmark_embeddings.py --device cuda   # GPU (default, DGX shared memory)
    python benchmark_embeddings.py --device cpu    # force CPU
"""

import argparse
import json
import time
from pathlib import Path
import sys
import re

import faiss
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[3]
CACHE_DIR = BASE_DIR / "eval" / "embed_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "medembed": {
        "label": "MedEmbed-large-v0.1 (baseline)",
        "hf_id": "abhinand/MedEmbed-large-v0.1",
        "type": "symmetric",
        "prefix_query": "",
        "prefix_doc": "",
        "batch_size": 512,
    },
    "biolord": {
        "label": "BioLORD-2023",
        "hf_id": "FremyCompany/BioLORD-2023",
        "type": "symmetric",
        "prefix_query": "",
        "prefix_doc": "",
        "batch_size": 1024,
    },
    "e5large": {
        "label": "E5-large-v2",
        "hf_id": "intfloat/e5-large-v2",
        "type": "symmetric",
        "prefix_query": "query: ",
        "prefix_doc": "passage: ",
        "batch_size": 512,
    },
    "pubmedbert_nli": {
        "label": "PubMedBERT-NLI-STS",
        "hf_id": "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb",
        "type": "symmetric",
        "prefix_query": "",
        "prefix_doc": "",
        "batch_size": 1024,
    },
    "medcpt": {
        "label": "MedCPT (asymmetric)",
        "hf_id_query": "ncbi/MedCPT-Query-Encoder",
        "hf_id_doc": "ncbi/MedCPT-Article-Encoder",
        "type": "asymmetric",
        "batch_size": 512,
    },
    "gte_large": {
        "label": "GTE-large-en-v1.5",
        "hf_id": "thenlper/gte-large",
        "type": "symmetric",
        "prefix_query": "",
        "prefix_doc": "",
        "batch_size": 512,
    },
    "biomedbert": {
        "label": "BiomedNLP-BiomedBERT-large",
        "hf_id": "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
        "type": "symmetric",
        "prefix_query": "",
        "prefix_doc": "",
        "batch_size": 512,
    },
    "bge_m3": {
        "label": "BGE-M3 (dense+sparse+colbert)",
        "hf_id": "BAAI/bge-m3",
        "type": "bge_m3",
        "batch_size": 512,
        # Fusion weights: dense + sparse + colbert
        "w_dense": 1.0,
        "w_sparse": 1.0,
        "w_colbert": 1.0,
    },
}


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def encode_symmetric(model, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
    """Encode texts using a sentence-transformers symmetric model."""
    if prefix:
        texts = [prefix + t for t in texts]
    embs = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embs.astype(np.float32)


def encode_asymmetric_medcpt(
    texts: list[str], model_id: str, batch_size: int = 512, device: str = "cuda"
) -> np.ndarray:
    """Encode texts with MedCPT (uses raw transformers, CLS pooling)."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    print(f"  Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    model.to(device)

    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if (i // batch_size) % 20 == 0:
            print(f"    [{i}/{len(texts)}]", flush=True)
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        # CLS token, move back to CPU for numpy
        cls = out.last_hidden_state[:, 0, :].cpu().float().numpy()
        # Normalize
        norms = np.linalg.norm(cls, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        cls = cls / norms
        all_embs.append(cls.astype(np.float32))

    return np.vstack(all_embs)


# ---------------------------------------------------------------------------
# BGE-M3 multi-granularity encoder
# ---------------------------------------------------------------------------

class BGEM3Encoder:
    """Wraps FlagEmbedding BGEM3FlagModel for dense+sparse+colbert retrieval.

    Strategy:
      Stage 1 — dense FAISS retrieval → top_k * expand candidates
      Stage 2 — re-score candidates with sparse + colbert
      Final   — weighted fusion: w_dense*dense + w_sparse*sparse + w_colbert*colbert
    """

    def __init__(self, device: str = "cuda", batch_size: int = 512):
        from FlagEmbedding import BGEM3FlagModel
        self.model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=device == "cuda",
            device=device,
        )
        self.batch_size = batch_size
        self.device = device

    def encode_corpus(self, texts: list[str]) -> dict:
        """Encode all corpus docs. Returns dict with dense, sparse, colbert."""
        print(f"  [BGE-M3] Encoding {len(texts)} docs (dense+sparse+colbert) ...")
        out = self.model.encode(
            texts,
            batch_size=self.batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        return {
            "dense": out["dense_vecs"].astype(np.float32),          # (N, 1024)
            "sparse": out["lexical_weights"],                         # list of dicts
            "colbert": out["colbert_vecs"],                           # list of (seq_len, 1024)
        }

    def encode_queries(self, texts: list[str]) -> dict:
        """Encode queries. Returns dict with dense, sparse, colbert."""
        print(f"  [BGE-M3] Encoding {len(texts)} queries ...")
        out = self.model.encode(
            texts,
            batch_size=self.batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        return {
            "dense": out["dense_vecs"].astype(np.float32),
            "sparse": out["lexical_weights"],
            "colbert": out["colbert_vecs"],
        }

    @staticmethod
    def sparse_score(q_sparse: dict, d_sparse: dict) -> float:
        """Dot product between two sparse token weight dicts."""
        score = 0.0
        for tok_id, w in q_sparse.items():
            if tok_id in d_sparse:
                score += float(w) * float(d_sparse[tok_id])
        return score

    @staticmethod
    def colbert_score(q_vecs: np.ndarray, d_vecs: np.ndarray) -> float:
        """MaxSim ColBERT score: mean over query tokens of max similarity to doc tokens."""
        # q_vecs: (q_len, dim), d_vecs: (d_len, dim)
        sims = q_vecs @ d_vecs.T           # (q_len, d_len)
        max_sims = sims.max(axis=1)        # (q_len,)
        return float(max_sims.mean())


def run_bge_m3_eval(
    encoder: "BGEM3Encoder",
    corpus: dict,
    queries: dict,
    samples: list[dict],
    top_k: int = 20,
    cfg: dict = None,
    expand: int = 4,
) -> dict:
    """Full BGE-M3 retrieval: dense→rerank with sparse+colbert."""
    w_dense  = cfg.get("w_dense",  1.0) if cfg else 1.0
    w_sparse = cfg.get("w_sparse", 1.0) if cfg else 1.0
    w_colbert = cfg.get("w_colbert", 1.0) if cfg else 1.0
    w_sum = w_dense + w_sparse + w_colbert

    # Build dense FAISS index
    dense_embs = corpus["dense"]
    dim = dense_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(dense_embs)

    n_candidates = min(top_k * expand, len(dense_embs))

    hits_at_5 = hits_at_10 = hits_at_20 = 0
    mrr = 0.0
    per_sample = []

    for i, sample in enumerate(samples):
        q_dense = queries["dense"][i : i + 1]
        q_sparse = queries["sparse"][i]
        q_colbert = queries["colbert"][i]  # (q_len, dim)

        # Stage 1: dense retrieve candidates
        dense_scores, cand_ids = index.search(q_dense, n_candidates)
        cand_ids = [c for c in cand_ids[0] if c >= 0]
        dense_scores_map = {
            cand_ids[j]: float(dense_scores[0][j]) for j in range(len(cand_ids))
        }

        # Stage 2: fuse with sparse + colbert
        fused = []
        for cid in cand_ids:
            d_dense  = dense_scores_map[cid]
            d_sparse = encoder.sparse_score(q_sparse, corpus["sparse"][cid])
            d_colbert = encoder.colbert_score(q_colbert, corpus["colbert"][cid])
            score = (w_dense * d_dense + w_sparse * d_sparse + w_colbert * d_colbert) / w_sum
            fused.append((score, cid))

        fused.sort(key=lambda x: x[0], reverse=True)
        ranked_ids = [cid for _, cid in fused[:top_k]]

        # Evaluate
        answer = sample["answer"].lower()
        first_hit = None
        for rank, cid in enumerate(ranked_ids):
            desc = corpus.get("descriptions", [])[cid] if "descriptions" in corpus else ""
            if match_score(answer, desc) > 0:
                first_hit = rank
                break

        hits_at_5  += int(first_hit is not None and first_hit < 5)
        hits_at_10 += int(first_hit is not None and first_hit < 10)
        hits_at_20 += int(first_hit is not None and first_hit < 20)
        if first_hit is not None:
            mrr += 1.0 / (first_hit + 1)

        per_sample.append({
            "idx": sample.get("idx", i),
            "answer": sample["answer"],
            "first_hit_rank": first_hit,
            "hit5": first_hit is not None and first_hit < 5,
            "hit10": first_hit is not None and first_hit < 10,
        })

    n = len(samples)
    return {
        "hit@5":  round(hits_at_5 / n, 4),
        "hit@10": round(hits_at_10 / n, 4),
        "hit@20": round(hits_at_20 / n, 4),
        "MRR@20": round(mrr / n, 4),
        "n": n,
        "per_sample": per_sample,
    }


# ---------------------------------------------------------------------------
# KG loading
# ---------------------------------------------------------------------------

def load_kg_descriptions(data_dir: Path) -> tuple[list[str], list[str]]:
    """Returns (hedge_ids, descriptions)."""
    import json
    with open(data_dir / "medical_hg.json") as f:
        hg = json.load(f)
    hedge_ids = [h["id"] for h in hg["hyperedges"]]
    descriptions = [h["description"] for h in hg["hyperedges"]]
    return hedge_ids, descriptions


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def cache_path(model_key: str, role: str) -> Path:
    return CACHE_DIR / f"{model_key}_{role}.npy"


def load_or_encode_docs(
    model_key: str, cfg: dict, descriptions: list[str], device: str = "cuda"
) -> np.ndarray:
    """For bge_m3 type, returns None (handled separately via load_or_encode_bge_m3)."""
    if cfg["type"] == "bge_m3":
        return None  # handled separately

    cp = cache_path(model_key, "docs")
    if cp.exists():
        print(f"  [cache hit] Loading doc embeddings from {cp}")
        return np.load(str(cp))

    print(f"  Encoding {len(descriptions)} KG descriptions on {device} ...")
    t0 = time.time()

    if cfg["type"] == "symmetric":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(cfg["hf_id"], device=device)
        embs = encode_symmetric(model, descriptions, cfg["prefix_doc"], cfg["batch_size"])
        del model
    else:  # asymmetric MedCPT
        embs = encode_asymmetric_medcpt(
            descriptions, cfg["hf_id_doc"], cfg["batch_size"], device=device
        )

    elapsed = time.time() - t0
    print(f"  Encoded {len(descriptions)} docs in {elapsed:.1f}s "
          f"({len(descriptions)/elapsed:.0f} docs/s), dim={embs.shape[1]}")
    np.save(str(cp), embs)
    return embs


def load_or_encode_bge_m3(
    model_key: str, cfg: dict, descriptions: list[str], questions: list[str],
    device: str = "cuda"
) -> tuple[dict, dict, float]:
    """Encode corpus + queries with BGE-M3 (all 3 modes). Returns (corpus, queries, q_time)."""
    cp_dense = cache_path(model_key, "docs")          # dense vectors
    cp_sparse = CACHE_DIR / f"{model_key}_sparse.npy"  # sparse dicts
    cp_colbert = CACHE_DIR / f"{model_key}_colbert.npy" # colbert vecs list

    encoder = BGEM3Encoder(device=device, batch_size=cfg["batch_size"])

    if cp_dense.exists() and cp_sparse.exists() and cp_colbert.exists():
        print("  [cache hit] Loading BGE-M3 doc embeddings ...")
        corpus = {
            "dense": np.load(str(cp_dense)),
            "sparse": np.load(str(cp_sparse), allow_pickle=True).tolist(),
            "colbert": np.load(str(cp_colbert), allow_pickle=True).tolist(),
            "descriptions": descriptions,
        }
    else:
        t0 = time.time()
        raw = encoder.encode_corpus(descriptions)
        elapsed = time.time() - t0
        print(f"  BGE-M3 corpus encoded in {elapsed:.1f}s "
              f"({len(descriptions)/elapsed:.0f} docs/s), dim={raw['dense'].shape[1]}")
        np.save(str(cp_dense), raw["dense"])
        np.save(str(cp_sparse), np.array(raw["sparse"], dtype=object), allow_pickle=True)
        np.save(str(cp_colbert), np.array(raw["colbert"], dtype=object), allow_pickle=True)
        corpus = {**raw, "descriptions": descriptions}

    # Queries (not cached — fast)
    t0 = time.time()
    queries = encoder.encode_queries(questions)
    q_time = time.time() - t0

    return corpus, queries, q_time


def encode_queries(
    model_key: str, cfg: dict, questions: list[str], device: str = "cuda"
) -> tuple[np.ndarray, float]:
    """Returns (query_embs, encode_time). Not used for bge_m3."""
    t0 = time.time()

    if cfg["type"] == "symmetric":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(cfg["hf_id"], device=device)
        embs = encode_symmetric(model, questions, cfg["prefix_query"], cfg["batch_size"])
        del model
    else:  # MedCPT
        embs = encode_asymmetric_medcpt(
            questions, cfg["hf_id_query"], cfg["batch_size"], device=device
        )

    return embs, time.time() - t0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def match_score(answer: str, description: str) -> float:
    """Check if answer appears in description. Returns [0, 1]."""
    ans = answer.lower().strip()
    desc = description.lower()

    # Exact substring match
    if ans in desc:
        return 1.0

    # Token-level match (at least 60% of answer tokens in desc)
    ans_tokens = set(re.findall(r"[a-z0-9]+", ans))
    desc_tokens = set(re.findall(r"[a-z0-9]+", desc))
    if len(ans_tokens) >= 2:
        overlap = len(ans_tokens & desc_tokens) / len(ans_tokens)
        if overlap >= 0.6:
            return overlap
    return 0.0


def evaluate_retrieval(
    query_embs: np.ndarray,
    doc_embs: np.ndarray,
    descriptions: list[str],
    samples: list[dict],
    top_k: int = 20,
) -> dict:
    """Build FAISS index and evaluate retrieval metrics."""
    dim = doc_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embs)

    hits_at_5 = 0
    hits_at_10 = 0
    hits_at_20 = 0
    mrr = 0.0

    per_sample = []

    for i, sample in enumerate(samples):
        q_emb = query_embs[i : i + 1]
        scores, ids = index.search(q_emb, top_k)
        retrieved = [(float(scores[0][j]), descriptions[ids[0][j]])
                     for j in range(top_k) if ids[0][j] >= 0]

        answer = sample["answer"]
        first_hit_rank = None
        for rank, (score, desc) in enumerate(retrieved):
            if match_score(answer, desc) > 0:
                first_hit_rank = rank
                break

        hit5 = first_hit_rank is not None and first_hit_rank < 5
        hit10 = first_hit_rank is not None and first_hit_rank < 10
        hit20 = first_hit_rank is not None and first_hit_rank < 20

        hits_at_5 += int(hit5)
        hits_at_10 += int(hit10)
        hits_at_20 += int(hit20)
        if first_hit_rank is not None:
            mrr += 1.0 / (first_hit_rank + 1)

        per_sample.append({
            "idx": sample.get("idx", i),
            "answer": answer,
            "first_hit_rank": first_hit_rank,
            "hit5": hit5,
            "hit10": hit10,
        })

    n = len(samples)
    return {
        "hit@5":  round(hits_at_5 / n, 4),
        "hit@10": round(hits_at_10 / n, 4),
        "hit@20": round(hits_at_20 / n, 4),
        "MRR@20": round(mrr / n, 4),
        "n": n,
        "per_sample": per_sample,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_table(results: list[dict]) -> None:
    if not results:
        return
    header = f"{'Model':<42} {'hit@5':>6} {'hit@10':>7} {'hit@20':>7} {'MRR@20':>7} {'EncTime':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: x["metrics"]["hit@10"], reverse=True):
        m = r["metrics"]
        print(
            f"{r['label']:<42} "
            f"{m['hit@5']:>6.1%} "
            f"{m['hit@10']:>7.1%} "
            f"{m['hit@20']:>7.1%} "
            f"{m['MRR@20']:>7.4f} "
            f"{r['query_encode_time']:>7.1f}s"
        )
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="all",
        help=f"Comma-separated model keys, or 'all'. Available: {', '.join(MODELS)}",
    )
    parser.add_argument(
        "--eval_set",
        default=str(BASE_DIR / "eval/retrieval_eval_200.json"),
    )
    parser.add_argument(
        "--data_dir",
        default=str(BASE_DIR / "data"),
    )
    parser.add_argument(
        "--output",
        default=str(BASE_DIR / "eval/embedding_benchmark_results.json"),
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List available model keys and exit",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Retrieve top-K candidates for evaluation",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Ignore cached doc embeddings and re-encode",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for encoding: 'cuda' (default) or 'cpu'",
    )
    args = parser.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    print(f"Using device: {args.device}")

    if args.list_models:
        print("Available models:")
        for key, cfg in MODELS.items():
            print(f"  {key:<20} {cfg['label']}")
        return

    # Select models
    if args.models == "all":
        selected_keys = list(MODELS.keys())
    else:
        selected_keys = [k.strip() for k in args.models.split(",")]
        unknown = [k for k in selected_keys if k not in MODELS]
        if unknown:
            print(f"Unknown model keys: {unknown}")
            print(f"Available: {list(MODELS.keys())}")
            sys.exit(1)

    # Load eval set
    eval_path = Path(args.eval_set)
    if not eval_path.exists():
        print(f"Eval set not found: {eval_path}")
        print("Run prepare_retrieval_eval.py first.")
        sys.exit(1)

    with open(eval_path) as f:
        eval_data = json.load(f)
    samples = eval_data["samples"]
    questions = [s["question"] for s in samples]
    print(f"Loaded {len(samples)} eval samples from {eval_path}")

    # Load KG
    data_dir = Path(args.data_dir)
    print(f"Loading KG descriptions from {data_dir} ...")
    hedge_ids, descriptions = load_kg_descriptions(data_dir)
    print(f"  {len(descriptions)} hyperedge descriptions loaded")

    # Handle cache invalidation
    if args.no_cache:
        for f in CACHE_DIR.glob("*.npy"):
            f.unlink()
        print("  Cleared embedding cache")

    # Run benchmark
    all_results = []
    failed = []

    for model_key in selected_keys:
        cfg = MODELS[model_key]
        print(f"\n{'='*60}")
        print(f"Model: {cfg['label']}")
        print(f"{'='*60}")

        try:
            if cfg["type"] == "bge_m3":
                # BGE-M3 special path: encode+retrieve all 3 modes
                corpus, queries_enc, q_enc_time = load_or_encode_bge_m3(
                    model_key, cfg, descriptions, questions, device=args.device
                )
                print(f"  Query encode time: {q_enc_time:.2f}s "
                      f"({len(questions)/q_enc_time:.1f} q/s)")
                print(f"  Evaluating retrieval (top-{args.top_k}, dense→sparse+colbert rerank) ...")
                metrics = run_bge_m3_eval(
                    BGEM3Encoder(device=args.device, batch_size=cfg["batch_size"]),
                    corpus, queries_enc, samples, top_k=args.top_k, cfg=cfg,
                )
                dim = int(corpus["dense"].shape[1])
            else:
                # 1. Encode / load KG docs
                doc_embs = load_or_encode_docs(model_key, cfg, descriptions, device=args.device)

                # 2. Encode queries
                print(f"  Encoding {len(questions)} queries on {args.device} ...")
                query_embs, q_enc_time = encode_queries(model_key, cfg, questions, device=args.device)
                print(f"  Query encode time: {q_enc_time:.2f}s "
                      f"({len(questions)/q_enc_time:.1f} q/s)")

                # 3. Evaluate
                print(f"  Evaluating retrieval (top-{args.top_k}) ...")
                metrics = evaluate_retrieval(
                    query_embs, doc_embs, descriptions, samples, top_k=args.top_k
                )
                dim = int(doc_embs.shape[1])

            result = {
                "model_key": model_key,
                "label": cfg["label"],
                "hf_id": cfg.get("hf_id", cfg.get("hf_id_query", "")),
                "dim": dim,
                "n_docs": len(descriptions),
                "metrics": {k: v for k, v in metrics.items() if k != "per_sample"},
                "query_encode_time": round(q_enc_time, 2),
                "queries_per_sec": round(len(questions) / q_enc_time, 1),
            }
            all_results.append(result)
            print(
                f"  hit@5={metrics['hit@5']:.1%}  hit@10={metrics['hit@10']:.1%}  "
                f"hit@20={metrics['hit@20']:.1%}  MRR={metrics['MRR@20']:.4f}"
            )

            # Save per-sample details
            detail_path = CACHE_DIR / f"{model_key}_per_sample.json"
            with open(detail_path, "w") as f:
                json.dump(metrics["per_sample"], f, indent=2)

        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed.append({"model_key": model_key, "error": str(e)})
            continue

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "eval_set": str(eval_path),
                "n_samples": len(samples),
                "top_k": args.top_k,
                "results": all_results,
                "failed": failed,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved → {output_path}")

    # Print summary table
    print_table(all_results)

    if failed:
        print(f"\nFailed models: {[f['model_key'] for f in failed]}")


if __name__ == "__main__":
    main()
