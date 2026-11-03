import json
from pathlib import Path

import psycopg

from .bm25_local import LocalBM25
from .dense import dense_search
from .embedder import embed_query
from .fusion import rrf
from .hybrid import hybrid_search
from .sparse import sparse_search

METHODS = [
    "sparse_and",
    "sparse_or",
    "bm25s",
    "dense",
    "rrf",
    "rrf_w_dense1.3",
    "convex_0.5",
    "convex_0.7",
    "rrf_bm25s_dense",
]


def load_golden(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """What fraction of the expected chunks appear in the top k."""
    return len(set(retrieved[:k]) & set(expected)) / len(expected)


def reciprocal_rank(retrieved: list[str], expected: list[str]) -> float:
    """1 / position of the first correct chunk; 0 if none found."""
    for i, cid in enumerate(retrieved, start=1):
        if cid in expected:
            return 1.0 / i
    return 0.0


def run_method(
    conn: psycopg.Connection,
    bm25: LocalBM25,
    method: str,
    query: str,
    k: int,
    pool: int,
) -> list[str]:
    """Return the ranked chunk IDs a method produces for one query."""
    if method == "sparse_and":
        return [h.chunk_id for h in sparse_search(conn, query, k=k, mode="and")]
    if method == "sparse_or":
        return [h.chunk_id for h in sparse_search(conn, query, k=k, mode="or")]
    if method == "bm25s":
        return [h.chunk_id for h in bm25.search(query, k=k)]
    if method == "dense":
        return [h.chunk_id for h in dense_search(conn, embed_query(query), k=k)]
    if method == "rrf":
        results, _ = hybrid_search(conn, query, k=k, pool=pool, method="rrf")
        return [r.chunk_id for r in results]
    if method == "rrf_w_dense1.3":
        results, _ = hybrid_search(
            conn, query, k=k, pool=pool, method="rrf", weights={"sparse": 0.7, "dense": 1.3}
        )
        return [r.chunk_id for r in results]
    if method == "convex_0.5":
        results, _ = hybrid_search(conn, query, k=k, pool=pool, method="convex", alpha=0.5)
        return [r.chunk_id for r in results]
    if method == "convex_0.7":
        results, _ = hybrid_search(conn, query, k=k, pool=pool, method="convex", alpha=0.7)
        return [r.chunk_id for r in results]
    if method == "rrf_bm25s_dense":
        rankings = {
            "sparse": bm25.search(query, k=pool),
            "dense": dense_search(conn, embed_query(query), k=pool),
        }
        return [f.chunk_id for f in rrf(rankings)[:k]]
    raise ValueError(f"unknown method: {method}")


def evaluate(
    conn: psycopg.Connection,
    golden_path: Path,
    k: int = 5,
    pool: int = 50,
    methods: list[str] | None = None,
) -> list[dict]:
    """Compute recall@k and MRR for each method. Returns one row per method."""
    golden = load_golden(golden_path)
    bm25 = LocalBM25.from_db(conn)
    rows = []
    for method in methods or METHODS:
        recalls, rrs = [], []
        for item in golden:
            retrieved = run_method(conn, bm25, method, item["question"], k, pool)
            recalls.append(recall_at_k(retrieved, item["expected"], k))
            rrs.append(reciprocal_rank(retrieved, item["expected"]))
        rows.append(
            {
                "method": method,
                f"recall@{k}": sum(recalls) / len(recalls),
                "mrr": sum(rrs) / len(rrs),
            }
        )
    return rows
