from typing import Literal

import psycopg

from .dense import dense_search
from .embedder import embed_query
from .fusion import convex, rrf
from .models import FusedHit, Hit, SearchResult
from .sparse import SparseMode, sparse_search

FusionMethod = Literal["rrf", "convex"]

FETCH_SQL = """
SELECT chunk_id, plan, heading_path, content
FROM chunks
WHERE chunk_id = ANY(%(ids)s)
"""


def fetch_chunks(conn: psycopg.Connection, ids: list[str]) -> dict[str, dict]:
    """Load the text for a list of chunk IDs, keyed by ID."""
    if not ids:
        return {}
    rows = conn.execute(FETCH_SQL, {"ids": ids}).fetchall()
    return {row["chunk_id"]: row for row in rows}


def attach_text(conn: psycopg.Connection, fused: list[FusedHit]) -> list[SearchResult]:
    """Join fused hits to their chunk rows, preserving fused order."""
    chunks = fetch_chunks(conn, [f.chunk_id for f in fused])
    return [
        SearchResult(
            chunk_id=f.chunk_id,
            score=f.score,
            ranks=f.ranks,
            plan=chunks[f.chunk_id]["plan"],
            heading_path=chunks[f.chunk_id]["heading_path"],
            content=chunks[f.chunk_id]["content"],
        )
        for f in fused
        if f.chunk_id in chunks
    ]


def hybrid_search(
    conn: psycopg.Connection,
    query: str,
    k: int = 10,
    pool: int = 50,
    method: FusionMethod = "rrf",
    rrf_k: int = 60,
    weights: dict[str, float] | None = None,
    alpha: float = 0.5,
    sparse_mode: SparseMode = "or",
    ef_search: int = 100,
) -> tuple[list[SearchResult], dict[str, list[Hit]]]:
    """Run both legs, fuse, and return (top-k results, raw candidate lists per leg).

    The raw lists are returned so callers can log every candidate (Article 15).
    """
    query_vec = embed_query(query)
    rankings: dict[str, list[Hit]] = {
        "sparse": sparse_search(conn, query, k=pool, mode=sparse_mode),
        "dense": dense_search(conn, query_vec, k=pool, ef_search=ef_search),
    }

    if method == "rrf":
        fused = rrf(rankings, k=rrf_k, weights=weights)
    else:
        fused = convex(rankings, alpha=alpha)

    return attach_text(conn, fused[:k]), rankings
