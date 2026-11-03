import psycopg

from .config import settings
from .embedder import embed_query
from .models import SearchResult
from .sparse import SparseMode, to_search_text

# RRF computed entirely in Postgres: both legs as CTEs, fused with a FULL JOIN.
SQL_HYBRID = """
WITH sparse AS (
    SELECT chunk_id,
           row_number() OVER (ORDER BY ts_rank_cd(tsv, q, 32) DESC, chunk_id) AS rank
    FROM chunks, websearch_to_tsquery(%(cfg)s::regconfig, %(query)s) AS q
    WHERE tsv @@ q
    ORDER BY rank
    LIMIT %(pool)s
),
dense AS (
    SELECT chunk_id, row_number() OVER (ORDER BY dist) AS rank
    FROM (
        SELECT chunk_id, embedding <=> %(vec)s AS dist
        FROM chunks
        ORDER BY embedding <=> %(vec)s
        LIMIT %(pool)s
    ) d
),
fused AS (
    SELECT coalesce(s.chunk_id, d.chunk_id) AS chunk_id,
           coalesce(1.0 / (%(rrf_k)s + s.rank), 0)
         + coalesce(1.0 / (%(rrf_k)s + d.rank), 0) AS score,
           s.rank AS sparse_rank,
           d.rank AS dense_rank
    FROM sparse s FULL JOIN dense d USING (chunk_id)
)
SELECT f.*, c.plan, c.heading_path, c.content
FROM fused f JOIN chunks c USING (chunk_id)
ORDER BY f.score DESC, f.chunk_id
LIMIT %(k)s
"""


def sql_hybrid_search(
    conn: psycopg.Connection,
    query: str,
    k: int = 10,
    pool: int = 50,
    rrf_k: int = 60,
    sparse_mode: SparseMode = "or",
    ef_search: int = 100,
) -> list[SearchResult]:
    """Same RRF as hybrid_search(method='rrf'), but fused in one SQL round trip."""
    ef = max(ef_search, pool)  # HNSW returns at most ef_search rows
    conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef),))
    params = {
        "cfg": settings.fts_config,
        "query": to_search_text(query, sparse_mode),
        "vec": embed_query(query),
        "pool": pool,
        "rrf_k": rrf_k,
        "k": k,
    }
    rows = conn.execute(SQL_HYBRID, params).fetchall()
    return [
        SearchResult(
            chunk_id=r["chunk_id"],
            score=float(r["score"]),
            ranks={
                leg: r[f"{leg}_rank"]
                for leg in ("sparse", "dense")
                if r[f"{leg}_rank"] is not None
            },
            plan=r["plan"],
            heading_path=r["heading_path"],
            content=r["content"],
        )
        for r in rows
    ]
