import numpy as np
import psycopg

from .models import Hit

DENSE_SQL = """
SELECT chunk_id,
       1 - (embedding <=> %(vec)s) AS score
FROM chunks
ORDER BY embedding <=> %(vec)s
LIMIT %(k)s
"""


def dense_search(
    conn: psycopg.Connection,
    query_vec: np.ndarray,
    k: int = 50,
    ef_search: int = 100,
) -> list[Hit]:
    """pgvector HNSW leg. Returns up to k hits by cosine similarity, best first."""
    ef = max(ef_search, k)  # HNSW returns at most ef_search rows
    conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef),))
    rows = conn.execute(DENSE_SQL, {"vec": query_vec, "k": k}).fetchall()
    return [
        Hit(chunk_id=row["chunk_id"], score=float(row["score"]), rank=i)
        for i, row in enumerate(rows, start=1)
    ]
