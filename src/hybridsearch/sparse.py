import re
from typing import Literal

import psycopg

from .config import settings
from .models import Hit

SparseMode = Literal["and", "or"]

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")

SPARSE_SQL = """
SELECT chunk_id,
       ts_rank_cd(tsv, q, 32) AS score
FROM chunks,
     websearch_to_tsquery(%(cfg)s::regconfig, %(query)s) AS q
WHERE tsv @@ q
ORDER BY score DESC, chunk_id
LIMIT %(k)s
"""


def to_search_text(query: str, mode: SparseMode) -> str:
    """Turn a user query into the text we hand to websearch_to_tsquery.

    'and' mode passes the query through: every word is required.
    'or' mode joins words with ' or ' so any word can match.
    """
    if mode == "and":
        return query
    words = _WORD.findall(query)
    return " or ".join(words) if words else query


def sparse_search(
    conn: psycopg.Connection,
    query: str,
    k: int = 50,
    mode: SparseMode = "or",
) -> list[Hit]:
    """Postgres full-text search leg. Returns up to k hits, best first."""
    params = {"cfg": settings.fts_config, "query": to_search_text(query, mode), "k": k}
    rows = conn.execute(SPARSE_SQL, params).fetchall()
    return [
        Hit(chunk_id=row["chunk_id"], score=float(row["score"]), rank=i)
        for i, row in enumerate(rows, start=1)
    ]
