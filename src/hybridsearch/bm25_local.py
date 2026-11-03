import bm25s
import psycopg
import Stemmer

from .models import Hit


class LocalBM25:
    """An in-process BM25 index over the chunks table, built with bm25s.

    This is the 'real BM25' alternative to Postgres full-text search.
    The index lives in memory; rebuild it when the table changes.
    """

    def __init__(self, ids: list[str], texts: list[str]):
        self.ids = ids
        self.stemmer = Stemmer.Stemmer("english")
        tokens = bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer)
        self.index = bm25s.BM25()  # defaults: k1=1.5, b=0.75
        self.index.index(tokens)

    @classmethod
    def from_db(cls, conn: psycopg.Connection) -> "LocalBM25":
        rows = conn.execute(
            "SELECT chunk_id, heading_path || E'\\n' || content AS text FROM chunks ORDER BY chunk_id"
        ).fetchall()
        return cls([r["chunk_id"] for r in rows], [r["text"] for r in rows])

    def search(self, query: str, k: int = 50) -> list[Hit]:
        q_tokens = bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer)
        n = min(k, len(self.ids))
        doc_idx, scores = self.index.retrieve(q_tokens, k=n)
        hits = []
        rank = 1
        for idx, score in zip(doc_idx[0], scores[0], strict=True):
            if score <= 0:
                break  # bm25s pads with zero-score docs when fewer match
            hits.append(Hit(chunk_id=self.ids[int(idx)], score=float(score), rank=rank))
            rank += 1
        return hits
