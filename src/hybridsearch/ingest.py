import json
from pathlib import Path

from .db import get_conn
from .embedder import embed_texts

UPSERT_SQL = """
INSERT INTO chunks (chunk_id, doc_id, plan, heading_path, content, embedding)
VALUES (%(chunk_id)s, %(doc_id)s, %(plan)s, %(heading_path)s, %(content)s, %(embedding)s)
ON CONFLICT (chunk_id) DO UPDATE SET
    doc_id       = EXCLUDED.doc_id,
    plan         = EXCLUDED.plan,
    heading_path = EXCLUDED.heading_path,
    content      = EXCLUDED.content,
    embedding    = EXCLUDED.embedding
"""


def load_corpus(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def embedded_text(chunk: dict) -> str:
    """The text we embed: heading breadcrumb on top of the content (Article 7)."""
    heading = chunk.get("heading_path") or ""
    return f"{heading}\n{chunk['content']}" if heading else chunk["content"]


def ingest(path: Path) -> int:
    """Embed every chunk in the file and upsert it. Returns the number of rows written."""
    chunks = load_corpus(path)
    vectors = embed_texts([embedded_text(c) for c in chunks])

    rows = [
        {
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "plan": c.get("plan"),
            "heading_path": c.get("heading_path") or "",
            "content": c["content"],
            "embedding": vec,
        }
        for c, vec in zip(chunks, vectors, strict=True)
    ]

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(UPSERT_SQL, rows)
        conn.commit()

    return len(rows)
