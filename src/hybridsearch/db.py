from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from .config import settings

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    """Open a connection with pgvector types registered and dict rows."""
    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        register_vector(conn)
        yield conn


def init_schema() -> None:
    """Create the extension, table, and indexes. Drops the table if it exists."""
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with psycopg.connect(settings.database_url) as conn:
        conn.execute(sql)
        conn.commit()
