from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .db import get_conn, init_schema
from .evaluate import METHODS, evaluate
from .hybrid import hybrid_search
from .ingest import ingest
from .sql_hybrid import sql_hybrid_search

app = typer.Typer(help="Hybrid search in Postgres: tsvector + pgvector + fusion.")
console = Console()


@app.command("init-db")
def init_db() -> None:
    """Create the extension, table, and indexes (drops the table if it exists)."""
    init_schema()
    console.print("[green]Schema created.[/green]")


@app.command("ingest")
def ingest_cmd(
    path: Path = typer.Argument(Path("data/corpus.json"), help="Corpus JSON file"),
) -> None:
    """Embed and upsert every chunk in the corpus file."""
    n = ingest(path)
    console.print(f"[green]Ingested {n} chunks from {path}.[/green]")


@app.command()
def search(
    query: str = typer.Argument(..., help="The question to search for"),
    k: int = typer.Option(10, help="Number of results to return"),
    pool: int = typer.Option(50, help="Candidates fetched per leg before fusion"),
    method: str = typer.Option("rrf", help="rrf | convex | sql"),
    alpha: float = typer.Option(0.5, help="Dense weight for convex fusion"),
    w_sparse: float = typer.Option(1.0, help="Sparse weight for RRF"),
    w_dense: float = typer.Option(1.0, help="Dense weight for RRF"),
    sparse_mode: str = typer.Option("or", help="and | or"),
    ef_search: int = typer.Option(100, help="HNSW ef_search for the dense leg"),
    explain: bool = typer.Option(False, help="Show the raw per-leg candidate lists"),
) -> None:
    """Run a hybrid search and print the fused results with per-leg ranks."""
    with get_conn() as conn:
        if method == "sql":
            results = sql_hybrid_search(
                conn, query, k=k, pool=pool, sparse_mode=sparse_mode, ef_search=ef_search
            )
            rankings = {}
        else:
            results, rankings = hybrid_search(
                conn,
                query,
                k=k,
                pool=pool,
                method=method,
                weights={"sparse": w_sparse, "dense": w_dense},
                alpha=alpha,
                sparse_mode=sparse_mode,
                ef_search=ef_search,
            )

    table = Table(title=f'"{query}"  ·  method={method}  pool={pool}  k={k}')
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    table.add_column("sparse", justify="right")
    table.add_column("dense", justify="right")
    table.add_column("chunk_id")
    table.add_column("heading / snippet", overflow="fold")
    for i, r in enumerate(results, start=1):
        table.add_row(
            str(i),
            f"{r.score:.4f}",
            str(r.ranks.get("sparse", "–")),
            str(r.ranks.get("dense", "–")),
            r.chunk_id,
            f"[dim]{r.heading_path}[/dim]\n{r.content[:110]}…",
        )
    console.print(table)

    if explain and rankings:
        for leg, hits in rankings.items():
            leg_table = Table(title=f"{leg} leg · top {len(hits)}")
            leg_table.add_column("rank", justify="right")
            leg_table.add_column("score", justify="right")
            leg_table.add_column("chunk_id")
            for h in hits[:15]:
                leg_table.add_row(str(h.rank), f"{h.score:.4f}", h.chunk_id)
            console.print(leg_table)


@app.command("eval")
def eval_cmd(
    golden: Path = typer.Option(Path("data/golden.json"), help="Golden set JSON"),
    k: int = typer.Option(5, help="k for recall@k"),
    pool: int = typer.Option(50, help="Candidates per leg before fusion"),
) -> None:
    """Compute recall@k and MRR for every method on the golden set."""
    with get_conn() as conn:
        rows = evaluate(conn, golden, k=k, pool=pool, methods=METHODS)

    table = Table(title=f"Golden set · k={k} · pool={pool}")
    table.add_column("method")
    table.add_column(f"recall@{k}", justify="right")
    table.add_column("MRR", justify="right")
    for row in rows:
        table.add_row(row["method"], f"{row[f'recall@{k}']:.3f}", f"{row['mrr']:.3f}")
    console.print(table)
