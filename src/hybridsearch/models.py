from dataclasses import dataclass, field


@dataclass(frozen=True)
class Hit:
    """One candidate from a single retrieval leg."""

    chunk_id: str
    score: float
    rank: int


@dataclass
class FusedHit:
    """One candidate after fusion, with the rank it held in each leg."""

    chunk_id: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A fused hit joined back to its chunk text, ready to show or log."""

    chunk_id: str
    score: float
    ranks: dict[str, int]
    plan: str | None
    heading_path: str
    content: str
