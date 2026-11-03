from collections import defaultdict

from .models import FusedHit, Hit


def rrf(
    rankings: dict[str, list[Hit]],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[FusedHit]:
    """Reciprocal Rank Fusion. rankings maps a leg name to its ranked hits.

    score(doc) = sum over legs of  weight[leg] / (k + rank_in_leg)
    """
    weights = weights or {}
    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for leg, hits in rankings.items():
        w = weights.get(leg, 1.0)
        for hit in hits:
            scores[hit.chunk_id] += w / (k + hit.rank)
            ranks[hit.chunk_id][leg] = hit.rank
    fused = [FusedHit(cid, score, ranks[cid]) for cid, score in scores.items()]
    fused.sort(key=lambda f: (-f.score, f.chunk_id))
    return fused


def _minmax(hits: list[Hit]) -> dict[str, float]:
    """Scale a leg's scores to 0..1 using this query's min and max."""
    if not hits:
        return {}
    hi = max(h.score for h in hits)
    lo = min(h.score for h in hits)
    if hi == lo:
        return {h.chunk_id: 1.0 for h in hits}
    return {h.chunk_id: (h.score - lo) / (hi - lo) for h in hits}


def convex(
    rankings: dict[str, list[Hit]],
    alpha: float = 0.5,
) -> list[FusedHit]:
    """Convex combination of min-max normalised scores.

    score(doc) = alpha * dense_norm + (1 - alpha) * sparse_norm
    Expects legs named 'dense' and 'sparse'. A doc missing from a leg scores 0 there.
    """
    dense = _minmax(rankings.get("dense", []))
    sparse = _minmax(rankings.get("sparse", []))
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for leg, hits in rankings.items():
        for hit in hits:
            ranks[hit.chunk_id][leg] = hit.rank
    fused = [
        FusedHit(cid, alpha * dense.get(cid, 0.0) + (1 - alpha) * sparse.get(cid, 0.0), ranks[cid])
        for cid in set(dense) | set(sparse)
    ]
    fused.sort(key=lambda f: (-f.score, f.chunk_id))
    return fused
