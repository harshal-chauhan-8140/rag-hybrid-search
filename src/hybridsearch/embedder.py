from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    """Load the embedding model once per process."""
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed many texts. Returns an array of shape (len(texts), dim), L2-normalised."""
    return _model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def embed_query(query: str) -> np.ndarray:
    """Embed one query. Returns a 1-D array of length dim."""
    return embed_texts([query])[0]
