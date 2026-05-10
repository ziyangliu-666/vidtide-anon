"""Text → 384-dim vector embedder for caption-based dedup.

Default: `sentence-transformers/all-MiniLM-L6-v2` — 22MB on disk, runs
CPU-fast, widely-used baseline for sentence similarity. 384 dims matches
the `vec_thumbnails.embedding FLOAT[384]` schema in vec_index.py.

Embeddings are L2-normalized on return so the downstream cosine distance
search behaves as expected (cosine distance = 1 - dot product on unit
vectors, which is what `distance_metric=cosine` computes).

Like the captioner, this is lazy-loaded — constructing an instance is
cheap, and the model only loads on first `embed()` call.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Contract: take a text string, return a fixed-dim unit-length vector."""

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Default: loop. Override for real batched inference."""
        return [self.embed(t) for t in texts]


class MiniLMEmbedder(BaseEmbedder):
    """sentence-transformers/all-MiniLM-L6-v2 — 384-dim unit-normalized."""

    name = "all-MiniLM-L6-v2"
    dim = 384

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_id = model_id
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer  # lazy

        logger.info("MiniLMEmbedder: loading %s", self.model_id)
        self._model = SentenceTransformer(self.model_id)
        logger.info("MiniLMEmbedder: loaded")

    def embed(self, text: str) -> list[float]:
        self._ensure_loaded()
        vec = self._model.encode(
            text, normalize_embeddings=True, convert_to_numpy=True
        )
        return vec.astype("float32").tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        vecs = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=32
        )
        return [v.astype("float32").tolist() for v in vecs]


def get_embedder(name: str = "minilm") -> BaseEmbedder:
    if name in ("minilm", "all-MiniLM-L6-v2"):
        return MiniLMEmbedder()
    raise ValueError(f"unknown embedder: {name!r}")
