"""CLIP-based cross-platform near-duplicate detector.

Full implementation lives under ``server/dedup/``; the relevant entry points
are :class:`server.dedup.deduplicator.Deduplicator` and the CLIP image embedder
in :mod:`server.dedup.image_embedder`.
"""
from server.dedup.deduplicator import Deduplicator
from server.dedup.image_embedder import CLIPImageEmbedder

__all__ = ["Deduplicator", "CLIPImageEmbedder"]
