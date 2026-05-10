"""Blob storage interface.

Backends must implement:
  - put(key, data, content_type) -> url
  - get(key) -> bytes | None
  - delete(key)
  - url_for(key) -> url  (no-fetch URL computation)
  - exists(key) -> bool

Keys are forward-slash-separated object paths (e.g. "videos/abc123.mp4").
The returned URL is the publicly-accessible (or signed) URL that
benchmark users can follow to download the bytes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseBlobStorage(ABC):
    @abstractmethod
    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Upload `data` at `key` and return a public URL."""

    @abstractmethod
    def get(self, key: str) -> bytes | None:
        """Download the bytes at `key`, or None if it doesn't exist."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove the object at `key`. No-op if it doesn't exist."""

    @abstractmethod
    def url_for(self, key: str) -> str:
        """Return the public URL for `key` without fetching."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return whether the object at `key` exists."""
