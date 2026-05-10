"""sqlite-vec backed near-neighbor index for thumbnail-caption embeddings.

Design:
- One virtual table `vec_thumbnails(video_id TEXT PRIMARY KEY, embedding FLOAT[384])`
  with cosine distance metric. 384 matches sentence-transformers/all-MiniLM-L6-v2.
- The index owns its own raw sqlite3 connection with the sqlite-vec extension
  loaded. It does NOT go through SQLAlchemy — loadable extensions require
  `enable_load_extension(True)` which SQLAlchemy's engine doesn't surface
  cleanly, and we don't want the cloud FastAPI app paying the sqlite-vec
  import cost on boot.
- `ensure_table()` is idempotent; the migration system does NOT create the
  virtual table (see migrations/0002_dedup.sql comment for the rationale).
- Writes go in via `upsert()`, reads via `knn()`. The caller is responsible
  for providing a pre-normalized float32 vector of the right dimension.

Cosine distance convention: sqlite-vec `distance_metric=cosine` returns
`1 - cosine_similarity`, so smaller = more similar. A threshold of
`distance < 0.08` corresponds to cosine_sim > 0.92 (the "likely duplicate"
cutoff documented in REQUIREMENTS.md R1).
"""

from __future__ import annotations

import logging
import sqlite3
import struct
from pathlib import Path

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512  # CLIP-vit-base-patch32 (was 384 for MiniLM text embed)


def _load_extension(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec on a raw connection. Lazy import so importing this
    module without the extension installed is still cheap."""
    import sqlite_vec  # noqa: WPS433 — lazy import

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def _pack(vec: list[float]) -> bytes:
    """Pack a Python float list into little-endian float32 bytes."""
    if len(vec) != EMBEDDING_DIM:
        raise ValueError(
            f"expected {EMBEDDING_DIM}-dim vector, got {len(vec)}"
        )
    return struct.pack(f"{EMBEDDING_DIM}f", *vec)


class VecIndex:
    """Thin wrapper around the `vec_thumbnails` sqlite-vec virtual table."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            _load_extension(conn)
            self._conn = conn
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "VecIndex":
        self._get_conn()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def ensure_table(self) -> None:
        """Create the vec_thumbnails virtual table if it doesn't exist.

        Idempotent — safe to call on every pipeline run. The IF NOT EXISTS
        pattern isn't supported on CREATE VIRTUAL TABLE, so we check
        sqlite_master first.
        """
        conn = self._get_conn()
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vec_thumbnails'"
        ).fetchone()
        if exists:
            return
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE vec_thumbnails USING vec0(
                video_id TEXT PRIMARY KEY,
                embedding FLOAT[{EMBEDDING_DIM}] distance_metric=cosine
            )
            """
        )
        conn.commit()
        logger.info("vec_index: created vec_thumbnails virtual table")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def upsert(self, video_id: str, embedding: list[float]) -> None:
        """Insert or replace an embedding for the given video_id."""
        conn = self._get_conn()
        packed = _pack(embedding)
        # vec0 doesn't support ON CONFLICT — delete then insert.
        conn.execute("DELETE FROM vec_thumbnails WHERE video_id = ?", (video_id,))
        conn.execute(
            "INSERT INTO vec_thumbnails(video_id, embedding) VALUES (?, ?)",
            (video_id, packed),
        )
        conn.commit()

    def delete(self, video_id: str) -> None:
        """Remove an embedding (e.g. when a video is itself marked duplicate)."""
        conn = self._get_conn()
        conn.execute("DELETE FROM vec_thumbnails WHERE video_id = ?", (video_id,))
        conn.commit()

    def knn(
        self,
        query_embedding: list[float],
        k: int = 5,
        exclude_video_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return the k nearest neighbors as [(video_id, cosine_distance), ...].

        Distance is `1 - cosine_similarity`, so lower is closer. An exact
        match returns 0.0.

        Optionally exclude a specific video_id — useful when the query
        vector is the video we just inserted and we want to find neighbors
        other than ourselves.
        """
        conn = self._get_conn()
        packed = _pack(query_embedding)
        # Fetch k+1 so we can drop self-match if exclude_video_id is given.
        limit = k + 1 if exclude_video_id is not None else k
        rows = conn.execute(
            """
            SELECT video_id, distance
            FROM vec_thumbnails
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (packed, limit),
        ).fetchall()
        results = [(vid, dist) for vid, dist in rows if vid != exclude_video_id]
        return results[:k]

    def count(self) -> int:
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM vec_thumbnails").fetchone()
        return row[0] if row else 0
