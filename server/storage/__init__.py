"""Blob storage abstraction for video files.

Single backend: `LocalBlobStorage` — writes to a directory on the
crawler host (`data/blobs/` by default). The crawler host is whatever
machine runs the full pipeline (crawl → filter → dedup → download →
review → publish); today that's a dev laptop, later it can be a
dedicated GPU cloud VM with enough storage + bandwidth.

In-flight videos live on the crawler host as `file://` blobs until
the monthly freeze + publish. Publishing uploads the mp4 files to
HuggingFace Datasets (via `server/release/hf_publisher.py`), after
which HF is the authoritative source and the local copies can be
deleted by the aging job (`scripts/age_videos.py`).

The Fly.io cloud container never stores video files — only metadata
and thumbnails. This is why there's no R2 backend: HF Datasets is
already the durable public endpoint, R2 would have been redundant.
"""

from server.storage.base import BaseBlobStorage
from server.storage.local import LocalBlobStorage

__all__ = ["BaseBlobStorage", "LocalBlobStorage"]
