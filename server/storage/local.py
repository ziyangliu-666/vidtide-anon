"""Local-filesystem blob storage — the only backend.

Stores files under a root directory on the crawler host (today: dev
laptop, future: cloud VM with adequate storage + bandwidth) and returns
`file://` URLs. The URL is valid on the machine that wrote the file;
for the published-benchmark case (where external researchers need to
download), the monthly freeze + HF publish step uploads the mp4 to
HuggingFace Datasets and rewrites each video's `blob_url` to the HF
raw URL (see `server/release/hf_publisher.py`).

There is no remote-blob-store backend. The original plan called for
Cloudflare R2 as a hot tier, but HF Datasets already provides the
publish target + public download CDN, and keeping a separate R2 tier
was pure redundancy.
"""

from __future__ import annotations

from pathlib import Path

from server.storage.base import BaseBlobStorage


class LocalBlobStorage(BaseBlobStorage):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Normalize away leading slashes so joins don't escape root
        key = key.lstrip("/")
        full = self.root / key
        full.parent.mkdir(parents=True, exist_ok=True)
        return full

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        path = self._path(key)
        path.write_bytes(data)
        return self.url_for(key)

    def get(self, key: str) -> bytes | None:
        path = self._path(key)
        if not path.exists():
            return None
        return path.read_bytes()

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()

    def url_for(self, key: str) -> str:
        return f"file://{self._path(key).resolve()}"

    def exists(self, key: str) -> bool:
        return self._path(key).exists()
