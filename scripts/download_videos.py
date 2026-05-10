"""Recover video files from a manifest (or split file) by re-fetching from
the original platform URL stored in the manifest record.

Following the VidProM precedent, VidTide does **not** redistribute video files.
Each manifest record carries the original ``source_url``; this script walks
the manifest and downloads each clip to ``--out`` using ``yt-dlp``.

Usage
-----

    python scripts/download_videos.py \
        --manifest manifests/M0/splits/test.jsonl \
        --out data/M0/test/

Records that fail to download (404, geo-block, removed by uploader) are
written to ``--out/_failures.jsonl`` along with the reason; those clips are
silently dropped from any downstream evaluation.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def yt_dlp_available() -> bool:
    return shutil.which("yt-dlp") is not None


def download_one(url: str, out_dir: Path, vid: str) -> tuple[bool, str]:
    """Return (success, message). Saves to ``out_dir/{vid}.mp4`` on success."""
    if not yt_dlp_available():
        return False, "yt-dlp not installed"
    out_path = out_dir / f"{vid}.mp4"
    if out_path.exists() and out_path.stat().st_size > 0:
        return True, "already downloaded"
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--no-playlist",
        "-f", "mp4/best",
        "-o", str(out_path),
        url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if proc.returncode != 0:
        return False, (proc.stderr or "yt-dlp failed").strip().splitlines()[-1][:200]
    if not out_path.exists() or out_path.stat().st_size == 0:
        return False, "empty file"
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path,
                    help="JSONL file with one record per clip; each record must "
                         "have id and source_url fields.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output directory; created if missing.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only download the first N records (debug).")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    failures_path = args.out / "_failures.jsonl"

    n_total = 0
    n_ok = 0
    n_skipped = 0
    with args.manifest.open() as f, failures_path.open("w") as ferr:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            vid = rec.get("id") or rec.get("video")
            url = rec.get("source_url")
            if not vid or not url:
                n_skipped += 1
                ferr.write(json.dumps({"id": vid, "reason": "missing id or source_url"}) + "\n")
                continue
            n_total += 1
            ok, msg = download_one(url, args.out, vid)
            status = "OK" if ok else "FAIL"
            print(f"[{n_ok + 1:>5}/{n_total:>5}] {status} {vid} {msg}", file=sys.stderr)
            if ok:
                n_ok += 1
            else:
                ferr.write(json.dumps({"id": vid, "url": url, "reason": msg}) + "\n")
            if args.limit is not None and n_total >= args.limit:
                break

    print(f"Done: {n_ok}/{n_total} downloaded; {n_skipped} skipped; "
          f"failures logged to {failures_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
