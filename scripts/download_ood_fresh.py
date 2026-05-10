#!/usr/bin/env python3
"""Download 50 fake + 50 real from a selection JSON into data/ood_fresh/.

Reuses the download / re-encode plumbing from scripts/batch_download.py
(yt-dlp, bilibili DASH fallback, 720p ffmpeg pass, sha256, rate limiter)
but writes into data/ood_fresh/{fake,real}/{id}.mp4 and emits an
ood_fresh.jsonl manifest for downstream feature extraction.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.batch_download import (  # noqa: E402
    _BilibiliRateLimitError,
    download_one,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ood-dl")

SELECTION = Path("/tmp/ood_fresh_selection.json")
OUT_ROOT = REPO / "data" / "ood_fresh"
WORK_DIR = Path("/tmp/vidtide-ood-dl")
MANIFEST = OUT_ROOT / "ood_fresh.jsonl"


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    sel = json.loads(SELECTION.read_text())
    todo = sel["fake"] + sel["real"]
    log.info("selection: fake=%d real=%d total=%d", len(sel["fake"]), len(sel["real"]), len(todo))

    manifest_rows = []
    # Load any existing manifest so reruns are idempotent
    done_ids: set[str] = set()
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            if not line.strip(): continue
            row = json.loads(line)
            if row.get("status") == "done":
                done_ids.add(row["video"])
                manifest_rows.append(row)
        log.info("resume: %d already done", len(done_ids))

    t0 = time.time()
    ok = 0
    fail = 0
    bili_backoff = 0
    for i, v in enumerate(todo, 1):
        vid = v["id"]
        if vid in done_ids:
            continue
        log.info("[%d/%d] %s label=%s plat=%s gen=%s dur=%.1fs",
                 i, len(todo), vid, v["label"], v["source_platform"],
                 v.get("claimed_generator"), v.get("duration_sec") or 0)
        try:
            res = download_one(v, blob_dir=OUT_ROOT, work_dir=WORK_DIR)
        except _BilibiliRateLimitError:
            bili_backoff += 1
            wait = min(60 * (2 ** min(bili_backoff, 4)), 600)
            log.warning("bilibili 412 rate limit; sleeping %ds", wait)
            time.sleep(wait)
            try:
                res = download_one(v, blob_dir=OUT_ROOT, work_dir=WORK_DIR)
            except _BilibiliRateLimitError:
                log.error("still rate-limited after backoff; marking failed and moving on")
                res = {"video_id": vid, "status": "failed", "error": "bili rate limit"}

        row = {
            "video": vid,
            "label": 1 if v["label"] == "fake" else 0,
            "label_str": v["label"],
            "platform": v["source_platform"],
            "generator": v.get("claimed_generator"),
            "source_url": v["source_url"],
            "source_id": v["source_id"],
            "duration_sec": v.get("duration_sec"),
            "status": res.get("status"),
            "error": res.get("error"),
            "sha256": res.get("sha256"),
            "file_size": res.get("file_size"),
        }
        manifest_rows.append(row)
        # Write-through so we don't lose progress
        with MANIFEST.open("w") as f:
            for r in manifest_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if res.get("status") == "done":
            ok += 1
        else:
            fail += 1
            log.warning("  FAILED %s: %s", vid, res.get("error"))

    elapsed = time.time() - t0
    log.info("done: ok=%d fail=%d elapsed=%.1fs → %s", ok, fail, elapsed, MANIFEST)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
