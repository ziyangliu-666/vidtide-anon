"""Evaluation runner — orchestrate detector inference over a video set.

Reads videos from the local DB, extracts frames, runs detector, caches
per-video scores as JSONL. Metrics are computed separately from cached
scores by scripts/compute_metrics.py.

Design: resumable per-video. Each video's result is written immediately
so interruptions don't lose work. Re-running with the same (detector,
benchmark) skips videos already scored.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass
class EvalSpec:
    detector: str
    benchmark: str      # "vidtide_m0" | "genvideo" | ...
    num_frames: int = 8
    resolution: int = 224
    limit: int | None = None
    platform_filter: str | None = None  # optional: eval only one platform


def iter_benchmark_videos(
    db_path: Path,
    benchmark: str,
    limit: int | None = None,
    platform_filter: str | None = None,
) -> Iterable[dict]:
    """Yield (video_id, label, local_path, source_platform, claimed_generator)
    from the local DB. Currently supports benchmark='vidtide_m0' which
    means: all reviewed canonical videos with blob_url locally available.

    Falls back to status='filtered' for research iteration before human
    review is complete.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    where = [
        "duplicate_of_id IS NULL",
        "blob_url LIKE 'file://%'",
    ]
    params = []

    if benchmark == "vidtide_m0":
        # Accept both reviewed-approved and filtered (pre-review) videos
        # during the paper-drafting phase; the published M0 slice will
        # require reviewed status.
        pass
    elif benchmark == "genvideo":
        where.append("source_platform = 'genvideo'")

    if platform_filter:
        where.append("source_platform = ?")
        params.append(platform_filter)

    sql = "SELECT * FROM videos WHERE " + " AND ".join(where)
    if limit:
        sql += f" LIMIT {limit}"

    for row in conn.execute(sql, params):
        d = dict(row)
        # Strip file:// prefix
        url = d.get("blob_url") or ""
        if url.startswith("file://"):
            d["local_path"] = url[len("file://"):]
        else:
            continue
        yield d
    conn.close()


def run_eval(spec: EvalSpec, db_path: Path, results_dir: Path) -> Path:
    """Run a detector over a benchmark. Returns the output JSONL path."""
    from server.detection.dataset import extract_frames
    from server.detection.registry import load_detector

    out_dir = results_dir / spec.detector
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.benchmark}.jsonl"

    # Resume: skip videos already scored
    already: set[str] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    already.add(json.loads(line)["video_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    logger.info(
        "%s/%s: %d already scored, resuming",
        spec.detector, spec.benchmark, len(already),
    )

    detector = load_detector(spec.detector)

    n_done = 0
    n_err = 0
    with open(out_path, "a") as out_f:
        for video in iter_benchmark_videos(
            db_path, spec.benchmark,
            limit=spec.limit,
            platform_filter=spec.platform_filter,
        ):
            vid = video["id"]
            if vid in already:
                continue

            try:
                frames = extract_frames(
                    Path(video["local_path"]),
                    num_frames=spec.num_frames,
                    resolution=spec.resolution,
                )
                score = detector.predict(frames)
            except Exception as e:
                logger.warning("eval error on %s: %s", vid, e)
                n_err += 1
                continue

            record = {
                "video_id": vid,
                "source_id": video["source_id"],
                "source_platform": video["source_platform"],
                "label": video.get("label"),
                "claimed_generator": video.get("claimed_generator"),
                "score": float(score),
                "label_pred": "fake" if score >= 0.5 else "real",
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            n_done += 1
            if n_done % 20 == 0:
                logger.info("eval progress: %d done / %d err", n_done, n_err)

    detector.close()
    logger.info(
        "%s/%s: done — %d scored, %d errors → %s",
        spec.detector, spec.benchmark, n_done, n_err, out_path,
    )
    return out_path
