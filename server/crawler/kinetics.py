"""Kinetics-700 real-video crawler.

Kinetics-700 (Carreira et al. 2020) is a large-scale video action recognition
dataset. For VidTide, we sample ~2000-3000 real videos to serve as the
negative class in AUROC evaluation.

The dataset provides (YouTube ID, start_sec, end_sec, label) tuples via a
public CSV. This crawler downloads a random sample using yt-dlp, clipping
each video to its Kinetics-specified time window.

Reference: https://github.com/cvdfoundation/kinetics-dataset
CSV mirror: https://s3.amazonaws.com/kinetics/700_2020/annotations/train.csv
"""

from __future__ import annotations

import csv
import io
import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Iterator

import requests

from server.crawler.base import BaseCrawler, CrawledVideo
from server.crawler.registry import register

logger = logging.getLogger(__name__)

# Kinetics-700-2020 train split CSV (~650k entries)
KINETICS_CSV_URL = (
    "https://s3.amazonaws.com/kinetics/700_2020/annotations/train.csv"
)


@register("kinetics")
class KineticsCrawler(BaseCrawler):
    """Sample Kinetics-700 for real (non-AI) video negatives."""

    name = "kinetics"
    tier = 1  # highest confidence: dataset-level ground truth

    def crawl(self, config: dict) -> Iterator[CrawledVideo]:
        max_videos = config.get("max_videos", 200)
        seed = config.get("seed", 42)
        min_duration = config.get("min_duration", 3)
        max_duration = config.get("max_duration", 60)
        csv_cache = Path(config.get("csv_cache", "data/cache/kinetics_train.csv"))

        # Step 1: load the Kinetics annotations CSV
        entries = self._load_annotations(csv_cache)
        logger.info("KineticsCrawler: loaded %d Kinetics annotations", len(entries))

        # Step 2: random sample with stable seed
        rng = random.Random(seed)
        sample = rng.sample(entries, min(max_videos * 3, len(entries)))  # 3x for yt-dlp failures

        # Step 3: yield CrawledVideo entries (without downloading yet — the
        # download stage handles that). We store start/end in raw_metadata
        # so downstream can clip correctly.
        yielded = 0
        for entry in sample:
            if yielded >= max_videos:
                break

            duration = entry["end_sec"] - entry["start_sec"]
            if duration < min_duration or duration > max_duration:
                continue

            yt_id = entry["youtube_id"]
            source_url = (
                f"https://www.youtube.com/watch?v={yt_id}"
                f"&t={entry['start_sec']}s"
            )

            yield CrawledVideo(
                source_platform="kinetics",
                source_url=source_url,
                source_id=yt_id,
                label="real",
                label_source="tier1_dataset",
                title=f"Kinetics: {entry['label']}",
                claimed_generator=None,  # not AI
                content_tags=[f"kinetics:{entry['label']}", "kinetics700"],
                raw_metadata={
                    "kinetics_label": entry["label"],
                    "start_sec": entry["start_sec"],
                    "end_sec": entry["end_sec"],
                    "split": entry["split"],
                },
                thumbnail_url=f"https://i.ytimg.com/vi/{yt_id}/hqdefault.jpg",
                duration_sec=float(duration),
                resolution_w=None,
                resolution_h=None,
                fps=None,
            )
            yielded += 1

        logger.info("KineticsCrawler: yielded %d candidates", yielded)

    def estimate_available(self, config: dict) -> int:
        return config.get("max_videos", 200)

    def _load_annotations(self, cache_path: Path) -> list[dict]:
        """Load Kinetics CSV from cache or fetch from S3."""
        if not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("KineticsCrawler: downloading %s", KINETICS_CSV_URL)
            resp = requests.get(KINETICS_CSV_URL, timeout=120, stream=True)
            resp.raise_for_status()
            with open(cache_path, "wb") as f:
                for chunk in resp.iter_content(1024 * 1024):
                    f.write(chunk)
            logger.info("KineticsCrawler: cached to %s", cache_path)

        entries = []
        with open(cache_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    entries.append({
                        "label": row["label"],
                        "youtube_id": row["youtube_id"],
                        "start_sec": int(float(row["time_start"])),
                        "end_sec": int(float(row["time_end"])),
                        "split": row.get("split", "train"),
                    })
                except (KeyError, ValueError):
                    continue
        return entries
