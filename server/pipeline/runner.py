"""Pipeline runner: orchestrates crawl → filter → dedup → download.

Key properties (added in the observable-pipeline refactor):
  - **Progress visible**: PipelineRun.stats updated after every video in
    slow stages (crawl, download) so the dashboard shows real-time counts.
  - **Cancellable**: if someone sets PipelineRun.status to 'cancelling'
    (via POST /api/pipeline/runs/{id}/cancel), the runner finishes the
    current video, writes progress, sets status='paused', and returns.
    The next `POST /api/pipeline/run` resumes implicitly because each
    stage skips already-processed rows.
  - **Crash-safe**: per-video commits mean a dead runner only loses the
    video it was working on. PipelineRun stays 'running' after a crash;
    the router detects this via the heartbeat timestamp and marks it
    'failed' on the next health check.
  - **Resumable by design**: crawl skips existing (platform, source_id);
    dedup skips rows where caption_model IS NOT NULL; download skips rows
    where blob_url IS NOT NULL. A new run after a crash or pause picks up
    from where the last one left off — no explicit "resume" API.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone

from tqdm import tqdm

from server.config import get_settings
from server.db.models import PipelineRun, Video

logger = logging.getLogger(__name__)


def _load_crawlers(config: dict) -> list:
    from server.crawler.registry import load_enabled
    return load_enabled(config)


class CancelledError(Exception):
    """Raised when the user cancels a pipeline run mid-flight."""


class PipelineRunner:
    """Orchestrates crawl → filter → dedup → download."""

    def __init__(self, config: dict, db_session=None, mode: str = "local") -> None:
        self.config = config
        self.db = db_session
        self.mode = mode
        self.settings = get_settings()
        self._run: PipelineRun | None = None
        self._stats: dict = {}

    # ------------------------------------------------------------------
    # Progress + cancellation helpers
    # ------------------------------------------------------------------

    def _sync_progress(self) -> None:
        """Flush current stats + heartbeat to PipelineRun row."""
        if self._run is None or self.db is None:
            return
        self._stats["heartbeat"] = datetime.now(timezone.utc).isoformat()
        self._run.stats = json.dumps(self._stats)
        self.db.commit()

    def _check_cancelled(self) -> None:
        """Re-read PipelineRun.status; raise CancelledError if 'cancelling'."""
        if self._run is None or self.db is None:
            return
        # Expire cached state so we get the latest from DB
        self.db.expire(self._run, ["status"])
        if self._run.status == "cancelling":
            logger.info("pipeline run %s: cancel requested", self._run.id[:8])
            raise CancelledError()

    # ------------------------------------------------------------------
    # Main orchestrator
    # ------------------------------------------------------------------

    def run(self, flow_type: str = "full", max_videos: int | None = None) -> dict:
        if self.db is not None:
            self._run = PipelineRun(
                flow_type=flow_type,
                config=json.dumps(self.config),
                status="running",
                started_at=datetime.now(timezone.utc),
            )
            self.db.add(self._run)
            self.db.commit()

        self._stats = {
            "flow_type": flow_type,
            "run_id": self._run.id if self._run else None,
            "stages": {},
        }
        start_time = time.time()

        try:
            # --- PROCESS PENDING ---
            # Phase 2: work on existing filtered videos (download + dedup).
            # No crawling — just processes what's already in the DB. This is
            # the same logic as scripts/process_pending.py but runs in-process
            # with full progress tracking and cancellation support.
            if flow_type == "process":
                self._run_process_pending(max_videos or 50)
                self._finish("completed", start_time)
                return self._stats

            if max_videos is None:
                total = 0
                for _, pcfg in self.config.get("crawl", {}).get("platforms", {}).items():
                    if pcfg.get("enabled", False):
                        total += pcfg.get("max_videos", 0)
                max_videos = total or 50

            # --- CRAWL ---
            stage_start = time.time()
            videos = self._crawl(max_videos)
            self._stats["stages"]["crawl"] = {
                "count": len(videos),
                "duration_sec": round(time.time() - stage_start, 2),
            }
            self._sync_progress()
            logger.info("Crawl complete: %d videos", len(videos))

            # --- REMOTE PUSH ---
            if self.mode == "remote":
                if videos:
                    stage_start = time.time()
                    videos = self._filter(videos)
                    self._stats["stages"]["filter"] = {
                        "count": len(videos),
                        "duration_sec": round(time.time() - stage_start, 2),
                    }

                import base64

                def _video_to_payload(v):
                    pub = getattr(v, "published_at", None)
                    if pub and not isinstance(pub, str):
                        pub = pub.isoformat()
                    payload = {
                        "source_platform": v.source_platform,
                        "source_url": v.source_url,
                        "source_id": v.source_id,
                        "title": v.title,
                        "label": v.label,
                        "label_source": v.label_source,
                        "claimed_generator": v.claimed_generator,
                        "duration_sec": v.duration_sec,
                        "resolution_w": v.resolution_w,
                        "resolution_h": v.resolution_h,
                        "fps": v.fps,
                        "content_tags": json.loads(v.content_tags) if v.content_tags else None,
                        "thumbnail_url": v.thumbnail_url,
                        "published_at": pub,
                        "status": "filtered",
                    }
                    thumb = getattr(v, "thumbnail_bytes", None)
                    if thumb:
                        payload["thumbnail_b64"] = base64.b64encode(thumb).decode("ascii")
                    return payload

                videos_data = [_video_to_payload(v) for v in videos]
                push_result = self._push_remote_chunked(videos_data, chunk_size=30)
                self._stats["stages"]["push"] = push_result
                self._finish("completed", start_time)
                return self._stats

            if flow_type == "crawl":
                self._finish("completed", start_time)
                return self._stats

            # --- FILTER ---
            stage_start = time.time()
            videos = self._filter(videos)
            self._stats["stages"]["filter"] = {
                "count": len(videos),
                "duration_sec": round(time.time() - stage_start, 2),
            }
            self._sync_progress()
            logger.info("Filter complete: %d videos", len(videos))

            # --- DEDUP ---
            dedup_cfg = self.config.get("dedup", {}) or {}
            if videos and dedup_cfg.get("enabled", True):
                self._check_cancelled()
                stage_start = time.time()
                dedup_stats = self._dedupe(videos, dedup_cfg)
                self._stats["stages"]["dedup"] = {
                    "processed": dedup_stats.processed,
                    "captioned": dedup_stats.captioned,
                    "embedded": dedup_stats.embedded,
                    "duplicates_found": dedup_stats.duplicates_found,
                    "new_canonicals": dedup_stats.new_canonicals,
                    "errors": dedup_stats.errors,
                    "duration_sec": round(time.time() - stage_start, 2),
                }
                self._sync_progress()
                logger.info("Dedup complete: %s", dedup_stats)

            # --- DOWNLOAD ---
            download_cfg = self.config.get("download", {}) or {}
            canonical_videos = [v for v in videos if not v.duplicate_of_id]
            if canonical_videos and download_cfg.get("enabled", True):
                self._check_cancelled()
                stage_start = time.time()
                download_stats = self._download(canonical_videos, download_cfg)
                self._stats["stages"]["download"] = {
                    **download_stats,
                    "duration_sec": round(time.time() - stage_start, 2),
                }
                self._sync_progress()
                logger.info("Download complete: %s", download_stats)

            self._finish("completed", start_time)

        except CancelledError:
            logger.info("Pipeline run paused by user")
            self._finish("paused", start_time)
        except Exception:
            logger.error("Pipeline run failed", exc_info=True)
            self._finish("failed", start_time)
            raise

        return self._stats

    def _finish(self, status: str, start_time: float) -> None:
        self._stats["total_duration_sec"] = round(time.time() - start_time, 2)
        if self._run is not None:
            self._run.status = status
            self._run.completed_at = datetime.now(timezone.utc)
            self._run.stats = json.dumps(self._stats)
            self.db.commit()

    # ------------------------------------------------------------------
    # Remote push (used in mode="remote" only)
    # ------------------------------------------------------------------

    def _push_remote(self, videos_data: list[dict]) -> dict:
        import requests
        import time as _time

        api_url = os.environ.get("VIDTIDE_API_URL", "https://your-vidtide-instance.example.com")
        api_key = os.environ.get("VIDTIDE_API_KEY", "")
        endpoint = f"{api_url}/api/videos/import"
        body = {"videos": videos_data}
        headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                resp = requests.post(endpoint, json=body, headers=headers, timeout=120)
                if resp.status_code in (500, 502, 503, 504) and attempt < 3:
                    _time.sleep(3 * (attempt + 1))
                    continue
                resp.raise_for_status()
                return resp.json()
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                if attempt == 2:
                    raise
                _time.sleep(3 * (attempt + 1))
        raise last_exc or RuntimeError("remote push exhausted retries")

    def _push_remote_chunked(self, videos_data: list[dict], chunk_size: int = 50) -> dict:
        totals = {"imported": 0, "updated": 0, "skipped": 0, "total": 0}
        for i in range(0, len(videos_data), chunk_size):
            chunk = videos_data[i : i + chunk_size]
            result = self._push_remote(chunk)
            for k in totals:
                totals[k] += int(result.get(k, 0))
        return totals

    # ------------------------------------------------------------------
    # Stage: crawl
    # ------------------------------------------------------------------

    def _crawl(self, max_videos: int) -> list[Video]:
        """Crawl videos from all enabled platforms (metadata only).

        Per-video commit + progress sync every 5 videos. Checks
        cancellation between platforms.
        """
        crawlers = _load_crawlers(self.config)
        if not crawlers:
            logger.warning("No crawlers enabled")
            return []

        saved_videos: list[Video] = []
        progress = {"saved": 0, "skipped_existing": 0, "errors": 0}

        for platform_name, crawler, platform_config in crawlers:
            self._check_cancelled()

            remaining = max_videos - len(saved_videos)
            if remaining <= 0:
                break

            platform_config = dict(platform_config)
            platform_config["max_videos"] = min(
                platform_config.get("max_videos", remaining), remaining
            )

            logger.info("Crawling %s (max %d)...", platform_name, platform_config["max_videos"])

            for crawled in tqdm(
                crawler.crawl(platform_config),
                desc=f"Crawling {platform_name}",
                total=platform_config["max_videos"],
            ):
                if len(saved_videos) >= max_videos:
                    break

                try:
                    if self.db is not None:
                        exists = self.db.query(Video.id).filter(
                            Video.source_platform == crawled.source_platform,
                            Video.source_id == crawled.source_id,
                        ).first()
                        if exists:
                            progress["skipped_existing"] += 1
                            continue

                    # Persist thumbnail bytes to disk so the dashboard can
                    # serve them from the same origin. Platform CDN URLs
                    # (hdslb.com, redd.it) hotlink-block cross-origin
                    # requests, so storing the raw CDN URL in thumbnail_url
                    # would show broken images on the dashboard. The
                    # /api/thumbnail/ static route serves from this dir.
                    thumb_url = crawled.thumbnail_url
                    if crawled.thumbnail_bytes:
                        from pathlib import Path
                        thumb_dir = Path(self.settings.data_dir) / "thumbnails"
                        thumb_dir.mkdir(parents=True, exist_ok=True)
                        thumb_name = f"{crawled.source_id}.jpg"
                        thumb_path = thumb_dir / thumb_name
                        try:
                            thumb_path.write_bytes(crawled.thumbnail_bytes)
                            thumb_url = f"/api/thumbnail/{thumb_name}"
                        except OSError:
                            pass  # fall back to CDN URL

                    # Parse published_at from the ISO string the crawler
                    # produces. Accepts "YYYY-MM-DD" (YouTube) and full
                    # ISO datetime with timezone (Bilibili, Reddit).
                    pub_dt = None
                    if crawled.published_at:
                        try:
                            pub_dt = datetime.fromisoformat(crawled.published_at)
                        except (ValueError, TypeError):
                            pub_dt = None

                    video = Video(
                        source_platform=crawled.source_platform,
                        source_url=crawled.source_url,
                        source_id=crawled.source_id,
                        title=crawled.title,
                        label=crawled.label,
                        label_source=crawled.label_source,
                        label_confidence=None,
                        claimed_generator=crawled.claimed_generator,
                        duration_sec=crawled.duration_sec,
                        resolution_w=crawled.resolution_w,
                        resolution_h=crawled.resolution_h,
                        fps=crawled.fps,
                        file_size_bytes=None,
                        content_tags=json.dumps(crawled.content_tags) if crawled.content_tags else None,
                        has_watermark=None,
                        storage_path=None,
                        thumbnail_path=None,
                        thumbnail_url=thumb_url,
                        status="filtered",
                        published_at=pub_dt,
                        crawled_at=datetime.now(timezone.utc),
                    )
                    video.thumbnail_bytes = crawled.thumbnail_bytes
                    if self.db is not None:
                        self.db.add(video)
                        self.db.commit()
                    saved_videos.append(video)
                    progress["saved"] += 1

                    # Sync progress + check cancel every video (not every
                    # 5 — yt-dlp searches can block 30s+ per video on slow
                    # platforms, so checking per-platform was too coarse)
                    self._stats["stages"]["crawl"] = {**progress, "stage": "running"}
                    self._sync_progress()
                    self._check_cancelled()

                except Exception:
                    logger.warning("Error processing %s, skipping", crawled.source_id, exc_info=True)
                    if self.db is not None:
                        self.db.rollback()
                    progress["errors"] += 1
                    continue

        return saved_videos

    # ------------------------------------------------------------------
    # Stage: dedup
    # ------------------------------------------------------------------

    def _dedupe(self, videos: list[Video], dedup_cfg: dict):
        from server.dedup.deduplicator import dedupe_batch, DEFAULT_COSINE_THRESHOLD
        from server.dedup.image_embedder import get_image_embedder

        embedder_name = dedup_cfg.get("embedder", "clip")
        threshold = float(dedup_cfg.get("cosine_threshold", DEFAULT_COSINE_THRESHOLD))
        thumbnails_root = dedup_cfg.get("thumbnails_root", "data/thumbnails")

        # Skip already-deduped videos (resume-safe)
        to_process = [v for v in videos if v.caption_model is None]
        if len(to_process) < len(videos):
            logger.info(
                "dedup: skipping %d already-processed videos, running on %d",
                len(videos) - len(to_process), len(to_process),
            )

        return dedupe_batch(
            to_process,
            self.db,
            db_path=self.settings.db_path,
            thumbnails_root=thumbnails_root,
            image_embedder=get_image_embedder(embedder_name),
            cosine_threshold=threshold,
        )

    # ------------------------------------------------------------------
    # Stage: download
    # ------------------------------------------------------------------

    def _download(self, videos: list[Video], download_cfg: dict) -> dict:
        """Download + re-encode + store mp4 blobs locally.

        Per-video commit + progress sync + cancellation check after each
        video. Skip already-downloaded (blob_url IS NOT NULL) for resume.
        """
        from server.pipeline.postprocess import DownloadStage
        from server.storage.local import LocalBlobStorage

        blob_dir = download_cfg.get("blob_dir", self.settings.blob_dir)
        work_dir = download_cfg.get("work_dir", "/tmp/vidtide-dl")
        max_duration = int(download_cfg.get("max_duration_sec", 600))

        storage = LocalBlobStorage(blob_dir)
        stage = DownloadStage(
            blob_storage=storage,
            work_dir=work_dir,
            max_duration=max_duration,
        )

        stats = {"attempted": 0, "downloaded": 0, "failed": 0, "skipped_existing": 0}

        for video in videos:
            if video.blob_url:
                stats["skipped_existing"] += 1
                continue

            self._check_cancelled()

            stats["attempted"] += 1
            try:
                result = stage.process(
                    source_url=video.source_url,
                    video_id=video.id,
                )
            except Exception:
                logger.warning(
                    "download: crashed on %s, continuing",
                    video.id[:8],
                    exc_info=True,
                )
                stats["failed"] += 1
                continue
            if result is None:
                stats["failed"] += 1
                continue
            video.blob_url = result.blob_url
            video.blob_sha256 = result.blob_sha256
            video.file_size_bytes = result.file_size_bytes
            if self.db is not None:
                self.db.commit()
            stats["downloaded"] += 1
            logger.info(
                "download: %s -> %s (%.1f MB, sha=%s...)",
                video.id[:8],
                result.blob_url,
                result.file_size_bytes / 1e6,
                result.blob_sha256[:12],
            )

            # Sync progress every video (downloads are slow)
            self._stats["stages"]["download"] = {**stats, "stage": "running"}
            self._sync_progress()

        return stats

    # ------------------------------------------------------------------
    # Stage: filter
    # ------------------------------------------------------------------

    def _run_process_pending(self, limit: int) -> None:
        """Phase 2: process existing filtered videos (dedup + download).

        Queries the DB for videos that are filtered but haven't been
        deduped or downloaded yet, then runs the dedup and download
        stages on them. Same progress tracking and cancellation as the
        normal pipeline flow.
        """
        # Find pending videos: anything not excluded that still needs
        # download + dedup. Status is no longer the gate — blob_url /
        # caption_model / duplicate_of_id carry the real progress signal.
        pending = (
            self.db.query(Video)
            .filter(Video.status != "excluded")
            .filter(Video.blob_url.is_(None))
            .filter(Video.duplicate_of_id.is_(None))
            .filter(Video.caption_model.is_(None))
            .order_by(Video.crawled_at.asc())
            .limit(limit)
            .all()
        )

        self._stats["stages"]["pending"] = {
            "total": len(pending),
        }
        self._sync_progress()
        logger.info("Process pending: %d videos to process", len(pending))

        if not pending:
            return

        # --- DEDUP ---
        dedup_cfg = self.config.get("dedup", {}) or {}
        if dedup_cfg.get("enabled", True):
            self._check_cancelled()
            import time as _time
            stage_start = _time.time()
            dedup_stats = self._dedupe(pending, dedup_cfg)
            self._stats["stages"]["dedup"] = {
                "processed": dedup_stats.processed,
                "captioned": dedup_stats.captioned,
                "embedded": dedup_stats.embedded,
                "duplicates_found": dedup_stats.duplicates_found,
                "new_canonicals": dedup_stats.new_canonicals,
                "errors": dedup_stats.errors,
                "duration_sec": round(_time.time() - stage_start, 2),
            }
            self._sync_progress()
            logger.info("Dedup complete: %s", dedup_stats)

        # --- DOWNLOAD ---
        download_cfg = self.config.get("download", {}) or {}
        canonical = [v for v in pending if not v.duplicate_of_id]
        if canonical and download_cfg.get("enabled", True):
            self._check_cancelled()
            import time as _time
            stage_start = _time.time()
            download_stats = self._download(canonical, download_cfg)
            self._stats["stages"]["download"] = {
                **download_stats,
                "duration_sec": round(_time.time() - stage_start, 2),
            }
            self._sync_progress()
            logger.info("Download complete: %s", download_stats)

    def _filter(self, videos: list[Video]) -> list[Video]:
        from server.filter.quality_filter import QualityFilter
        from server.filter.tag_filter import TagFilter

        filter_config = self.config.get("filter", {})

        video_dicts = [
            {
                "id": v.id,
                "source_id": v.source_id,
                "source_platform": v.source_platform,
                "source_url": v.source_url,
                "title": v.title or "",
                "description": "",
                "label_source": v.label_source,
                "claimed_generator": v.claimed_generator,
                "content_tags": json.loads(v.content_tags) if v.content_tags else [],
                "resolution_w": v.resolution_w,
                "resolution_h": v.resolution_h,
                "duration_sec": v.duration_sec,
                "file_size_bytes": v.file_size_bytes,
            }
            for v in videos
        ]

        quality_filter = QualityFilter()
        quality_cfg = {
            "min_resolution": filter_config.get("quality", {}).get("min_resolution", 480),
            "min_duration": filter_config.get("quality", {}).get("min_duration_sec", 3),
            "max_duration": filter_config.get("quality", {}).get("max_duration_sec", 60),
        }
        filtered = quality_filter.filter(video_dicts, quality_cfg)
        logger.info("QualityFilter: %s", quality_filter.stats())

        tag_filter = TagFilter()
        tag_cfg = filter_config.get("tag", {})
        filtered = tag_filter.filter(filtered, tag_cfg)
        logger.info("TagFilter: %s", tag_filter.stats())

        whitelist_cfg = filter_config.get("model_whitelist", {})
        if whitelist_cfg.get("enabled", False):
            from server.filter.model_whitelist_filter import ModelWhitelistFilter
            whitelist_filter = ModelWhitelistFilter()
            filtered = whitelist_filter.filter(filtered, whitelist_cfg)
            logger.info("ModelWhitelistFilter: %s", whitelist_filter.stats())

        llm_cfg = filter_config.get("llm", {})
        if llm_cfg.get("enabled", False):
            from server.filter.llm_filter import LLMFilter
            llm_filter = LLMFilter()
            filtered = llm_filter.filter(filtered, llm_cfg)
            logger.info("LLMFilter: %s", llm_filter.stats())

        surviving_ids = {d["id"] for d in filtered}

        kept: list[Video] = []
        for v in videos:
            if v.id in surviving_ids:
                kept.append(v)
            else:
                # Filter rejected this row — soft-delete so it stays out of
                # review / download / listing queues. Default status is
                # already "filtered" for newly-crawled rows, so only the
                # rejects need touching.
                v.status = "excluded"
        if self.db is not None:
            self.db.commit()

        return kept
