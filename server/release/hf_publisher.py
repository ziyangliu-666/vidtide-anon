"""HuggingFace Datasets publisher for monthly benchmark slices.

One public HF Datasets repo (default: `vidtide/benchmark`) hosts all
VidTide slices. Each monthly publish uploads both the mp4 bytes and a
JSON manifest in a single commit:

    <repo>/videos/vidtide-v2026.04/<video_id>.mp4  (git-lfs)
    <repo>/videos/vidtide-v2026.04/<video_id>.mp4  ...
    <repo>/slices/vidtide-v2026.04.json

HF is the sole published-benchmark endpoint. There is no R2 hot tier —
mp4 bytes live on the crawler host under `data/blobs/` until publish
time; after publish they live on HF and the local copies can be
deleted by the aging job (`scripts/age_videos.py`). The video's
`blob_url` is rewritten from `file://...` to the HF raw URL on success,
and the local file is left in place (aging handles cleanup separately
so that a crash after publish can be recovered from).

Why one repo with many files instead of one repo per slice:
- Users can clone the whole benchmark at once
- Git-lfs history keeps deduplicated storage across slices
- HF dataset viewer can show cross-slice stats

Why JSON manifest + raw mp4 files instead of a `datasets`-compatible
Parquet / Audio feature layout:
- Manifest is the abstraction detector authors actually iterate over
- Raw mp4 files are directly addressable via HF raw URLs, no `datasets`
  lib required
- Per-video licensing and edge-case fields are easier to extend in
  JSON than in a fixed-schema Parquet file
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from server.db.models import BenchmarkSlice, SliceVideo, Video

logger = logging.getLogger(__name__)


def _hf_raw_url(repo_id: str, path_in_repo: str) -> str:
    """Build a `/resolve/main/` HF raw URL for a dataset file.

    This is the direct-download URL format (as opposed to `/blob/main/`
    which returns a git-lfs pointer). It's what `<video src>` should
    use in the manifest and what users get from `hf_hub_download()`.
    """
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path_in_repo}"


def _video_to_manifest_entry(
    video: Video,
    decision: str,
    *,
    blob_url_override: str | None = None,
) -> dict[str, Any]:
    """Serialize a single video into the slice manifest schema.

    For published slices, `blob_url_override` is the HF raw URL that
    replaces the video row's local `file://...` path. The manifest
    consumer should always use `blob_url` as the primary playback URL
    and `source_url` as the provenance link back to the original
    platform upload.
    """
    return {
        "id": video.id,
        "source_platform": video.source_platform,
        "source_url": video.source_url,
        "source_id": video.source_id,
        "title": video.title,
        "label": video.label,
        "label_source": video.label_source,
        "claimed_generator": video.claimed_generator,
        "duration_sec": video.duration_sec,
        "resolution_w": video.resolution_w,
        "resolution_h": video.resolution_h,
        "fps": video.fps,
        "blob_url": blob_url_override if blob_url_override is not None else video.blob_url,
        "blob_sha256": video.blob_sha256,
        "file_size_bytes": video.file_size_bytes,
        "source_license": video.source_license or "unknown",
        "caption_text": video.caption_text,
        "caption_model": video.caption_model,
        "human_decision": decision,
    }


def build_manifest(slice_id: str, db: Session) -> dict[str, Any]:
    """Assemble the full manifest for a slice without publishing.

    Used by `/api/slices/{id}/export` and by `publish_slice()` as its
    pre-upload payload. In the pre-upload path, `blob_url` fields are
    whatever is currently on the Video rows (pre-publish: `file://...`);
    `publish_slice` rewrites them to HF URLs after upload and re-builds
    the manifest for the HF-side JSON file.
    """
    bs = db.query(BenchmarkSlice).filter_by(id=slice_id).first()
    if bs is None:
        raise ValueError(f"slice {slice_id} not found")

    rows = (
        db.query(Video, SliceVideo)
        .join(SliceVideo, SliceVideo.video_id == Video.id)
        .filter(SliceVideo.slice_id == slice_id)
        .all()
    )
    videos = [_video_to_manifest_entry(v, sv.decision) for v, sv in rows]

    return {
        "version": bs.version,
        "window_start": bs.window_start.isoformat() if bs.window_start else None,
        "window_end": bs.window_end.isoformat() if bs.window_end else None,
        "total_videos": bs.total_videos,
        "approved_videos": bs.approved_videos,
        "rejected_videos": bs.rejected_videos,
        "platforms": json.loads(bs.platforms) if bs.platforms else {},
        "generators": json.loads(bs.generators) if bs.generators else {},
        "created_at": bs.created_at.isoformat() if bs.created_at else None,
        "notes": bs.notes,
        "schema_version": 2,
        "videos": videos,
    }


def _stage_slice_for_upload(
    slice_id: str,
    db: Session,
    *,
    repo_id: str,
    staging_dir: Path,
) -> tuple[Path, list[tuple[Video, str]]]:
    """Copy each slice video's mp4 into `staging_dir/videos/<version>/`.

    Returns (staging_dir, list of (video, hf_raw_url) pairs). The raw
    URL on each pair is the one that will be reachable after the
    upload commit lands, so the caller can stamp it onto the Video row
    and the manifest without another round-trip to HF.

    Raises if any approved video is missing its local blob — we refuse
    to publish a partial slice.
    """
    bs = db.query(BenchmarkSlice).filter_by(id=slice_id).first()
    if bs is None:
        raise ValueError(f"slice {slice_id} not found")

    videos_dir = staging_dir / "videos" / bs.version
    videos_dir.mkdir(parents=True, exist_ok=True)

    approved = (
        db.query(Video, SliceVideo)
        .join(SliceVideo, SliceVideo.video_id == Video.id)
        .filter(SliceVideo.slice_id == slice_id)
        .filter(SliceVideo.decision == "approved")
        .all()
    )

    staged: list[tuple[Video, str]] = []
    for video, _sv in approved:
        blob_url = video.blob_url or ""
        if not blob_url.startswith("file://"):
            raise ValueError(
                f"video {video.id} is not in local tier (blob_url={blob_url!r}); "
                "publish would drop its bytes. Run the download stage first."
            )
        local_path = Path(blob_url.replace("file://", "", 1))
        if not local_path.exists():
            raise FileNotFoundError(
                f"video {video.id} blob missing at {local_path}"
            )
        dest_filename = f"{video.id}.mp4"
        dest_path = videos_dir / dest_filename
        shutil.copyfile(local_path, dest_path)
        path_in_repo = f"videos/{bs.version}/{dest_filename}"
        hf_url = _hf_raw_url(repo_id, path_in_repo)
        staged.append((video, hf_url))

    return videos_dir, staged


def publish_slice(
    slice_id: str,
    db: Session,
    *,
    hf_token: str,
    repo_id: str = "vidtide/benchmark",
) -> str:
    """Push a slice (mp4 files + JSON manifest) to HuggingFace Datasets.

    Workflow:
      1. stage — copy approved videos' local mp4 files into a temp dir
         under `videos/<version>/<id>.mp4`
      2. build manifest with blob_url rewritten to each file's eventual
         HF raw URL
      3. write the manifest to the same staging dir under
         `slices/<version>.json`
      4. HfApi.upload_folder() commits both trees in a single commit
      5. update DB: `video.blob_url = <hf raw url>` for each approved
         video, `bs.published_at = now()`, `bs.export_url = manifest url`
      6. cleanup staging dir (videos persist locally in data/blobs until
         aging job runs — that's how we recover if this step fails
         between upload and DB commit)

    Idempotent-ish: re-publishing the same version overwrites the
    previous upload in a new commit. Old HF commit history is kept,
    so consumers who pinned a specific commit SHA still see the old
    files. Consumers who didn't pin see the latest.

    Raises on any failure (staging error, HF API exception, DB error)
    so the caller can 500 cleanly and retry after fixing the issue.
    """
    from huggingface_hub import HfApi

    bs = db.query(BenchmarkSlice).filter_by(id=slice_id).first()
    if bs is None:
        raise ValueError(f"slice {slice_id} not found")

    api = HfApi(token=hf_token)
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("hf: create_repo raised (%s); continuing", exc)

    staging_root = Path(tempfile.mkdtemp(prefix=f"vidtide-publish-{bs.version}-"))
    try:
        # 1-2. Stage mp4 files + compute their HF URLs
        _, staged = _stage_slice_for_upload(
            slice_id, db, repo_id=repo_id, staging_dir=staging_root
        )
        logger.info(
            "hf: staged %d videos for %s under %s",
            len(staged), bs.version, staging_root,
        )

        # 3. Build the manifest with HF URLs and write it alongside.
        # Rejected videos still get listed (with their original local blob_url
        # — which is fine because the manifest preserves decision state) but
        # we only uploaded the approved ones, so rejected videos' blob_url
        # will be the local path and consumers should not try to play them.
        rejected_or_unstaged = {
            v.id: url for v, url in staged
        }
        all_rows = (
            db.query(Video, SliceVideo)
            .join(SliceVideo, SliceVideo.video_id == Video.id)
            .filter(SliceVideo.slice_id == slice_id)
            .all()
        )
        manifest_videos = []
        for v, sv in all_rows:
            override = rejected_or_unstaged.get(v.id)
            manifest_videos.append(
                _video_to_manifest_entry(v, sv.decision, blob_url_override=override)
            )

        manifest = {
            "version": bs.version,
            "window_start": bs.window_start.isoformat() if bs.window_start else None,
            "window_end": bs.window_end.isoformat() if bs.window_end else None,
            "total_videos": bs.total_videos,
            "approved_videos": bs.approved_videos,
            "rejected_videos": bs.rejected_videos,
            "platforms": json.loads(bs.platforms) if bs.platforms else {},
            "generators": json.loads(bs.generators) if bs.generators else {},
            "created_at": bs.created_at.isoformat() if bs.created_at else None,
            "notes": bs.notes,
            "schema_version": 2,
            "videos": manifest_videos,
        }
        slices_dir = staging_root / "slices"
        slices_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = slices_dir / f"{bs.version}.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )

        # 4. Upload both trees in a single commit. upload_folder handles
        # LFS for large files automatically and resumes on retry.
        api.upload_folder(
            folder_path=str(staging_root),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Publish {bs.version}",
        )

        # 5. Stamp DB
        manifest_url = _hf_raw_url(repo_id, f"slices/{bs.version}.json")
        for video, hf_url in staged:
            video.blob_url = hf_url
        bs.published_at = datetime.now(timezone.utc)
        bs.export_url = manifest_url
        db.commit()

        logger.info(
            "hf: published %s (%d files) -> %s",
            bs.version, len(staged), manifest_url,
        )
        return manifest_url
    finally:
        # 6. Cleanup staging. Local data/blobs is NOT touched here —
        # scripts/age_videos.py handles that separately so a
        # publish-then-crash doesn't delete the only copy.
        shutil.rmtree(staging_root, ignore_errors=True)
