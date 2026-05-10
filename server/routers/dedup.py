"""Dedup cluster browsing API.

Clusters aren't stored explicitly — they're derived as the transitive
closure of `duplicate_of_id` pointers. Since we commit `duplicate_of_id`
as a flat pointer (no cluster_id column), a canonical is any row with
`duplicate_of_id IS NULL` that is referenced by at least one other row's
`duplicate_of_id`. Single rows with no duplicates aren't "clusters" —
they're just uncontested canonicals, and the /clusters endpoint hides
them by default.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from server.config import get_settings
from server.db.database import get_db
from server.db.models import Video

router = APIRouter(tags=["dedup"])


# ---- Schemas ----------------------------------------------------------------


class ClusterMember(BaseModel):
    id: str
    source_platform: str
    source_url: str
    source_id: str
    title: str | None = None
    thumbnail_url: str | None = None
    caption_text: str | None = None
    resolution_w: int | None = None
    resolution_h: int | None = None
    fps: float | None = None
    duration_sec: float | None = None
    claimed_generator: str | None = None
    is_canonical: bool


class DedupCluster(BaseModel):
    canonical_id: str
    member_count: int
    members: list[ClusterMember]


class ClustersResponse(BaseModel):
    clusters: list[DedupCluster]
    total_canonicals: int
    total_duplicates: int


class DedupStatsResponse(BaseModel):
    total_canonicals: int
    total_duplicates: int
    captioned: int
    caption_models: dict[str, int]


class SearchHit(BaseModel):
    id: str
    source_platform: str
    source_url: str
    source_id: str
    title: str | None = None
    thumbnail_url: str | None = None
    claimed_generator: str | None = None
    resolution_w: int | None = None
    resolution_h: int | None = None
    duration_sec: float | None = None
    similarity: float  # 0-1, higher = more similar


class SearchResponse(BaseModel):
    query: ClusterMember
    results: list[SearchHit]
    vec_index_size: int


# ---- Routes -----------------------------------------------------------------


def _video_to_member(v: Video, is_canonical: bool) -> ClusterMember:
    return ClusterMember(
        id=v.id,
        source_platform=v.source_platform,
        source_url=v.source_url,
        source_id=v.source_id,
        title=v.title,
        thumbnail_url=v.thumbnail_url,
        caption_text=v.caption_text,
        resolution_w=v.resolution_w,
        resolution_h=v.resolution_h,
        fps=v.fps,
        duration_sec=v.duration_sec,
        claimed_generator=v.claimed_generator,
        is_canonical=is_canonical,
    )


@router.get("/dedup/clusters", response_model=ClustersResponse)
def list_clusters(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> ClustersResponse:
    """Return dedup clusters (canonical + ≥ 1 duplicate).

    A cluster is any canonical video that is pointed at by at least one
    other row's `duplicate_of_id`. Uncontested canonicals (no
    duplicates) are hidden because they'd make the page a list of every
    video in the DB.

    Orders by duplicate count descending — the most interesting
    collisions surface first.
    """
    # Find canonical ids that have at least one duplicate
    canonical_ids_with_dups = (
        db.query(Video.duplicate_of_id)
        .filter(Video.duplicate_of_id.isnot(None))
        .distinct()
        .all()
    )
    canonical_ids = [row[0] for row in canonical_ids_with_dups]

    if not canonical_ids:
        total_canonicals = (
            db.query(Video).filter(Video.duplicate_of_id.is_(None)).count()
        )
        return ClustersResponse(
            clusters=[],
            total_canonicals=total_canonicals,
            total_duplicates=0,
        )

    canonicals = (
        db.query(Video).filter(Video.id.in_(canonical_ids)).all()
    )
    canonical_by_id = {v.id: v for v in canonicals}

    # Grouping: fetch all duplicates in one query, group by duplicate_of_id
    dup_rows = (
        db.query(Video)
        .filter(Video.duplicate_of_id.in_(canonical_ids))
        .all()
    )
    dups_by_canonical: dict[str, list[Video]] = {}
    for d in dup_rows:
        dups_by_canonical.setdefault(d.duplicate_of_id, []).append(d)

    clusters = []
    for canonical_id, dups in dups_by_canonical.items():
        canonical = canonical_by_id.get(canonical_id)
        if canonical is None:
            # Stale pointer — canonical row was deleted somehow; skip.
            continue
        members = [_video_to_member(canonical, is_canonical=True)]
        members.extend(_video_to_member(d, is_canonical=False) for d in dups)
        clusters.append(
            DedupCluster(
                canonical_id=canonical.id,
                member_count=len(members),
                members=members,
            )
        )

    # Sort by cluster size desc
    clusters.sort(key=lambda c: c.member_count, reverse=True)
    clusters = clusters[:limit]

    total_canonicals = (
        db.query(Video).filter(Video.duplicate_of_id.is_(None)).count()
    )
    total_duplicates = (
        db.query(Video).filter(Video.duplicate_of_id.isnot(None)).count()
    )
    return ClustersResponse(
        clusters=clusters,
        total_canonicals=total_canonicals,
        total_duplicates=total_duplicates,
    )


@router.get("/dedup/stats", response_model=DedupStatsResponse)
def dedup_stats(db: Session = Depends(get_db)) -> DedupStatsResponse:
    total_canonicals = (
        db.query(Video).filter(Video.duplicate_of_id.is_(None)).count()
    )
    total_duplicates = (
        db.query(Video).filter(Video.duplicate_of_id.isnot(None)).count()
    )
    captioned = (
        db.query(Video).filter(Video.caption_text.isnot(None)).count()
    )
    model_rows = (
        db.query(Video.caption_model, Video.id)
        .filter(Video.caption_model.isnot(None))
        .all()
    )
    caption_models: dict[str, int] = {}
    for model, _ in model_rows:
        caption_models[model] = caption_models.get(model, 0) + 1

    return DedupStatsResponse(
        total_canonicals=total_canonicals,
        total_duplicates=total_duplicates,
        captioned=captioned,
        caption_models=caption_models,
    )


@router.get("/dedup/search", response_model=SearchResponse)
def search_similar(
    video_id: str = Query(..., description="Video ID to find similar videos for"),
    k: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
) -> SearchResponse:
    """Semantic similarity search: given a video, find the K most similar
    videos by CLIP thumbnail embedding.

    Reads from the vec_thumbnails sqlite-vec virtual table. If the video
    isn't in the index (dedup hasn't run on it), returns an empty result
    set with a hint in the query member.

    This endpoint requires sqlite-vec to be loaded, which means it works
    on the crawler host (where dedup runs) but NOT on the Fly cloud
    container (which doesn't have the vec_thumbnails table populated).
    The frontend gracefully handles 500s from Fly by showing a message.
    """
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    query_member = _video_to_member(video, is_canonical=video.duplicate_of_id is None)

    settings = get_settings()
    try:
        from server.dedup.vec_index import VecIndex

        with VecIndex(settings.db_path) as index:
            index.ensure_table()
            vec_count = index.count()

            if vec_count == 0:
                return SearchResponse(query=query_member, results=[], vec_index_size=0)

            # Look up this video's embedding in the index
            # KNN returns [(video_id, distance), ...]
            neighbors = index.knn(
                # We need the query vector — read it from the index.
                # sqlite-vec doesn't expose a "get vector by key" API,
                # so we do a self-KNN with k=1 to verify the video is
                # in the index, then a real KNN with k+1 excluding self.
                _get_embedding_for(index, video_id),
                k=k,
                exclude_video_id=video_id,
            )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"vec_thumbnails not available: {exc}",
        ) from exc

    results: list[SearchHit] = []
    for neighbor_id, distance in neighbors:
        sim = round(1.0 - distance, 4)
        v = db.query(Video).filter(Video.id == neighbor_id).first()
        if v is None:
            continue
        results.append(
            SearchHit(
                id=v.id,
                source_platform=v.source_platform,
                source_url=v.source_url,
                source_id=v.source_id,
                title=v.title,
                thumbnail_url=v.thumbnail_url,
                claimed_generator=v.claimed_generator,
                resolution_w=v.resolution_w,
                resolution_h=v.resolution_h,
                duration_sec=v.duration_sec,
                similarity=sim,
            )
        )

    return SearchResponse(
        query=query_member, results=results, vec_index_size=vec_count,
    )


def _get_embedding_for(index, video_id: str) -> list[float]:
    """Read a video's embedding from the vec index.

    sqlite-vec doesn't have a direct "get by key" operation, so we
    query the shadow table that vec0 maintains internally.
    """
    import struct
    from server.dedup.vec_index import EMBEDDING_DIM

    conn = index._get_conn()
    row = conn.execute(
        "SELECT embedding FROM vec_thumbnails WHERE video_id = ?",
        (video_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"video {video_id} not in vec index")
    raw = row[0]
    return list(struct.unpack(f"{EMBEDDING_DIM}f", raw))
