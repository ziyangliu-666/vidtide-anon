"""Pipeline router — list / trigger / cancel runs."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from server.db.database import SessionLocal, get_db
from server.db.models import PipelineRun

router = APIRouter(tags=["pipeline"])
logger = logging.getLogger(__name__)


# ---- Request / Response schemas ---------------------------------------------


class TriggerRunRequest(BaseModel):
    flow_type: str = "full"  # "full" | "crawl" | "process"
    max_videos: int = 20
    mode: str = "local"


class TriggerRunResponse(BaseModel):
    run_id: str
    status: str


class PipelineRunOut(BaseModel):
    id: str
    run_name: str | None = None
    flow_type: str
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    stats: dict[str, Any] | None = None
    created_at: str

    class Config:
        from_attributes = True


class CancelResponse(BaseModel):
    id: str
    status: str


# ---- Background execution ---------------------------------------------------


# Global lock to prevent concurrent pipeline runs. Only one run at a time
# is allowed because the runner does heavy per-video DB writes and we
# don't want race conditions on the single SQLite file.
_run_lock = threading.Lock()


def _execute_run(run_id: str, flow_type: str, max_videos: int, mode: str = "local") -> None:
    import os

    if mode == "remote" and os.environ.get("FLY_APP_NAME"):
        logger.warning("Ignoring mode=remote on production server, falling back to local")
        mode = "local"

    if not _run_lock.acquire(blocking=False):
        logger.warning("Another pipeline run is already active, skipping")
        return

    db = SessionLocal()
    try:
        from server.pipeline.runner import PipelineRunner

        config_path = "config/pipeline.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        runner = PipelineRunner(config, db, mode=mode)
        stats = runner.run(flow_type=flow_type, max_videos=max_videos)
        logger.info("Background pipeline completed: %s", stats)

    except Exception as exc:
        logger.error("Background pipeline failed", exc_info=True)
        run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
        if run and run.status == "running":
            run.status = "failed"
            run.completed_at = datetime.now(timezone.utc)
            run.stats = json.dumps({"error": str(exc)})
            db.commit()
    finally:
        _run_lock.release()
        db.close()


# ---- Routes -----------------------------------------------------------------


def _run_to_out(r: PipelineRun) -> PipelineRunOut:
    return PipelineRunOut(
        id=r.id,
        run_name=r.run_name,
        flow_type=r.flow_type,
        status=r.status,
        started_at=r.started_at.isoformat() if r.started_at else None,
        completed_at=r.completed_at.isoformat() if r.completed_at else None,
        stats=json.loads(r.stats) if r.stats else None,
        created_at=r.created_at.isoformat(),
    )


@router.get("/pipeline/runs", response_model=list[PipelineRunOut])
def list_runs(db: Session = Depends(get_db)) -> list[PipelineRunOut]:
    runs = db.query(PipelineRun).order_by(PipelineRun.created_at.desc()).limit(20).all()
    return [_run_to_out(r) for r in runs]


@router.get("/pipeline/runs/{run_id}", response_model=PipelineRunOut)
def get_run(run_id: str, db: Session = Depends(get_db)) -> PipelineRunOut:
    """Get a single run's details including live progress stats.

    The dashboard polls this every 2s while a run is 'running' to get
    real-time per-stage progress (the runner flushes stats after each
    video in slow stages).
    """
    run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return _run_to_out(run)


@router.post("/pipeline/run", response_model=TriggerRunResponse)
def trigger_run(
    body: TriggerRunRequest,
    db: Session = Depends(get_db),
) -> TriggerRunResponse:
    """Start a new pipeline run in a background thread.

    Rejects if another run is already in progress (returns 409). This
    prevents concurrent writes to the same SQLite DB — the per-video
    commit pattern is safe for ONE writer but not two.
    """
    # Check for an already-running run
    active = (
        db.query(PipelineRun)
        .filter(PipelineRun.status.in_(["running", "cancelling"]))
        .first()
    )
    if active:
        raise HTTPException(
            status_code=409,
            detail=f"Another run is already active (id={active.id}, status={active.status}). "
            "Cancel or wait for it to finish.",
        )

    temp_id = f"pending-{datetime.now(timezone.utc).strftime('%H%M%S')}"

    thread = threading.Thread(
        target=_execute_run,
        args=(temp_id, body.flow_type, body.max_videos, body.mode),
        daemon=True,
    )
    thread.start()

    return TriggerRunResponse(run_id=temp_id, status="started")


@router.post("/pipeline/runs/{run_id}/cancel", response_model=CancelResponse)
def cancel_run(
    run_id: str,
    db: Session = Depends(get_db),
) -> CancelResponse:
    """Request graceful cancellation of a running pipeline.

    Sets status to 'cancelling'. The runner checks this flag after each
    video and exits cleanly with status='paused'. Already-processed
    videos are committed and the next run resumes from where this one
    left off.

    If the run isn't 'running', this is a no-op (returns current status).
    """
    run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status == "running":
        run.status = "cancelling"
        db.commit()
        logger.info("pipeline run %s: cancel requested", run_id[:8])

    return CancelResponse(id=run.id, status=run.status)
