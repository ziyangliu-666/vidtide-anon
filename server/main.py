"""RollingForge API -- FastAPI application entry-point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.config import get_settings
from server.db.migrations import run_migrations

from server.routers import (
    config,
    dedup,
    import_videos,
    pipeline,
    review,
    slices,
    stats,
    storage,
    thumbnails,
    videos,
)

# ---------------------------------------------------------------------------
# Lifespan: apply pending SQL migrations on startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Import models so SQLAlchemy's ORM registry is populated for downstream
    # session usage. (The schema itself is managed by `migrations/*.sql` via
    # server.db.migrations, NOT by Base.metadata.create_all — see
    # server/db/migrations.py for the rationale.)
    import server.db.models  # noqa: F401

    run_migrations(get_settings().db_path)

    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VidTide",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS -- allow the Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-vidtide-instance.example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(stats.router, prefix="/api")
app.include_router(videos.router, prefix="/api")
app.include_router(pipeline.router, prefix="/api")
app.include_router(review.router, prefix="/api")
app.include_router(slices.router, prefix="/api")
app.include_router(config.router, prefix="/api")
app.include_router(import_videos.router, prefix="/api")
app.include_router(thumbnails.router, prefix="/api")
app.include_router(dedup.router, prefix="/api")
app.include_router(storage.router, prefix="/api")


# ---------------------------------------------------------------------------
# Root + Config
# ---------------------------------------------------------------------------


@app.get("/")
def root():
    return {"name": "VidTide", "version": "0.1.0"}


