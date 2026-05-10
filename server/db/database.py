"""SQLAlchemy engine, session factory, and FastAPI dependency."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from server.config import get_settings

_settings = get_settings()

# Ensure the directory for the SQLite file exists.
_db_file = Path(_settings.db_path)
_db_file.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    f"sqlite:///{_db_file}",
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a DB session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
