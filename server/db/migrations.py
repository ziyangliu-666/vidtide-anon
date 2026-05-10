"""Lightweight forward-only SQL migrations.

Why not Alembic: this project is a single-developer SQLite research tool
pre-v1. Alembic's autogenerate, revision graphs, and Python DSL would
outweigh the value. A numbered `migrations/NNNN_name.sql` directory +
60-line runner is easier to review in PRs and easier to debug with the
`sqlite3` CLI.

Contract:
- Every migration is one `.sql` file at the project root `migrations/`
  directory, named `NNNN_description.sql` where NNNN is a zero-padded
  4-digit integer. Files are applied in sorted order.
- Each file is applied inside a single transaction. If any statement in
  it fails, the whole file rolls back and the migration is NOT recorded.
- Applied migrations are recorded in `schema_migrations(version, name,
  applied_at)`. `run_migrations()` is idempotent — already-applied
  versions are skipped.
- Migrations are forward-only. There is no `down`. If you need to revert,
  write a new forward migration that undoes the change.

Typical call site: `server/main.py` lifespan hook.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Project root resolution: this file lives at server/db/migrations.py,
# migrations/ is a sibling of server/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations"


def _ensure_schema_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )


def _already_applied(conn: sqlite3.Connection) -> set[int]:
    cursor = conn.execute("SELECT version FROM schema_migrations")
    return {row[0] for row in cursor.fetchall()}


def _parse_version(path: Path) -> tuple[int, str] | None:
    """Parse `NNNN_name.sql` into (version, name). Return None on malformed."""
    stem = path.stem  # e.g. "0001_initial"
    if len(stem) < 5 or stem[4] != "_":
        return None
    try:
        version = int(stem[:4])
    except ValueError:
        return None
    name = stem[5:]
    return version, name


def discover_migrations(directory: Path = _MIGRATIONS_DIR) -> list[tuple[int, str, Path]]:
    """Return sorted list of (version, name, path) for all valid migration files."""
    if not directory.exists():
        return []
    entries: list[tuple[int, str, Path]] = []
    for path in directory.glob("*.sql"):
        parsed = _parse_version(path)
        if parsed is None:
            logger.warning("migrations: skipping malformed filename %s", path.name)
            continue
        version, name = parsed
        entries.append((version, name, path))
    entries.sort(key=lambda t: t[0])
    return entries


def run_migrations(db_path: str, directory: Path = _MIGRATIONS_DIR) -> int:
    """Apply all unapplied migrations in order. Return the count applied.

    Each migration runs inside an implicit transaction — sqlite3's default
    isolation level wraps `executescript()` in BEGIN/COMMIT so a partial
    failure rolls the whole file back cleanly.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        _ensure_schema_table(conn)
        applied = _already_applied(conn)
        migrations = discover_migrations(directory)
        count = 0
        for version, name, path in migrations:
            if version in applied:
                continue
            sql = path.read_text(encoding="utf-8")
            logger.info("migrations: applying %04d_%s", version, name)
            try:
                conn.executescript(sql)
                conn.execute(
                    "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                    (version, name, datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
                count += 1
            except sqlite3.Error:
                conn.rollback()
                logger.error("migrations: failed to apply %04d_%s", version, name, exc_info=True)
                raise
        logger.info("migrations: applied %d migration(s)", count)
        return count
    finally:
        conn.close()
