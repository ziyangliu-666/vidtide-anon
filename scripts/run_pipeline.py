#!/usr/bin/env python3
"""CLI entry point for running the RollingForge data pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Ensure the project root is on sys.path so that `server.*` imports work
# when running this script directly.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from server.config import get_settings
from server.db.database import SessionLocal
from server.db.migrations import run_migrations
from server.pipeline.runner import PipelineRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the RollingForge data pipeline.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline.yaml",
        help="Path to pipeline YAML config (default: config/pipeline.yaml)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["crawl", "full", "process"],
        default="full",
        help="Pipeline flow type: 'crawl' (crawl only), 'full' (crawl -> filter -> dedup -> download), or 'process' (dedup + download on existing filtered videos). Default: full",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum number of videos to crawl (overrides config value)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("run_pipeline")

    # Load config
    config_path = Path(args.config)
    if not config_path.is_file():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded config from %s", config_path)

    # Resolve max_videos: CLI arg > config > default 50
    max_videos = args.max_videos
    if max_videos is None:
        crawl_cfg = config.get("crawl", {}).get("platforms", {}).get("youtube", {})
        max_videos = crawl_cfg.get("max_videos", 50)

    # Apply any pending migrations. This replaces the old
    # Base.metadata.create_all() path — schema is now governed by the
    # numbered migrations/*.sql files via server.db.migrations.
    applied = run_migrations(get_settings().db_path)
    logger.info("Migrations applied: %d", applied)

    # Create DB session
    db = SessionLocal()
    try:
        # Run pipeline
        runner = PipelineRunner(config=config, db_session=db)
        logger.info(
            "Starting pipeline: stage=%s, max_videos=%d",
            args.stage,
            max_videos,
        )
        stats = runner.run(flow_type=args.stage, max_videos=max_videos)

        # Print summary
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"  Run ID:    {stats.get('run_id', 'N/A')}")
        print(f"  Flow type: {stats.get('flow_type', 'N/A')}")
        print(f"  Duration:  {stats.get('total_duration_sec', 'N/A')}s")
        print()

        stages = stats.get("stages", {})
        for stage_name, stage_stats in stages.items():
            print(f"  [{stage_name}]")
            # Print all non-duration keys generically so dedup (which
            # reports processed/captioned/duplicates_found instead of
            # a flat count) shows its full stats without a special case.
            for key, val in stage_stats.items():
                if key == "duration_sec":
                    continue
                print(f"    {key}: {val}")
            print(f"    duration: {stage_stats.get('duration_sec', 0)}s")

        print("=" * 60)
        print(f"\nFull stats: {json.dumps(stats, indent=2)}")

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(130)
    except Exception:
        logger.error("Pipeline failed", exc_info=True)
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
