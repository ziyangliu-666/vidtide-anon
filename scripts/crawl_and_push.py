#!/usr/bin/env python3
"""Crawl videos locally and push metadata to VidTide cloud.

Usage:
    VIDTIDE_API_URL=https://your-vidtide-instance.example.com \
    VIDTIDE_API_KEY=your-key \
    python scripts/crawl_and_push.py --max-videos 20
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from server.pipeline.runner import PipelineRunner


def main():
    parser = argparse.ArgumentParser(description="Crawl and push to VidTide cloud")
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    api_url = os.environ.get("VIDTIDE_API_URL")
    api_key = os.environ.get("VIDTIDE_API_KEY")

    if not api_url:
        print("ERROR: Set VIDTIDE_API_URL environment variable")
        print("Example: VIDTIDE_API_URL=https://your-vidtide-instance.example.com")
        sys.exit(1)

    print(f"Target: {api_url}")
    print(f"API Key: {'set' if api_key else 'NOT SET (will fail if server requires it)'}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    runner = PipelineRunner(config=config, db_session=None, mode="remote")
    stats = runner.run(flow_type="crawl", max_videos=args.max_videos)

    print(f"\nDone! {stats}")


if __name__ == "__main__":
    main()
