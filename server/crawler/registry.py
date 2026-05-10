"""Crawler registry — single source of truth for which crawlers exist.

Each crawler module imports `register` from this module and decorates its
class. The runner then calls `load_enabled(config)` to instantiate every
crawler whose `crawl.platforms.<name>.enabled` flag is true.

The discovery step is config-driven: `load_enabled` lazy-imports
`server.crawler.<name>` so an optional dep failure (e.g. missing yt-dlp,
playwright, etc.) only disables that one crawler instead of breaking the
whole pipeline. This preserves the try/except guards previously hardcoded
in `runner._load_crawlers()`.
"""

from __future__ import annotations

import importlib
import logging
from typing import Callable, TypeVar

from server.crawler.base import BaseCrawler

logger = logging.getLogger(__name__)

CRAWLER_REGISTRY: dict[str, type[BaseCrawler]] = {}

T = TypeVar("T", bound=type[BaseCrawler])


def register(name: str) -> Callable[[T], T]:
    """Class decorator: register a crawler class under *name*."""

    def deco(cls: T) -> T:
        CRAWLER_REGISTRY[name] = cls
        return cls

    return deco


def load_enabled(config: dict) -> list[tuple[str, BaseCrawler, dict]]:
    """Instantiate every enabled crawler from `config["crawl"]["platforms"]`.

    Returns a list of (name, instance, platform_config) tuples in iteration
    order of the platforms dict. Crawlers whose module fails to import (e.g.
    missing optional deps) are logged and skipped, not raised.
    """
    platforms: dict = config.get("crawl", {}).get("platforms", {})
    crawlers: list[tuple[str, BaseCrawler, dict]] = []

    for name, pcfg in platforms.items():
        if not pcfg.get("enabled", False):
            continue

        # Lazy import — registers the class on first import.
        if name not in CRAWLER_REGISTRY:
            try:
                importlib.import_module(f"server.crawler.{name}")
            except ImportError:
                logger.warning(
                    "Crawler '%s' enabled in config but module import failed; skipping",
                    name,
                    exc_info=True,
                )
                continue

        cls = CRAWLER_REGISTRY.get(name)
        if cls is None:
            logger.warning(
                "Crawler '%s' enabled in config but not registered; skipping",
                name,
            )
            continue

        try:
            crawlers.append((name, cls(), pcfg))
        except Exception:
            logger.warning("Failed to instantiate crawler '%s'", name, exc_info=True)
            continue

    return crawlers
