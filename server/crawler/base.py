from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

@dataclass
class CrawledVideo:
    """A crawled video record before it lands in the DB / cloud import.

    See `server/crawler/__init__.py` for the full crawler-author checklist.
    The two fields with non-obvious conventions:

    - `source_url` MUST be a directly playable URL (CDN mp4/webm/mov/m4v)
      whenever the platform exposes one. The dashboard's `VideoEmbed` feeds
      this straight into an HTML5 `<video>` element via the smart-default
      branch. Storing a marketing-page URL or HTML wrapper here will break
      inline playback. For platforms that only expose an embed iframe
      (YouTube, Bilibili), `source_url` may be the watch page URL — the
      embed component switches on `source_platform` for those.

    - `content_tags` is the place for the umbrella-expansion convention:
      umbrella crawlers (showcase, social, ...) stamp a
      `<source_platform>:<source_key>` tag here so the dashboard can show
      the real vendor (`deepmind_veo`, `tiktok`, etc.) instead of the
      platform name. See `displayPlatform()` and `_expand_umbrella_platform`.
    """

    source_platform: str
    source_url: str
    source_id: str
    label: str                    # "fake" | "real" | "unknown"
    label_source: str             # tier1_gallery | tier2_platform_tag | tier2_channel_whitelist | tier3_llm
    title: str | None = None
    claimed_generator: str | None = None
    content_tags: list[str] = field(default_factory=list)
    published_at: str | None = None
    raw_metadata: dict = field(default_factory=dict)
    video_bytes: bytes | None = None
    download_url: str | None = None
    thumbnail_url: str | None = None
    # Locally-generated thumbnail JPEG bytes. Set by crawlers via
    # `server.crawler._thumbnail.extract_thumbnail(cdn_url)` so the runner
    # can ship them with the remote-push payload. This keeps the cloud Fly
    # machine off the ffmpeg hot path — it's memory-constrained and OOM-kills
    # uvicorn the moment ffmpeg runs in-process there.
    thumbnail_bytes: bytes | None = None
    duration_sec: float | None = None
    resolution_w: int | None = None
    resolution_h: int | None = None
    fps: float | None = None

class BaseCrawler(ABC):
    name: str
    tier: int

    @abstractmethod
    def crawl(self, config: dict) -> Iterator[CrawledVideo]:
        ...

    @abstractmethod
    def estimate_available(self, config: dict) -> int:
        ...
