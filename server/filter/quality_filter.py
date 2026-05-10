import logging

from server.filter.base import BaseFilter

logger = logging.getLogger(__name__)


class QualityFilter(BaseFilter):
    """Remove videos that do not meet minimum quality thresholds."""

    name = "quality"

    def __init__(self) -> None:
        self._total_in = 0
        self._total_out = 0
        self._removed_resolution = 0
        self._removed_duration = 0
        self._removed_corrupt = 0

    def filter(self, candidates: list[dict], config: dict) -> list[dict]:
        # `min_resolution` is the long-edge threshold (HD = 720). `min_short_edge`
        # is the short-edge threshold and exists so portrait clips (e.g. 576x1024
        # TikToks) can pass at HD long-edge without being rejected by their narrow
        # width. Default short edge of 400 keeps phone-format content viable while
        # still excluding tiny thumbnails.
        min_long_edge: int = config.get("min_resolution", 360)
        min_short_edge: int = config.get("min_short_edge", min(400, min_long_edge))
        min_duration: float = config.get("min_duration", 1.0)
        max_duration: float = config.get("max_duration", 300.0)

        self._total_in += len(candidates)
        kept: list[dict] = []

        for c in candidates:
            # Corrupt / empty file check — skip when file_size is unknown (metadata-only).
            file_size = c.get("file_size_bytes")
            if file_size is not None and file_size == 0:
                self._removed_corrupt += 1
                logger.debug("QualityFilter: removed corrupt/empty video %s", c.get("source_id", "?"))
                continue

            # Resolution check — long edge must meet HD threshold, short edge
            # must meet the (more permissive) short-edge threshold. Works for
            # both landscape and portrait sources. When BOTH dimensions are
            # unknown (metadata-only crawl, e.g. showcase pages), skip the
            # check and let a later download/probe stage verify.
            res_w = c.get("resolution_w") or 0
            res_h = c.get("resolution_h") or 0
            if res_w > 0 and res_h > 0:
                long_edge = max(res_w, res_h)
                short_edge = min(res_w, res_h)
                if long_edge < min_long_edge or short_edge < min_short_edge:
                    self._removed_resolution += 1
                    logger.debug(
                        "QualityFilter: removed low-res video %s (%dx%d, long<%d or short<%d)",
                        c.get("source_id", "?"),
                        res_w,
                        res_h,
                        min_long_edge,
                        min_short_edge,
                    )
                    continue

            # Duration check. Unknown duration (None) passes through — let a
            # later probe stage verify. Zero duration is treated as known-bad.
            duration_raw = c.get("duration_sec")
            if duration_raw is not None:
                duration = float(duration_raw)
                if duration < min_duration or duration > max_duration:
                    self._removed_duration += 1
                    logger.debug(
                        "QualityFilter: removed out-of-range duration video %s (%.1fs)",
                        c.get("source_id", "?"),
                        duration,
                    )
                    continue

            kept.append(c)

        self._total_out += len(kept)
        return kept

    def stats(self) -> dict:
        return {
            "total_in": self._total_in,
            "total_out": self._total_out,
            "removed_resolution": self._removed_resolution,
            "removed_duration": self._removed_duration,
            "removed_corrupt": self._removed_corrupt,
        }
