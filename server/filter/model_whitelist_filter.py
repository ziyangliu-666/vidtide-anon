"""ModelWhitelistFilter — gate candidates by claimed_generator.

Two modes:

* **lenient (default)**: reject only when `claimed_generator` is in the
  blacklist (known pre-2025 models). Pass missing/unknown values through.
  Goal — strip out the SVD/Sora-1/ModelScope slop without nuking the entire
  existing DB whose `claimed_generator` is sparsely populated.

* **strict (opt-in)**: keep only candidates whose `claimed_generator` is in
  the whitelist (the SOTA roster: Veo 3, Sora 2, Kling 2.x, Runway Gen-4,
  Pika 2, Hailuo, Hunyuan, Wan 2.1, etc.). Reserved for "freeze a benchmark
  slice with only confirmed SOTA content."

Reads `claimed_generator` from the dict IR built in
`server.pipeline.runner._filter()`. Phase 0.1 added the field to that IR.
"""

from __future__ import annotations

import logging

from server.filter.base import BaseFilter

logger = logging.getLogger(__name__)


class ModelWhitelistFilter(BaseFilter):
    name = "model_whitelist"

    def __init__(self) -> None:
        self._stats: dict = {
            "total_in": 0,
            "total_out": 0,
            "by_verdict": {
                "whitelisted": 0,
                "blacklisted": 0,
                "unknown": 0,
                "missing": 0,
            },
        }

    def filter(self, candidates: list[dict], config: dict) -> list[dict]:
        self._stats["total_in"] += len(candidates)

        if not config.get("enabled", False):
            self._stats["total_out"] += len(candidates)
            return candidates

        whitelist = {m.lower() for m in config.get("whitelist", [])}
        blacklist = {m.lower() for m in config.get("blacklist", [])}
        strict = bool(config.get("strict", False))

        kept: list[dict] = []
        verdicts = self._stats["by_verdict"]

        for c in candidates:
            raw = c.get("claimed_generator")
            gen = raw.lower() if isinstance(raw, str) and raw else None

            verdict, keep = self._classify(gen, whitelist, blacklist, strict)
            verdicts[verdict] += 1

            if keep:
                kept.append(c)
            else:
                logger.debug(
                    "ModelWhitelistFilter: REJECT %s (claimed_generator=%s, verdict=%s)",
                    c.get("source_id", "?"),
                    raw,
                    verdict,
                )

        self._stats["total_out"] += len(kept)
        return kept

    def stats(self) -> dict:
        return {
            "total_in": self._stats["total_in"],
            "total_out": self._stats["total_out"],
            "by_verdict": dict(self._stats["by_verdict"]),
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _classify(
        gen: str | None,
        whitelist: set[str],
        blacklist: set[str],
        strict: bool,
    ) -> tuple[str, bool]:
        """Return (verdict_label, keep_decision).

        verdict_label is one of: whitelisted, blacklisted, unknown, missing.
        """
        if gen is None:
            # Lenient: pass missing through. Strict: reject.
            return ("missing", not strict)

        if gen in blacklist:
            return ("blacklisted", False)

        if gen in whitelist:
            return ("whitelisted", True)

        # Known model name but not in either list. Lenient passes; strict rejects.
        return ("unknown", not strict)
