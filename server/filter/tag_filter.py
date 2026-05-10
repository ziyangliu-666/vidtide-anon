import logging

from server.filter.base import BaseFilter

logger = logging.getLogger(__name__)

# Mapping from label_source strings to numeric tiers. Multiple sources can
# map to the same tier — that's how we distinguish "vendor-shown on an
# official showcase page" (tier1_gallery) from "creator uploaded to a
# trusted AI-dedicated channel/subreddit" (tier2_channel_whitelist) from
# "creator claimed a model in title/desc and platform let us crawl it"
# (tier2_platform_tag) while still sharing the same numeric-tier acceptance
# gate in pipeline.yaml.
_LABEL_SOURCE_TO_TIER: dict[str, int] = {
    "tier1_gallery": 1,
    "tier2_platform_tag": 2,
    "tier2_channel_whitelist": 2,
    "tier3_llm": 3,
}


class TagFilter(BaseFilter):
    """Keep only videos whose label-source tier is in the accepted set."""

    name = "tag"

    def __init__(self) -> None:
        self._total_in = 0
        self._total_out = 0
        self._by_tier: dict[int, int] = {}  # tier -> count of videos that passed

    def filter(self, candidates: list[dict], config: dict) -> list[dict]:
        accept_tiers: list[int] = config.get("accept_tiers", [1, 2])

        self._total_in += len(candidates)
        kept: list[dict] = []

        for c in candidates:
            label_source: str = c.get("label_source", "")
            tier = _LABEL_SOURCE_TO_TIER.get(label_source)

            if tier is None:
                logger.debug(
                    "TagFilter: unknown label_source '%s' for video %s, skipping",
                    label_source,
                    c.get("source_id", "?"),
                )
                continue

            if tier not in accept_tiers:
                logger.debug(
                    "TagFilter: tier %d not in accept_tiers for video %s",
                    tier,
                    c.get("source_id", "?"),
                )
                continue

            self._by_tier[tier] = self._by_tier.get(tier, 0) + 1
            kept.append(c)

        self._total_out += len(kept)
        return kept

    def stats(self) -> dict:
        return {
            "total_in": self._total_in,
            "total_out": self._total_out,
            "by_tier": dict(self._by_tier),
        }
