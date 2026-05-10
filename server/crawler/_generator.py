"""Shared generator-name extraction across all crawlers.

The extracted name is the *canonical model id* used by ModelWhitelistFilter.
Pattern order matters — more-specific patterns must come first so generic
fallbacks (e.g. bare "Veo") don't shadow versioned matches (e.g. "Veo 3").

Canonical ids align with config/pipeline.yaml `filter.model_whitelist`:

  Modern / SOTA (2025-2026):
    sora2, veo3, veo2, kling21, kling2, runway-gen4, runway-gen3,
    pika2, hailuo, hunyuan, wan21, ltxv, mochi1, dreamina3

  Legacy / pre-2025 (blacklist):
    sora1, runway_gen2, pika1, stable_video_diffusion,
    animatediff, zeroscope, modelscope_t2v, kling1

Bare model names with no version (e.g. "made with Veo") resolve to the
*latest known* public version, on the assumption that 2026 posts saying
"Veo" almost always mean Veo 3.
"""

from __future__ import annotations

import re

# Each entry: (compiled regex, canonical id). Order = priority. The first
# match wins, so versioned patterns must precede their unversioned siblings.
_GENERATOR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # ---- Sora ----
    (re.compile(r"\bsora\s*2(?:\.0)?\b", re.IGNORECASE), "sora2"),
    (re.compile(r"\bsora\s*1(?:\.0)?\b", re.IGNORECASE), "sora1"),
    (re.compile(r"\bsora\b", re.IGNORECASE), "sora2"),  # bare → latest
    # ---- Google Veo ----
    (re.compile(r"\bveo[\s\-]*3\b", re.IGNORECASE), "veo3"),
    (re.compile(r"\bveo[\s\-]*2\b", re.IGNORECASE), "veo2"),
    (re.compile(r"\bveo\b", re.IGNORECASE), "veo3"),  # bare → latest
    # ---- Kling (Kuaishou) ----
    (re.compile(r"\bkling[\s\-]*2\.1\b", re.IGNORECASE), "kling21"),
    (re.compile(r"\bkling[\s\-]*2(?:\.0)?\b", re.IGNORECASE), "kling2"),
    (re.compile(r"\bkling[\s\-]*1(?:\.\d)?\b", re.IGNORECASE), "kling1"),
    (re.compile(r"可灵\s*2\.1", re.IGNORECASE), "kling21"),
    (re.compile(r"可灵\s*2", re.IGNORECASE), "kling2"),
    (re.compile(r"可灵|kling", re.IGNORECASE), "kling21"),  # bare → latest
    # ---- Runway ----
    (re.compile(r"\b(?:runway[\s\-]*)?gen[\s\-]*4\b", re.IGNORECASE), "runway-gen4"),
    (re.compile(r"\b(?:runway[\s\-]*)?gen[\s\-]*3\b", re.IGNORECASE), "runway-gen3"),
    (re.compile(r"\b(?:runway[\s\-]*)?gen[\s\-]*2\b", re.IGNORECASE), "runway_gen2"),
    (re.compile(r"\brunway\b", re.IGNORECASE), "runway-gen4"),  # bare → latest
    # ---- Pika ----
    (re.compile(r"\bpika[\s\-]*2(?:\.\d)?\b", re.IGNORECASE), "pika2"),
    (re.compile(r"\bpika[\s\-]*1(?:\.\d)?\b", re.IGNORECASE), "pika1"),
    (re.compile(r"\bpika\b", re.IGNORECASE), "pika2"),  # bare → latest
    # ---- MiniMax / Hailuo ----
    (re.compile(r"\bhailuo\b", re.IGNORECASE), "hailuo"),
    (re.compile(r"\bminimax\b", re.IGNORECASE), "hailuo"),  # MiniMax video = Hailuo
    # ---- Tencent Hunyuan ----
    (re.compile(r"\bhunyuan(?:[\s\-]*video)?\b", re.IGNORECASE), "hunyuan"),
    (re.compile(r"混元", re.IGNORECASE), "hunyuan"),
    # ---- Alibaba Wan ----
    (re.compile(r"\bwan[\s\-]*2\.1\b", re.IGNORECASE), "wan21"),
    (re.compile(r"\bwan2\.1\b", re.IGNORECASE), "wan21"),
    (re.compile(r"通义\s*万相", re.IGNORECASE), "wan21"),
    # ---- Lightricks LTX-Video ----
    (re.compile(r"\bltx[\s\-]*video\b", re.IGNORECASE), "ltxv"),
    (re.compile(r"\bltxv\b", re.IGNORECASE), "ltxv"),
    # ---- Genmo Mochi ----
    (re.compile(r"\bmochi[\s\-]*1\b", re.IGNORECASE), "mochi1"),
    (re.compile(r"\bmochi\b", re.IGNORECASE), "mochi1"),
    # ---- ByteDance Dreamina / Jimeng ----
    (re.compile(r"\bdreamina[\s\-]*3\b", re.IGNORECASE), "dreamina3"),
    (re.compile(r"\bdreamina\b", re.IGNORECASE), "dreamina3"),
    (re.compile(r"\bjimeng\b", re.IGNORECASE), "dreamina3"),
    (re.compile(r"即梦", re.IGNORECASE), "dreamina3"),
    # ---- Luma Dream Machine ----
    (re.compile(r"\bdream[\s\-]*machine\b", re.IGNORECASE), "luma"),
    (re.compile(r"\bluma\b", re.IGNORECASE), "luma"),
    # ---- Vidu / PixVerse (kept for completeness, not in default whitelist) ----
    (re.compile(r"\bvidu\b", re.IGNORECASE), "vidu"),
    (re.compile(r"\bpixverse\b", re.IGNORECASE), "pixverse"),
    # ---- Legacy / pre-2025 (will be blacklisted) ----
    (re.compile(r"\bstable[\s\-]*video(?:[\s\-]*diffusion)?\b", re.IGNORECASE), "stable_video_diffusion"),
    (re.compile(r"\bsvd\b", re.IGNORECASE), "stable_video_diffusion"),
    (re.compile(r"\banimate[\s\-]*diff\b", re.IGNORECASE), "animatediff"),
    (re.compile(r"\bzeroscope\b", re.IGNORECASE), "zeroscope"),
    (re.compile(r"\bmodel[\s\-]*scope(?:[\s\-]*t2v)?\b", re.IGNORECASE), "modelscope_t2v"),
]


def extract_generator(text: str) -> str | None:
    """Identify the claimed generator from free-form text.

    Returns a canonical id (matching ModelWhitelistFilter's whitelist/blacklist),
    or None if no known model name appears.
    """
    if not text:
        return None
    for pattern, name in _GENERATOR_PATTERNS:
        if pattern.search(text):
            return name
    return None
