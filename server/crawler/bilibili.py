"""Bilibili crawler — AI videos via search + region newlist, `argue_msg`-gated.

Design notes
============

**Two discovery paths**

The crawler pipes candidates from two Bilibili endpoints through the
same `argue_info.argue_msg` platform-AI gate:

1. **Keyword search** (`/x/web-interface/search/type`) — up to 1000
   results per query, paginated 50/page, order=pubdate. High argue_msg
   hit rate (AI-titled videos are usually AI), but saturates at ~700
   unique survivors across the whole AI-keyword bucket because the same
   bvids appear under many queries.

2. **Region newlist** (`/x/web-interface/newlist?rid=X`) — deep paging
   across every non-gaming zone. At page 50 of region 1 we're ~1 minute
   behind the newest upload, so effectively unlimited recent volume.
   Lower per-candidate argue_msg hit rate than keyword search, but the
   raw candidate pool is orders of magnitude larger — this is how bulk
   crawls get past the keyword-search ceiling.

Gaming regions are excluded from `_DEFAULT_REGIONS`, and a per-video
gaming-tag check still runs as a second gate (gamers post AI-titled
gameplay to non-gaming zones).

**Argue_msg is the ground truth**

Only videos with Bilibili's server-side `argue_info.argue_msg` AI
synthesis warning are accepted. Keyword match alone produces too many
false positives (tutorials, news about AI, gamers joking about AI).
Even with argue_msg we still check tags for gaming keywords because the
platform's detection false-positives on gameplay titled as "AI".

**Concurrency**

Per-candidate platform + tag checks go through a ThreadPoolExecutor
(default 8 workers) so the expanded region candidate pool isn't
bottlenecked on sequential ~100ms API calls. requests.Session is
thread-safe and shared across workers.

**Resolution left None**

Neither endpoint returns width/height reliably; `QualityFilter` handles
None gracefully. Duration is parsed from the response — "H:MM:SS"
string for search, int-seconds for newlist.
"""

from __future__ import annotations

import html as _html
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

import requests

from server.crawler._generator import extract_generator as _extract_generator
from server.crawler._thumbnail import fetch_thumbnail_bytes
from server.crawler.base import BaseCrawler, CrawledVideo
from server.crawler.registry import register

logger = logging.getLogger(__name__)

# AI-keyword bucket used when config doesn't specify `search_queries`.
# Model-name-heavy because specific model names have the highest
# per-result argue_msg hit rate; broader recall comes from the region
# newlist path, not from stuffing more generic keywords in here.
_DEFAULT_QUERIES = [
    # Model-specific (Chinese)
    "可灵AI视频", "可灵2.1", "可灵3.0",
    "即梦AI", "即梦3.0",
    "通义万相", "智影AI",
    # Model-specific (international)
    "Sora 2", "Sora视频", "veo3", "veo4",
    "runway gen4", "runway gen3",
    "Pika AI", "Hailuo minimax", "Hunyuan Video",
    "Kling 2.1", "Seedance", "Luma Dream Machine",
    # Generic AI creation
    "AI视频", "AIGC", "AI短片", "AI动画",
    "AI创作", "AI合成视频", "AI生成视频",
    "数字人视频", "虚拟主播",
    # Subject + AI (broader recall — argue_msg filters)
    "AI科幻", "AI机甲", "AI古风", "AI特效",
    "AI美女", "AI恐龙", "AI梦境",
    # Music & MV
    "AI音乐MV", "AI MV",
    # Events
    "AI创作大赛", "B站AI", "AIGC创作大赛",
]

# Default region (tid) bucket for the newlist browsing path. Gaming
# zones (tid=4 and children) are intentionally omitted here; a
# per-video gaming-tag check also runs downstream as a second gate.
#
# Region mapping:
#   1   综合 (animation umbrella)
#   3   MV / music
#   5   娱乐
#   36  人文历史
#   119 鬼畜
#   129 舞蹈
#   155 时尚
#   160 生活
#   181 影视剪辑
#   188 科技
#   202 资讯
#   211 美食
_DEFAULT_REGIONS = [1, 3, 5, 36, 119, 129, 155, 160, 181, 188, 202, 211]

# Title / description exclusion rules — drop tutorial & reaction content.
# Bilibili AI-keyword search is dense with "2 hours to master AI video
# creation"-style screen recordings; we only want raw generated clips.
_EXCLUDE_TITLE_TERMS = [
    "教程", "教学", "入门", "保姆级", "手把手", "全流程", "秘籍", "方法",
    "攻略", "测评", "评测", "对比", "反应", "盘点", "讲解", "解说",
    "教会你", "使用方法", "使用指南", "工作流", "comfyui", "comfy ui",
    # Tool-pack / deployment distribution — creator-packaged software bundles,
    # not generative video content.
    "整合包", "一键包", "一键整合", "本地部署",
    "免安装", "附安装包",
    "tutorial", "how to", "guide", "review", "reaction", "tips",
    "step by step", "step-by-step", "walkthrough", "beginner",
]

# VTuber exclusion — A-SOUL / Hololive / Neuro-sama / 虚拟主播 etc. are
# dedicated virtual-idol / streamer content, not AI-generated videos. Even
# when their tags include "AI" (e.g. AI-generated dance covers of an A-SOUL
# character), the domain is categorically different: stylized avatars on
# live-streaming platforms, not generative video output. Drop at title/tag
# level before the platform gate. Covered at two levels:
#
#   1. Title terms: character names and the VTuber industry vocabulary.
#   2. Tag keywords: generic VTuber markers independent of character name.
_VTUBER_TITLE_TERMS = [
    # A-SOUL members (romanized + CN)
    "a-soul", "asoul",
    "东雪莲", "東雪蓮",
    "阿梓", "嘉然", "乃琳", "贝拉", "珈乐", "向晚",
    "嘉心糖", "乃贝",
    # Hololive / Neuro / misc VTubers
    "hololive", "neuro-sama", "vedal",
    "夏色まつり", "神楽めあ", "月ノ美兎", "月之美兔",
    "にじさんじ", "彩虹社",
]

_VTUBER_TAG_KEYWORDS = {
    "vtuber", "vtube", "vup",
    "虚拟主播", "虚拟偶像", "虚拟up主",
    "虚拟youtuber", "虚拟yt",
    "管人痴", "皮套", "中之人",
    "hololive", "a-soul", "asoul",
    "にじさんじ", "彩虹社",
    "neuro-sama", "vedal",
}

# AI anchor / AI digital-human-as-host / AI livestream-sales — same domain
# category as VTuber: avatar-based presenter/streamer content, not generative
# AI video output. Also catches face-swap / lip-sync tool demos. Applied at
# both title and tag level.
_AI_ANCHOR_TITLE_TERMS = [
    "ai主播", "ai女主播", "ai男主播",
    "ai数字人", "ai虚拟主播",
    "数字人主播", "数字主播", "ai数字分身",
    "ai带货", "ai播报", "ai播音",
    "ai新闻主播", "ai直播间",
]

_AI_ANCHOR_TAG_KEYWORDS = {
    "ai主播", "ai女主播", "ai男主播",
    "ai数字人",
    "数字人主播", "数字主播", "数字分身",
    "ai带货", "ai播报", "ai播音",
    "ai新闻主播", "ai虚拟主播",
}


# Tutorial tag keywords — Bilibili creators flag their own tutorial content
# with tags like "教程", "ai教学", "sd学习", "comfyui教程". Titles often
# frame the tutorial as content ("学做AI动画一月到手一年房租") and the
# title-only filter misses it. Post-gate tag match is the safety net.
# No video-gen-tool safety guard: a "教程 + Kling" tag combo is still a
# Kling-usage tutorial, not a benchmark candidate.
_TUTORIAL_TAG_KEYWORDS = {
    "教程", "教学",
    "入门教程", "入门教学",
    "保姆级", "手把手", "教会你",
    "跟我学", "学习ai", "ai学习", "学ai",
    "ai教程", "ai教学",
    "sd教程", "sd教学", "sd学习",
    "comfyui教程", "comfyui教学", "comfyui入门",
    "工作流教程", "提示词教学",
    "tutorial", "beginner", "walkthrough",
}

# AI music exclusion — Bilibili's argue_msg fires on AI-generated AUDIO
# just as readily as AI-generated VIDEO. Suno song covers, AI翻唱
# (AI voice cover), AI歌手 (AI singer), and AI music competition entries
# are categorically different from text-to-video and don't belong in the
# benchmark. Exclude at two levels:
#
#   1. Title patterns (Phase 1, cheap): bracket-prefix naming convention
#      used by Suno/music creators, and explicit 翻唱/歌手/cover labels.
#   2. Tag keywords (Phase 2.5, post-gate): checked only when no video
#      generation tool is present in resolved tags — so "Pika video with
#      Suno background music" is kept while "pure Suno cover" is dropped.

_AI_MUSIC_TITLE_TERMS = [
    "[suno",     # [Suno V5]xxx, [Suno]xxx, [suno ai]xxx
    "【suno",    # 【suno ai】xxx — Japanese/Chinese brackets
    "ai翻唱",    # AI voice cover
    "ai歌手",    # AI singer
    "ai配乐",    # AI background music
    "ai cover",  # English AI cover naming
    "ai歌曲",    # AI song
    "ai音乐生成", # AI music generation
    # Explicit composition/arrangement/writing — audio-only output.
    "ai作曲", "ai编曲", "ai作词", "ai写歌",
    "ai原创歌曲", "ai原创音乐",
]

# Tags that, in the absence of any video-gen tool tag, indicate the
# video is primarily about music production, NOT AI video generation.
# Combined with the "no video-gen tool" safety guard in
# _is_ai_music_tags_only(), this catches rows the title filter misses
# (e.g. title is neutral but creator tagged it "翻唱" or "说唱").
#
# Do NOT add broad tags like "音乐" (background music in AI videos) or
# "mv" (AI-generated music videos are valid entries).
_AI_MUSIC_TAG_KEYWORDS = {
    # Explicit AI-music generation signals
    "ai翻唱",           # AI voice cover
    "ai歌手",           # AI singer
    "ai配乐",           # AI background music composition
    "ai歌曲",           # AI song
    "ai cover",         # English AI cover
    "sunoai",
    # B站 music competition series (pure audio generation contests)
    "ai音乐征集大赛",   # explicitly AI music competition
    "全能音乐挑战赛",   # music challenge
    "ai虚拟之声实验室",  # B站 AI voice lab contest
    # NOTE: "b站ai创作大赛" is intentionally excluded here — it is a
    # multi-track competition covering video, image AND music. Only the
    # music-specific sub-competitions above should trigger exclusion.
    # Music-genre content tags (vocal cover / performance)
    "翻唱",             # voice cover
    "古风翻唱",         # ancient-Chinese-style vocal cover
    "戏腔",             # operatic singing style
    # Rap / hip-hop genre
    "说唱",             # rap / hip-hop
    "中文说唱",         # Chinese rap
    "嘻哈",             # hip-hop
    "hiphop",
    "rap",
    # K-pop / vocal covers
    "kpop",
    "k-pop",
    "cover",            # music cover (vocal/instrumental)
}

# Video generation tool tags — if any of these appear alongside AI music
# tags, the video is an AI video (with AI music as BGM) and should be kept.
_VIDEO_GEN_TOOL_TAGS = {
    "pika", "pika ai", "pikalabs",
    "kling", "可灵", "可灵ai",
    "runway",
    "即梦", "dreamina",
    "hailuo",
    "luma", "dream machine",
    "hunyuan",
    "通义", "万相", "通义万相",
    "wan2",
    "ai视频", "ai动画", "ai生成视频", "视频生成",
}

# Gaming tag exclusion — gamers post game footage claiming it's "AI
# generated" for humor (e.g. "sora2帮我生成一个世一猎艾许视频" which is
# actually APEX gameplay). Bilibili's argue_msg detection false-positives
# on these because the title says "AI"; we catch them via the tags API.
_GAMING_TAG_KEYWORDS = {
    # FPS / shooter
    "fps", "apex", "apex英雄", "csgo", "cs2", "valorant", "pubg", "绝地求生",
    "和平精英", "cod", "使命召唤", "战地", "battlefield", "堡垒之夜", "fortnite",
    "守望先锋", "overwatch", "彩虹六号",
    # MOBA
    "lol", "英雄联盟", "王者荣耀", "dota",
    # Open world / RPG
    "原神", "genshin", "崩坏", "鸣潮", "塞尔达", "zelda", "艾尔登法环",
    "elden ring", "黑神话", "怪物猎人", "monster hunter", "赛博朋克",
    "cyberpunk",
    # Sandbox / other
    "minecraft", "我的世界", "gta", "roblox",
    # Generic gaming terms
    "游戏实况", "游戏录屏", "游戏视频", "gameplay", "实机演示",
}

# Deepfake / face-swap tag exclusion — videos using DeepFaceLab, Roop,
# deepfacelive, or similar face-substitution pipelines on existing footage.
# Face-swap is categorically different from text-to-video generation and
# should not be in the fake class (same reasoning as the 0010 migration).
# Safety guard: only exclude if no AI video-gen tool tag is also present,
# so "AI-generated scene where a face is later swapped" is not caught.
_DEEPFAKE_TAG_KEYWORDS = {
    "换脸",           # face swap (generic)
    "ai换脸",         # AI face swap
    "换头",           # head swap
    "人脸替换",       # face replacement
    "deepfacelab",
    "deepfacelive",
    "roop",
    "faceswap",
    "face swap",
    "facefusion",
    "insightface",
}

# Title terms that, combined with deepfake tags, confirm face-swap content.
# Used in _is_deepfake_title() — even without tag confirmation, titles that
# explicitly name face-swap tools are unambiguous.
_DEEPFAKE_TITLE_TERMS = [
    "换脸",
    "换头",
    "人脸替换",
    "deepfacelab",
    "deepfacelive",
    "roop换脸",
    "facefusion",
]

_EM_TAG_RE = re.compile(r"</?em[^>]*>")


def _parse_duration(dur) -> float | None:
    """Accept both search-API 'H:MM:SS' strings and newlist-API int seconds."""
    if dur is None:
        return None
    if isinstance(dur, (int, float)):
        return float(dur) if dur > 0 else None
    if not isinstance(dur, str):
        return None
    parts = dur.split(":")
    try:
        parts_int = [int(p) for p in parts]
    except ValueError:
        return None
    if len(parts_int) == 2:
        m, s = parts_int
        return float(m * 60 + s)
    if len(parts_int) == 3:
        h, m, s = parts_int
        return float(h * 3600 + m * 60 + s)
    return None


def _clean_title(raw: str) -> str:
    """Strip Bilibili search `<em class="keyword">...</em>` hit-markers."""
    return _html.unescape(_EM_TAG_RE.sub("", raw or ""))


def _has_excluded_term(text: str) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in _EXCLUDE_TITLE_TERMS)


def _is_ai_music_title(text: str) -> bool:
    """True if the title pattern unambiguously marks this as AI music, not AI video."""
    lower = text.lower()
    return any(term in lower for term in _AI_MUSIC_TITLE_TERMS)


def _is_ai_music_tags_only(tags: list[str]) -> bool:
    """True if tags signal AI music AND no video-generation tool is present.

    Keeps "AI video with Suno background music" (has video-gen tool tag)
    while dropping "pure Suno song cover" (no video-gen tool tag).

    Uses substring matching (not exact) so tags like
    "AI音乐征集大赛·2025第三期" still match keyword "AI音乐征集大赛".
    """
    lowered = [t.lower() for t in tags]
    has_music = any(kw in tag for tag in lowered for kw in _AI_MUSIC_TAG_KEYWORDS)
    has_video_tool = any(kw in tag for tag in lowered for kw in _VIDEO_GEN_TOOL_TAGS)
    return has_music and not has_video_tool


def _is_deepfake_title(text: str) -> bool:
    """True if the title explicitly names a face-swap tool / technique."""
    lower = text.lower()
    return any(term in lower for term in _DEEPFAKE_TITLE_TERMS)


def _is_vtuber_title(text: str) -> bool:
    """True if the title or tags text contains a VTuber character / industry term."""
    lower = (text or "").lower()
    return any(term in lower for term in _VTUBER_TITLE_TERMS)


def _is_vtuber_tags(tags: list[str]) -> bool:
    """True if tags contain any VTuber industry marker.

    No video-gen-tool safety guard here (unlike music/deepfake): VTuber
    streams with AI edits are still VTuber content and don't belong in a
    generative-video benchmark.
    """
    lowered = [t.lower() for t in tags]
    return any(kw in tag for tag in lowered for kw in _VTUBER_TAG_KEYWORDS)


def _is_tutorial_tags(tags: list[str]) -> bool:
    """True if tags flag this as tutorial / teaching / workflow-sharing
    content. Title-based _has_excluded_term() catches the obvious case;
    this covers the "title is content-flavored but tags are 教程" case."""
    lowered = [t.lower() for t in tags]
    return any(kw in tag for tag in lowered for kw in _TUTORIAL_TAG_KEYWORDS)


def _is_ai_anchor_title(text: str) -> bool:
    """True if the title explicitly names an AI-anchor / digital-human /
    livestream-sales format."""
    lower = (text or "").lower()
    return any(term in lower for term in _AI_ANCHOR_TITLE_TERMS)


def _is_ai_anchor_tags(tags: list[str]) -> bool:
    """True if tags flag this as AI-anchor / digital-human-host content.
    No video-gen-tool safety guard: same reasoning as VTuber — avatar-based
    presenter content is categorically different from generative AI video."""
    lowered = [t.lower() for t in tags]
    return any(kw in tag for tag in lowered for kw in _AI_ANCHOR_TAG_KEYWORDS)


def _is_deepfake_tags_only(tags: list[str]) -> bool:
    """True if tags contain face-swap keywords AND no AI video-gen tool is present.

    Keeps AI-generated scenes with incidental face work; drops pure DeepFaceLab /
    Roop / face-swap-on-existing-footage content.

    Uses substring matching so "deepfacelab v3" still matches "deepfacelab".
    """
    lowered = [t.lower() for t in tags]
    has_deepfake = any(kw in tag for tag in lowered for kw in _DEEPFAKE_TAG_KEYWORDS)
    has_video_tool = any(kw in tag for tag in lowered for kw in _VIDEO_GEN_TOOL_TAGS)
    return has_deepfake and not has_video_tool


# UA rotation pool. Bilibili's anti-bot edge tracks rate limits per
# (UA, IP) tuple — once a UA gets flagged and starts returning
# v_voucher, the only way to recover without waiting 30+ minutes is
# to switch to a different UA. Fresh UA = fresh rate budget.
_UA_POOL = [
    # Desktop Chrome variants
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
     "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
     "https://www.bilibili.com/"),
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
     "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
     "https://www.bilibili.com/"),
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2_1) AppleWebKit/605.1.15 "
     "(KHTML, like Gecko) Version/17.2 Safari/605.1.15",
     "https://www.bilibili.com/"),
    # Mobile Safari — empirically bypasses Bilibili's desktop UA flag
    ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
     "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
     "Mobile/15E148 Safari/604.1",
     "https://m.bilibili.com/"),
    # Firefox
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) "
     "Gecko/20100101 Firefox/122.0",
     "https://www.bilibili.com/"),
    # Edge
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
     "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
     "https://www.bilibili.com/"),
]

_ua_index = [0]

def _get_next_ua() -> tuple[str, str]:
    """Round-robin advance through the UA pool. Returns (UA, Referer)."""
    idx = _ua_index[0] % len(_UA_POOL)
    _ua_index[0] += 1
    return _UA_POOL[idx]


_HEADERS = {
    # Default UA — used only if no rotation happened. Kept for
    # backward-compat with tests that import _HEADERS directly.
    "User-Agent": _UA_POOL[0][0],
    "Referer": _UA_POOL[0][1],
}

# The unsigned /x/web-interface/search/type endpoint is now IP-banned
# on anything resembling a scraper pattern — Bilibili migrated the
# official web client to the WBI-signed /wbi/search/type variant in
# 2024. Every request to the WBI endpoint must include a `wts`
# timestamp and a `w_rid` md5 signature computed from a mixin key that
# we extract from the wbi_img URLs returned by /x/web-interface/nav.
_SEARCH_URL = "https://api.bilibili.com/x/web-interface/wbi/search/type"
_NAV_URL = "https://api.bilibili.com/x/web-interface/nav"
_NEWLIST_URL = "https://api.bilibili.com/x/web-interface/newlist"
_VIEW_URL = "https://api.bilibili.com/x/web-interface/view"
_TAGS_URL = "https://api.bilibili.com/x/tag/archive/tags"
_HOMEPAGE_URL = "https://www.bilibili.com/"

# Bilibili surfaces two AI-disclosure strings in argue_info.argue_msg:
#   1. Platform auto-detection: "该内容疑似使用AI技术合成，请谨慎甄别"
#      — fired by Bilibili's own detector; less reliable, may miss subtler AI.
#   2. Author self-declaration: "作者声明：该视频使用人工智能合成技术"
#      — creator opted in; stronger signal (explicit disclosure, not inference).
# Accept either — both count as AI-generated for our purposes.
_AI_ARGUE_MSGS = {
    "该内容疑似使用AI技术合成，请谨慎甄别",   # platform auto-detected
    "作者声明：该视频使用人工智能合成技术",     # author self-declared
}

# Fixed permutation used to mix img_key and sub_key into the signing
# material. This is constant in Bilibili's web client; lifted from the
# minified source of their search page.
_WBI_MIXIN_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
    27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
    22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
]


class _WbiSigner:
    """Compute the `wts` + `w_rid` signature pair for WBI-protected endpoints.

    Caches the mixin key derived from `/x/web-interface/nav` and
    refreshes it after 30 minutes (Bilibili rotates keys daily; 30 min
    is well inside the safe window and avoids the round trip on every
    request). Thread-safe: signing is stateless once the key is loaded,
    and the refresh path is guarded by a lock.
    """

    def __init__(self, session: requests.Session) -> None:
        self._session = session
        self._mixin_key: str | None = None
        self._refreshed_at: float = 0.0
        import threading
        self._lock = threading.Lock()

    def _refresh_locked(self) -> None:
        try:
            resp = self._session.get(_NAV_URL, timeout=10)
            data = resp.json().get("data") or {}
            wbi = data.get("wbi_img") or {}
            img_url = wbi.get("img_url") or ""
            sub_url = wbi.get("sub_url") or ""
            img_key = img_url.rsplit("/", 1)[-1].split(".")[0]
            sub_key = sub_url.rsplit("/", 1)[-1].split(".")[0]
            if not img_key or not sub_key:
                raise ValueError("missing wbi img/sub keys")
            raw = img_key + sub_key
            self._mixin_key = "".join(raw[i] for i in _WBI_MIXIN_TAB)[:32]
            self._refreshed_at = time.time()
            logger.debug(
                "BilibiliCrawler: refreshed wbi mixin key (len=%d)",
                len(self._mixin_key or ""),
            )
        except (requests.RequestException, ValueError, KeyError) as exc:
            logger.warning("BilibiliCrawler: wbi nav refresh failed: %s", exc)

    def sign(self, params: dict) -> dict:
        import hashlib
        import urllib.parse

        with self._lock:
            if self._mixin_key is None or (time.time() - self._refreshed_at) > 1800:
                self._refresh_locked()
            mixin = self._mixin_key

        signed = dict(params)
        signed["wts"] = int(time.time())
        if not mixin:
            # Best-effort fallback: still return wts so callers don't
            # crash, but the request will get rejected. The outer
            # retry/rebootstrap loop will re-trigger a refresh.
            return signed
        sorted_items = sorted(signed.items())
        query = "&".join(
            f"{k}={urllib.parse.quote(str(v), safe='')}" for k, v in sorted_items
        )
        signed["w_rid"] = hashlib.md5((query + mixin).encode()).hexdigest()
        return signed


def _check_platform_ai_label(session: requests.Session, bvid: str) -> bool:
    """Return True iff argue_info.argue_msg matches any AI-disclosure string.

    Accepts both platform auto-detection ("疑似") and author self-declaration
    ("作者声明") — the latter is the stronger signal.
    """
    resp = _gentle_get(session, _VIEW_URL, {"bvid": bvid}, timeout=8, tries=2)
    if resp is None or resp.status_code != 200:
        return False
    try:
        data = resp.json().get("data", {})
        argue_info = data.get("argue_info") or {}
        msg = argue_info.get("argue_msg", "") or ""
        return any(s in msg for s in _AI_ARGUE_MSGS)
    except (ValueError, KeyError):
        return False


def _fetch_tag_list(session: requests.Session, bvid: str) -> list[str]:
    """Return the video's tag name list via the public tag endpoint.

    Used for the gaming false-positive filter AND to populate
    content_tags when the candidate comes from the newlist endpoint
    (which doesn't inline tags). Returns [] on any error — fail open so
    a flaky endpoint can't block a candidate, the argue_msg gate
    already ran.
    """
    resp = _gentle_get(session, _TAGS_URL, {"bvid": bvid}, timeout=8, tries=2)
    if resp is None or resp.status_code != 200:
        return []
    try:
        arr = resp.json().get("data", []) or []
        return [(t.get("tag_name") or "").strip() for t in arr if t.get("tag_name")]
    except (ValueError, KeyError):
        return []


def _platform_gate(
    session: requests.Session, bvid: str,
    inline_tags: list[str] | None = None,
    require_non_ai: bool = False,
    historical_mode: bool = False,
) -> tuple[bool, list[str]]:
    """Combined argue_msg + gaming-tag gate, returns (accept, tags).

    Modes:
      - default: accept videos WITH argue_msg AI label (fake side)
      - require_non_ai=True: accept videos WITHOUT argue_msg (still
        runs the label check per-video)
      - historical_mode=True: skip argue_msg entirely — callers should
        scope the search to pre-AI-era pubdates (e.g. before 2023) so
        absence is guaranteed by time, not by a live API check. Cuts
        per-candidate HTTP cost in half, which is what lets a real-video
        crawl stay under Bilibili's rate limits.

    Gaming tag filter applies in all three modes.
    """
    if historical_mode:
        # Skip argue_msg check — assumed non-AI by temporal priority.
        pass
    else:
        has_ai_label = _check_platform_ai_label(session, bvid)
        if require_non_ai:
            if has_ai_label:
                return (False, [])
        else:
            if not has_ai_label:
                return (False, [])
    if inline_tags is not None:
        tags = inline_tags
    else:
        tags = _fetch_tag_list(session, bvid)
    lowered = {t.lower() for t in tags}
    if lowered & _GAMING_TAG_KEYWORDS:
        return (False, tags)
    return (True, tags)


def _normalize_search_item(raw: dict) -> dict:
    """Shape search-API item into the uniform downstream dict."""
    return {
        "bvid": raw.get("bvid") or "",
        "title": _clean_title(raw.get("title", "") or ""),
        "description": raw.get("description", "") or "",
        "tag_str": raw.get("tag", "") or "",
        "duration_sec": _parse_duration(raw.get("duration")),
        "pic": raw.get("pic", "") or "",
        "pubdate": raw.get("pubdate") or 0,
        "raw_metadata": {
            "aid": raw.get("aid"),
            "mid": raw.get("mid"),
            "author": raw.get("author"),
            "typename": raw.get("typename"),
            "play": raw.get("play"),
            "like": raw.get("like"),
            "source": "search",
        },
    }


def _normalize_newlist_item(raw: dict) -> dict:
    """Shape newlist-API item into the uniform downstream dict."""
    stat = raw.get("stat") or {}
    owner = raw.get("owner") or {}
    return {
        "bvid": raw.get("bvid") or "",
        "title": raw.get("title", "") or "",
        "description": raw.get("desc", "") or "",
        "tag_str": "",  # newlist doesn't inline tags
        "duration_sec": _parse_duration(raw.get("duration")),
        "pic": raw.get("pic", "") or "",
        "pubdate": raw.get("pubdate") or 0,
        "raw_metadata": {
            "aid": raw.get("aid"),
            "mid": owner.get("mid"),
            "author": owner.get("name"),
            "typename": raw.get("tname"),
            "play": stat.get("view"),
            "like": stat.get("like"),
            "source": f"newlist:rid={raw.get('tid')}",
        },
    }


def _bootstrap_session() -> requests.Session:
    """Create a session pre-seeded with Bilibili's anti-bot cookies.

    Picks the next UA from `_UA_POOL` (round-robin) and fetches the
    matching homepage to seed buvid3/b_nut. Rotating UA per session is
    what lets us come back after a v_voucher flag — Bilibili tracks
    rate per (UA, IP) tuple.
    """
    import uuid

    session = requests.Session()
    ua, referer = _get_next_ua()
    session.headers.update({"User-Agent": ua, "Referer": referer})
    homepage = referer or _HOMEPAGE_URL

    try:
        session.get(homepage, timeout=10)
    except requests.RequestException as exc:
        logger.info("BilibiliCrawler: homepage bootstrap failed: %s", exc)

    if "buvid3" not in session.cookies.get_dict():
        synthetic3 = f"{str(uuid.uuid4()).upper()}infoc"
        session.cookies.set(
            "buvid3", synthetic3, domain=".bilibili.com", path="/",
        )
        session.cookies.set(
            "b_nut", str(int(time.time())),
            domain=".bilibili.com", path="/",
        )
        logger.info("BilibiliCrawler: using synthetic buvid3 fallback")

    return session


def _get_signer(session: requests.Session) -> _WbiSigner:
    """Attach (or reuse) a `_WbiSigner` on the session.

    Stashed as an attribute so it's shared across all `_search` calls
    on the same session — one nav refresh, many signed requests.
    """
    signer = getattr(session, "_wbi_signer", None)
    if signer is None:
        signer = _WbiSigner(session)
        session._wbi_signer = signer  # type: ignore[attr-defined]
    return signer


def _rebootstrap(session: requests.Session) -> None:
    """Rotate UA + fresh cookies on an existing session.

    Called when the crawler hits 2 consecutive empty page 1s (voucher
    cascade). Rotating to the NEXT UA in the pool is what clears the
    voucher state — simply refreshing cookies under the same UA isn't
    enough because Bilibili's rate tracker is (UA, IP)-keyed.
    """
    session.cookies.clear()
    ua, referer = _get_next_ua()
    session.headers.update({"User-Agent": ua, "Referer": referer})
    logger.info(
        "BilibiliCrawler: rebootstrap rotating UA to %s...",
        ua[:40],
    )
    # Also drop the stale WBI signer so the next request refreshes it
    # under the new UA/cookies (some operators tie signature material
    # to client fingerprint).
    if hasattr(session, "_wbi_signer"):
        delattr(session, "_wbi_signer")
    homepage = referer or _HOMEPAGE_URL
    for attempt in (1, 2):
        try:
            session.get(homepage, timeout=10)
            if "buvid3" in session.cookies.get_dict():
                return
        except requests.RequestException:
            pass
        time.sleep(1 + attempt)


# Global token-bucket rate limiter. Every HTTP call to Bilibili goes
# through _rate_limit_wait() before hitting the wire. Empirically,
# sustained rate above ~60 req/min triggers the v_voucher soft-limit
# even on a cooled IP, so we cap at ~30 req/min (one call per 2.2s).
# Rate limit tunable via BILIBILI_MIN_INTERVAL env var. Local IP needs
# ~1.8s (heavily flagged); Fly's Tokyo IP tolerates ~0.8s. Default to
# the gentle side.
import os as _os
_rate_lock = None
_last_call_at = [0.0]
_MIN_CALL_INTERVAL = float(_os.environ.get("BILIBILI_MIN_INTERVAL", "1.8"))


def _rate_limit_wait() -> None:
    """Block until at least _MIN_CALL_INTERVAL seconds have passed since
    the last HTTP call. Thread-safe (used by the ThreadPoolExecutor gate
    workers + the main crawl loop)."""
    import threading
    global _rate_lock
    if _rate_lock is None:
        _rate_lock = threading.Lock()
    with _rate_lock:
        now = time.time()
        elapsed = now - _last_call_at[0]
        if elapsed < _MIN_CALL_INTERVAL:
            time.sleep(_MIN_CALL_INTERVAL - elapsed)
        _last_call_at[0] = time.time()


def _gentle_get(
    session: requests.Session, url: str, params: dict, timeout: float = 15,
    tries: int = 3,
) -> requests.Response | None:
    """GET with rate limiting + 412-aware retry + exponential backoff.

    Every request goes through the global token bucket first so the
    crawler can't burst past Bilibili's per-minute ceiling. On 412 or
    voucher we rebootstrap cookies and retry; on other transient errors
    we back off. Returns None if all retries fail — callers treat that
    as "skip this page" rather than crashing the crawl.
    """
    import random
    for attempt in range(tries):
        _rate_limit_wait()
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code == 412:
                backoff = (2 ** attempt) * 4 + random.random() * 2
                logger.warning(
                    "BilibiliCrawler: 412 from %s, rebootstrap+sleep %.1fs",
                    url.split("/")[-1], backoff,
                )
                _rebootstrap(session)
                time.sleep(backoff)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == tries - 1:
                logger.warning(
                    "BilibiliCrawler: %s failed after %d tries: %s",
                    url.split("/")[-1], tries, exc,
                )
                return None
            time.sleep((2 ** attempt) + random.random())
    return None


@register("bilibili")
class BilibiliCrawler(BaseCrawler):
    """Crawl Bilibili for fresh AI videos via search + region newlist."""

    name = "bilibili"
    tier = 2

    def crawl(self, config: dict) -> Iterator[CrawledVideo]:
        queries: list[str] = config.get("search_queries", _DEFAULT_QUERIES)
        # Region browsing defaults OFF — empirical measurement showed 0%
        # argue_msg hit rate on 500+ random recent uploads across 7
        # non-gaming regions. The `_DEFAULT_REGIONS` list is kept as a
        # constant for callers that want to opt-in for experimentation.
        regions: list[int] = config.get("regions", [])
        # Search order modes — Bilibili caps each (query, order) at ~1000
        # results, but different orders return different subsets of the
        # total candidate space (pubdate = newest, click = most-viewed,
        # scores = most-liked). Iterating multiple orders per query
        # multiplies the unique-bvid coverage without needing more
        # distinct keywords.
        order_modes: list[str] = config.get(
            "search_order_modes", ["pubdate"],
        )
        max_videos: int = config.get("max_videos", 50)
        min_duration: float = config.get("min_duration", 3)
        max_duration: float = config.get("max_duration", 60)
        pages_per_query: int = config.get("pages_per_query", 3)
        pages_per_region: int = config.get("pages_per_region", 20)
        page_size: int = config.get("page_size", 50)
        exclude_tutorials: bool = config.get("exclude_tutorials", True)
        exclude_ai_music: bool = config.get("exclude_ai_music", True)
        exclude_deepfake: bool = config.get("exclude_deepfake", True)
        # Concurrency is deliberately low — Bilibili's anti-bot edge
        # starts returning 412 above ~3 in-flight requests from the same
        # client fingerprint. Prefer long-running crawls over burst.
        check_workers: int = config.get("check_workers", 3)
        # Jittered page sleep keeps us under the edge's rate detector.
        page_sleep: float = float(config.get("page_sleep", 2.0))
        # Minimum wall-clock gap between consecutive search page calls.
        # With many bvids pre-seeded, Phase 1 filters out most candidates
        # so argue_msg checks are few and pages complete in ~15-20s —
        # much faster than the ~45s Bilibili needs between search calls on
        # flagged IPs. Set min_search_interval_sec >= 45 on Aliyun IPs to
        # prevent v_voucher regardless of candidate density.
        # Default 0 (no extra sleep, backward compatible).
        min_search_interval: float = float(
            config.get("min_search_interval_sec", 0.0)
        )
        max_age_days: int | None = config.get("max_age_days")
        require_non_ai: bool = bool(config.get("require_non_ai", False))
        historical_mode: bool = bool(config.get("historical_mode", False))
        skip_thumbnails: bool = bool(config.get("skip_thumbnails", False))
        pubtime_end_sec: int | None = config.get("pubtime_end_sec")
        pubtime_begin_sec: int | None = config.get("pubtime_begin_sec")

        min_pubdate: float | None = None
        if max_age_days is not None and max_age_days > 0:
            min_pubdate = time.time() - (max_age_days * 86400)

        # Client-side pubdate UPPER bound for historical-mode crawls.
        # Bilibili's search API accepts `pubtime_end_s` but ignores it
        # when order=scores/click (not fully supported there). Belt-and-
        # suspenders: drop any item whose pubdate crosses the ceiling
        # after the page comes back.
        max_pubdate: float | None = (
            float(pubtime_end_sec) if pubtime_end_sec is not None else None
        )

        # Pre-seed seen_bvids with source_ids the caller already has
        # in the database. This skips known bvids in the early filter
        # (before the expensive argue_msg + tag API calls), so a
        # re-crawl with overlapping queries doesn't burn API budget
        # re-validating rows the DB already knows about.
        preseeded: list[str] = config.get("seen_source_ids") or []
        seen_bvids: set[str] = set(preseeded)
        yielded = 0
        session = _bootstrap_session()

        # --- Phase 1: AI-keyword search across multiple sort orders ---
        # Bilibili's 1000-result cap is per (query, order), so iterating
        # pubdate → click → scores for the same query unlocks three
        # disjoint slices of the same keyword's candidate space.
        #
        # Session-level soft limit: after ~10-15 search calls on the
        # same session, Bilibili starts silently returning empty result
        # arrays (not 412, just code=0 items=[]). On *any* empty
        # response, increment a counter; at 2 consecutive empties we
        # assume the current UA is voucher-flagged and swap UA via
        # _rebootstrap(), then retry the current page on the fresh UA.
        consecutive_empty = 0
        _last_search_at: float = 0.0  # wall time of last search page call
        for query in queries:
            if yielded >= max_videos:
                break
            for order in order_modes:
                if yielded >= max_videos:
                    break
                cutoff_is_monotonic = order == "pubdate"
                for page in range(1, pages_per_query + 1):
                    if yielded >= max_videos:
                        break
                    # Enforce minimum gap between consecutive search calls.
                    # When bvid pre-seeding is heavy, pages complete fast
                    # (few argue_msg checks) and search calls come too
                    # quickly for flagged IPs — triggering v_voucher even
                    # at 1.8s/call. min_search_interval ensures spacing
                    # regardless of candidate density.
                    if min_search_interval > 0:
                        _gap = time.time() - _last_search_at
                        if _gap < min_search_interval:
                            time.sleep(min_search_interval - _gap)
                    _last_search_at = time.time()
                    logger.info(
                        "BilibiliCrawler: search '%s' order=%s page=%d",
                        query, order, page,
                    )
                    raw_items = self._search(
                        session, query, page, page_size, order=order,
                        pubtime_end_sec=pubtime_end_sec,
                        pubtime_begin_sec=pubtime_begin_sec,
                    )
                    if not raw_items:
                        consecutive_empty += 1
                        if consecutive_empty >= 2:
                            logger.warning(
                                "BilibiliCrawler: %d consecutive empty responses, "
                                "rotating UA via rebootstrap",
                                consecutive_empty,
                            )
                            _rebootstrap(session)
                            time.sleep(5)
                            consecutive_empty = 0
                            # Retry current page with fresh UA
                            raw_items = self._search(
                                session, query, page, page_size, order=order,
                                pubtime_end_sec=pubtime_end_sec,
                                pubtime_begin_sec=pubtime_begin_sec,
                            )
                            if not raw_items:
                                # Still empty after rotation — this query/order
                                # is probably genuinely exhausted. Move on.
                                break
                        else:
                            # First empty — might just be end of results.
                            break
                    else:
                        consecutive_empty = 0
                    items = [_normalize_search_item(r) for r in raw_items]
                    if cutoff_is_monotonic:
                        items, stop = self._apply_pubdate_cutoff(items, min_pubdate)
                        if stop:
                            logger.info(
                                "BilibiliCrawler: '%s' order=%s page=%d older than cutoff, stopping",
                                query, order, page,
                            )
                            break
                    elif min_pubdate is not None:
                        items = [
                            it for it in items
                            if float(it.get("pubdate") or 0) >= min_pubdate
                        ]
                    if max_pubdate is not None:
                        items = [
                            it for it in items
                            if float(it.get("pubdate") or 0) <= max_pubdate
                        ]
                    for video in self._process_page(
                        items, seen_bvids, min_duration, max_duration,
                        exclude_tutorials, session, check_workers,
                        require_non_ai=require_non_ai,
                        historical_mode=historical_mode,
                        skip_thumbnails=skip_thumbnails,
                        exclude_ai_music=exclude_ai_music,
                        exclude_deepfake=exclude_deepfake,
                    ):
                        if yielded >= max_videos:
                            break
                        yield video
                        yielded += 1
                    import random as _rnd
                    time.sleep(page_sleep + _rnd.random())

        # --- Phase 2: region newlist ---
        for rid in regions:
            if yielded >= max_videos:
                break
            for page in range(1, pages_per_region + 1):
                if yielded >= max_videos:
                    break
                logger.info(
                    "BilibiliCrawler: region rid=%d page=%d", rid, page,
                )
                raw_items = self._newlist(session, rid, page, page_size)
                if not raw_items:
                    break
                items = [_normalize_newlist_item(r) for r in raw_items]
                items, stop = self._apply_pubdate_cutoff(items, min_pubdate)
                if stop:
                    logger.info(
                        "BilibiliCrawler: rid=%d page=%d older than cutoff, stopping region",
                        rid, page,
                    )
                    break
                for video in self._process_page(
                    items, seen_bvids, min_duration, max_duration,
                    exclude_tutorials, session, check_workers,
                    require_non_ai=require_non_ai,
                    historical_mode=historical_mode,
                    skip_thumbnails=skip_thumbnails,
                    exclude_ai_music=exclude_ai_music,
                    exclude_deepfake=exclude_deepfake,
                ):
                    if yielded >= max_videos:
                        break
                    yield video
                    yielded += 1
                time.sleep(1.2)

        logger.info("BilibiliCrawler: yielded %d videos total", yielded)

    def estimate_available(self, config: dict) -> int:
        queries = config.get("search_queries", _DEFAULT_QUERIES)
        regions = config.get("regions", _DEFAULT_REGIONS)
        page_size = config.get("page_size", 50)
        search_cap = len(queries) * config.get("pages_per_query", 3) * page_size
        region_cap = len(regions) * config.get("pages_per_region", 20) * page_size
        return min(search_cap + region_cap, config.get("max_videos", 50))

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _search(
        session: requests.Session, query: str, page: int, page_size: int,
        order: str = "pubdate",
        pubtime_end_sec: int | None = None,
        pubtime_begin_sec: int | None = None,
    ) -> list[dict]:
        params: dict = {
            "search_type": "video",
            "keyword": query,
            "order": order,
            "page": page,
            "page_size": page_size,
        }
        if pubtime_end_sec is not None:
            params["pubtime_end_s"] = pubtime_end_sec
        if pubtime_begin_sec is not None:
            params["pubtime_begin_s"] = pubtime_begin_sec
        signed = _get_signer(session).sign(params)
        resp = _gentle_get(session, _SEARCH_URL, signed, timeout=15, tries=3)
        if resp is None:
            return []
        try:
            body = resp.json()
        except ValueError:
            return []
        if body.get("code") != 0:
            logger.warning(
                "BilibiliCrawler: search API error for '%s' page=%d: code=%s msg=%s",
                query, page, body.get("code"), body.get("message"),
            )
            return []
        data = body.get("data") or {}
        # Bilibili's anti-bot challenge: when suspicious traffic is
        # detected, the API returns code=0 but with `v_voucher` in place
        # of `result`. Treat it as soft-limit, not a real empty result.
        if data.get("v_voucher") and not data.get("result"):
            logger.warning(
                "BilibiliCrawler: v_voucher challenge on '%s' page=%d — "
                "session flagged, caller should rebootstrap",
                query, page,
            )
            return []
        return data.get("result") or []

    @staticmethod
    def _newlist(
        session: requests.Session, rid: int, page: int, page_size: int,
    ) -> list[dict]:
        params = {"rid": rid, "pn": page, "ps": page_size}
        resp = _gentle_get(session, _NEWLIST_URL, params, timeout=15, tries=3)
        if resp is None:
            return []
        try:
            body = resp.json()
        except ValueError:
            return []
        if body.get("code") != 0:
            logger.warning(
                "BilibiliCrawler: newlist API error for rid=%d page=%d: code=%s msg=%s",
                rid, page, body.get("code"), body.get("message"),
            )
            return []
        return body.get("data", {}).get("archives") or []

    # ------------------------------------------------------------------
    # Page processing
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_pubdate_cutoff(
        items: list[dict], min_pubdate: float | None,
    ) -> tuple[list[dict], bool]:
        """Filter items to those at/after min_pubdate, and report end-of-range.

        Both search (order=pubdate) and newlist return newest-first, so
        once the head of the page is older than the cutoff we know the
        rest of this query/region will be too — caller should stop.
        """
        if min_pubdate is None or not items:
            return items, False
        first_pubdate = float(items[0].get("pubdate") or 0)
        if first_pubdate < min_pubdate:
            return [], True
        survivors = [
            it for it in items
            if float(it.get("pubdate") or 0) >= min_pubdate
        ]
        return survivors, False

    @staticmethod
    def _process_page(
        items: list[dict],
        seen_bvids: set[str],
        min_duration: float,
        max_duration: float,
        exclude_tutorials: bool,
        session: requests.Session,
        check_workers: int,
        require_non_ai: bool = False,
        historical_mode: bool = False,
        skip_thumbnails: bool = False,
        exclude_ai_music: bool = True,
        exclude_deepfake: bool = True,
    ) -> list[CrawledVideo]:
        """Run early-reject → concurrent gate → CrawledVideo build for a page."""
        # Phase 1: cheap in-memory filters (title, duration, seen dedup).
        survivors: list[dict] = []
        for it in items:
            bvid = it.get("bvid") or ""
            if not bvid or bvid in seen_bvids:
                continue
            seen_bvids.add(bvid)
            if exclude_tutorials and _has_excluded_term(it.get("title", "")):
                continue
            if exclude_ai_music and _is_ai_music_title(it.get("title", "")):
                continue
            if exclude_deepfake and _is_deepfake_title(it.get("title", "")):
                continue
            if _is_vtuber_title(it.get("title", "")):
                continue
            if _is_ai_anchor_title(it.get("title", "")):
                continue
            dur = it.get("duration_sec")
            if dur is None or dur < min_duration or dur > max_duration:
                continue
            survivors.append(it)

        if not survivors:
            return []

        # Phase 2: concurrent platform gate (argue_msg + gaming tag).
        # When inline tag_str is present (search path) we pass it to
        # the gate and skip the per-video tag API call — a 2× HTTP
        # reduction per candidate, which matters a lot for rate limits.
        def _gate(item: dict) -> dict | None:
            inline_tag_str = item.get("tag_str") or ""
            if inline_tag_str:
                inline_tags = [
                    t.strip() for t in inline_tag_str.split(",") if t.strip()
                ]
                ok, tags = _platform_gate(
                    session, item["bvid"], inline_tags=inline_tags,
                    require_non_ai=require_non_ai,
                    historical_mode=historical_mode,
                )
            else:
                ok, tags = _platform_gate(
                    session, item["bvid"],
                    require_non_ai=require_non_ai,
                    historical_mode=historical_mode,
                )
            if not ok:
                return None
            item["resolved_tags"] = tags
            return item

        with ThreadPoolExecutor(max_workers=check_workers) as ex:
            accepted = [r for r in ex.map(_gate, survivors) if r is not None]

        if not accepted:
            return []

        # Phase 2.5: AI music tag filter — drop rows whose resolved tags
        # indicate AI audio generation with no video-gen tool present.
        # E.g. Suno covers, AI翻唱 competitions. Runs after the gate so
        # resolved_tags is available. Kept separate from gaming to avoid
        # false-positives on "AI video with Suno background music" (those
        # have video-gen tool tags alongside the Suno tag).
        if exclude_ai_music:
            accepted = [
                it for it in accepted
                if not _is_ai_music_tags_only(it.get("resolved_tags", []))
            ]

        # Phase 2.5b: deepfake / face-swap tag filter — drop rows whose
        # resolved tags indicate face-swap pipelines (换脸, deepfacelab,
        # roop, etc.) with no AI video-gen tool tag. Same safety guard as
        # the music filter.
        if exclude_deepfake:
            accepted = [
                it for it in accepted
                if not _is_deepfake_tags_only(it.get("resolved_tags", []))
            ]

        # Phase 2.5c: VTuber tag filter — drop rows tagged with 虚拟主播 /
        # VUP / hololive / A-SOUL etc. No video-gen-tool safety guard here:
        # VTuber streams with AI edits are still VTuber content.
        accepted = [
            it for it in accepted
            if not _is_vtuber_tags(it.get("resolved_tags", []))
        ]

        # Phase 2.5d: tutorial tag filter — drop rows whose creator tagged
        # them with 教程 / sd学习 / ai教学 / comfyui教程 etc., regardless
        # of whether the title sounds tutorial-y.
        if exclude_tutorials:
            accepted = [
                it for it in accepted
                if not _is_tutorial_tags(it.get("resolved_tags", []))
            ]

        # Phase 2.5e: AI-anchor / digital-human-host tag filter — drop rows
        # tagged AI主播 / AI数字人 / AI带货 / 数字分身 etc. Same domain as
        # VTuber: avatar-based presenter content, not generative video.
        accepted = [
            it for it in accepted
            if not _is_ai_anchor_tags(it.get("resolved_tags", []))
        ]

        if not accepted:
            return []

        # Phase 3: build CrawledVideo from accepted items (synchronous —
        # fetches thumbnail bytes, which also needs the right Referer).
        return [
            BilibiliCrawler._build_video(
                it,
                require_non_ai=require_non_ai or historical_mode,
                skip_thumbnails=skip_thumbnails,
            )
            for it in accepted
        ]

    @staticmethod
    def _build_video(
        item: dict,
        require_non_ai: bool = False,
        skip_thumbnails: bool = False,
    ) -> CrawledVideo:
        bvid = item["bvid"]
        title = item.get("title", "")
        description = item.get("description", "")
        tag_str = item.get("tag_str", "")
        resolved_tags = item.get("resolved_tags") or []
        combined_text = f"{title} {description} {tag_str} {' '.join(resolved_tags)}"
        claimed_generator = _extract_generator(combined_text)

        # Thumbnail: Bilibili returns protocol-relative URLs (`//i0.hdslb...`).
        # The CDN hotlink-blocks (403) any request whose Referer isn't
        # bilibili.com, so we fetch bytes locally with the right Referer
        # and ship them inline — the cloud writes them to
        # /app/data/thumbnails/ and serves them from the same-origin
        # static endpoint.
        raw_pic = item.get("pic", "") or ""
        if raw_pic.startswith("//"):
            thumbnail_url = f"https:{raw_pic}"
        elif raw_pic.startswith("http"):
            thumbnail_url = raw_pic
        else:
            thumbnail_url = None
        thumbnail_bytes = None
        if thumbnail_url and not skip_thumbnails:
            thumbnail_bytes = fetch_thumbnail_bytes(
                thumbnail_url, referer="https://www.bilibili.com/",
            )

        # pubdate is a unix timestamp; convert to ISO 8601 with timezone
        # so the runner can parse it into a DateTime column.
        pubdate = item.get("pubdate") or 0
        published_at = None
        if pubdate:
            try:
                import datetime as _dt
                published_at = _dt.datetime.fromtimestamp(
                    float(pubdate), tz=_dt.timezone.utc
                ).isoformat()
            except (ValueError, TypeError, OSError):
                pass

        # Prefer inline search-API tag_str when present, else fall back
        # to the resolved tag list from the per-video tags endpoint.
        if tag_str:
            content_tags = [t.strip() for t in tag_str.split(",") if t.strip()]
        else:
            content_tags = resolved_tags

        source_url = f"https://www.bilibili.com/video/{bvid}"

        return CrawledVideo(
            source_platform="bilibili",
            source_url=source_url,
            source_id=bvid,
            label="real" if require_non_ai else "fake",
            label_source=(
                "tier1_platform_absence" if require_non_ai
                else "tier2_platform_tag"
            ),
            title=title,
            claimed_generator=None if require_non_ai else claimed_generator,
            content_tags=content_tags,
            published_at=published_at,
            raw_metadata=item.get("raw_metadata", {}),
            download_url=source_url,
            thumbnail_url=thumbnail_url,
            thumbnail_bytes=thumbnail_bytes,
            duration_sec=item.get("duration_sec"),
            resolution_w=None,
            resolution_h=None,
            fps=None,
        )
