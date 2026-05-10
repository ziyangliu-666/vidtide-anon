"""Mark VTuber / tutorial videos as status='excluded'.

Runs against the prod SQLite DB (default /app/data/vidtide.db on Fly,
or pass --db for local use). Soft delete only — sets status='excluded' so
the rows can be restored by flipping status back.

Usage:
    python3 scripts/filter_vtuber_tutorial.py              # dry run, print counts + sample
    python3 scripts/filter_vtuber_tutorial.py --apply      # actually update
    python3 scripts/filter_vtuber_tutorial.py --dump out.csv  # write full match list
"""
import argparse
import csv
import json
import sqlite3
import sys

VT_TAG_TERMS = [
    "vtuber", "vtube", "vup",
    "虚拟主播", "虚拟偶像", "虚拟up主", "虚拟youtuber", "虚拟yt",
    "管人痴", "皮套", "中之人",
    "hololive", "a-soul", "asoul", "にじさんじ", "彩虹社",
    "neuro-sama", "vedal",
]

VT_NAME_TERMS = [
    "东雪莲", "東雪蓮",
    "阿梓", "嘉然", "乃琳", "贝拉", "珈乐", "向晚",
    "嘉心糖", "乃贝", "鹿鸣", "叶瞬光",
    "neuro", "夏色まつり", "神楽めあ",
    "月ノ美兎", "月之美兔",
]

TUT_TERMS = [
    # Chinese — strong
    "教程", "教学", "入门", "保姆级", "手把手",
    "教会你", "一分钟教会", "秒教会",
    "使用方法", "使用指南", "ai教程",
    "工作流", "comfyui", "comfy ui",
    # Tool-pack / deployment distribution — creator-packaged software bundles,
    # not generative video content.
    "整合包", "一键包", "一键整合", "本地部署",
    "免安装", "附安装包",
    # English
    "tutorial", "step by step", "step-by-step",
    "how to use", "how to make", "how to create",
    "beginner", "walkthrough", "guide",
]

# Tag-level tutorial markers. Tags are typically short and creator-chosen,
# so even single-word matches like "教程" or "教学" are reliable signals.
# Bilibili creators aggressively tag their tutorial content (e.g.
# "sd教学", "ai学习", "ai跟我学", "comfyui教程") so tag match catches
# videos whose title is ambiguous ("学做AI动画") but tags are unambiguous.
TUT_TAG_TERMS = [
    "教程", "教学", "入门教程", "入门教学",
    "保姆级", "手把手", "教会你",
    "跟我学", "学习ai", "ai学习", "学ai",
    "ai教程", "ai教学",
    "sd教程", "sd教学", "sd学习",
    "comfyui教程", "comfyui教学", "comfyui入门",
    "工作流教程", "提示词教学",
    "tutorial", "beginner", "walkthrough",
]

# Negative patterns — title contains one of these → NOT a tutorial
# ("从零开始的异世界生活" is the Re:Zero anime name; "guide" appears in misc content)
TUT_NEGATIVE = [
    "从零开始的异世界",  # Re:Zero
]

# AI-music-only title markers. These produce audio (song / arrangement), not
# AI video — should be excluded from a video-forensics benchmark. The crawler
# has an `_AI_MUSIC_TITLE_TERMS` list but this one-shot filter needs its own
# copy since it runs against rows already in the DB.
AI_MUSIC_TITLE_TERMS = [
    "[suno", "【suno",
    "ai翻唱", "ai歌手", "ai配乐", "ai cover", "ai歌曲", "ai音乐生成",
    "ai作曲", "ai编曲", "ai作词", "ai写歌",
    "ai原创歌曲", "ai原创音乐",
]

# AI-anchor / AI-digital-human-as-host / AI-livestream-sales content.
# Same reasoning as VTuber: these are avatar-based presenter/streamer
# content, not generative AI video art. Tool demos for face-swap /
# lip-sync / talking-head software land in the same bucket.
AI_ANCHOR_TITLE_TERMS = [
    "ai主播", "ai女主播", "ai男主播",
    "ai数字人", "ai虚拟主播",
    "数字人主播", "数字主播", "ai数字分身",
    "ai带货", "ai播报", "ai播音",
    "ai新闻主播", "ai直播间",
]

AI_ANCHOR_TAG_TERMS = [
    "ai主播", "ai女主播", "ai男主播",
    "ai数字人",
    "数字人主播", "数字主播", "数字分身",
    "ai带货", "ai播报", "ai播音",
    "ai新闻主播", "ai虚拟主播",
]


AI_VIDEO_NEGATIVE = [
    "ai电影", "ai短片", "ai动画", "ai视频", "ai剧集",
    "ai仿真人", "ai真人", "ai短剧", "ai长片",
    "ai mv",  # music video — has both music and video; keep
]


def row_ai_music_reason(title: str) -> str | None:
    tl = (title or "").lower()
    for neg in AI_VIDEO_NEGATIVE:
        if neg in tl:
            return None
    for term in AI_MUSIC_TITLE_TERMS:
        if term in tl:
            return f"ai_music:{term}"
    return None


def row_ai_anchor_reason(title: str, tags_json: str) -> str | None:
    tl = (title or "").lower()
    for term in AI_ANCHOR_TITLE_TERMS:
        if term in tl:
            return f"ai_anchor:{term}"
    tgl = _decode_tags(tags_json)
    for term in AI_ANCHOR_TAG_TERMS:
        if term in tgl:
            return f"ai_anchor_tag:{term}"
    return None


def _decode_tags(tags_json: str) -> str:
    """content_tags is stored as JSON with \\u-escaped CJK. Decode to a
    lowercased, tab-joined flat string so substring match works on both
    Chinese and English tag content."""
    if not tags_json:
        return ""
    try:
        parsed = json.loads(tags_json)
    except Exception:
        return tags_json.lower()
    if not isinstance(parsed, list):
        return str(parsed).lower()
    return "\t".join(str(t) for t in parsed).lower()


def row_vt_reason(title: str, tags_json: str) -> str | None:
    tl = (title or "").lower()
    tgl = _decode_tags(tags_json)
    for t in VT_TAG_TERMS:
        tlow = t.lower()
        if tlow in tgl or tlow in tl:
            return f"vt_tag:{t}"
    for n in VT_NAME_TERMS:
        nlow = n.lower()
        if nlow in tgl or nlow in tl:
            return f"vt_name:{n}"
    return None


def row_tut_reason(title: str, tags_json: str) -> str | None:
    t = title or ""
    tl = t.lower()
    for neg in TUT_NEGATIVE:
        if neg in t:
            return None
    for term in TUT_TERMS:
        if term in tl:
            return f"tut:{term}"
    tgl = _decode_tags(tags_json)
    for term in TUT_TAG_TERMS:
        if term.lower() in tgl:
            return f"tut_tag:{term}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/app/data/vidtide.db")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dump", metavar="CSV")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, title, content_tags, source_platform, source_url, status "
        "FROM videos"
    ).fetchall()

    matches = []  # (id, reason, title, platform)
    for r in rows:
        if r["status"] == "excluded":
            continue
        reason = row_vt_reason(r["title"] or "", r["content_tags"] or "")
        if reason is None:
            reason = row_tut_reason(r["title"] or "", r["content_tags"] or "")
        if reason is None:
            reason = row_ai_music_reason(r["title"] or "")
        if reason is None:
            reason = row_ai_anchor_reason(r["title"] or "", r["content_tags"] or "")
        if reason is None:
            continue
        matches.append((r["id"], reason, r["title"], r["source_platform"], r["source_url"]))

    by_cat = {"vt": 0, "tut": 0, "music": 0, "anchor": 0}
    by_plat = {}
    for _, reason, _, plat, _ in matches:
        if reason.startswith("vt_"):
            cat = "vt"
        elif reason.startswith("ai_music"):
            cat = "music"
        elif reason.startswith("ai_anchor"):
            cat = "anchor"
        else:
            cat = "tut"
        by_cat[cat] += 1
        by_plat[plat] = by_plat.get(plat, 0) + 1

    print(f"Total rows: {len(rows)}")
    print(
        f"Matched: {len(matches)} "
        f"(vtuber={by_cat['vt']}, tutorial={by_cat['tut']}, "
        f"ai_music={by_cat['music']}, ai_anchor={by_cat['anchor']})"
    )
    print("By platform:")
    for p, c in sorted(by_plat.items(), key=lambda x: -x[1]):
        print(f"  {p}: {c}")

    # sample
    import random
    random.seed(0)
    print("\n--- Sample 20 matches ---")
    for m in random.sample(matches, min(20, len(matches))):
        mid, reason, title, plat, _ = m
        print(f"  [{reason:>20}] {plat} | {(title or '')[:90]}")

    if args.dump:
        with open(args.dump, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "reason", "title", "platform", "url"])
            w.writerows(matches)
        print(f"\nDumped {len(matches)} rows → {args.dump}")

    if args.apply:
        ids = [m[0] for m in matches]
        cur = con.cursor()
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            q = f"UPDATE videos SET status='excluded' WHERE id IN ({','.join('?' * len(chunk))})"
            cur.execute(q, chunk)
        con.commit()
        print(f"\n[APPLIED] Set status='excluded' on {len(ids)} rows.")
    else:
        print("\n(dry run — pass --apply to update status='excluded')")

    return 0


if __name__ == "__main__":
    sys.exit(main())
