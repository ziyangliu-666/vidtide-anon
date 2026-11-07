#!/usr/bin/env python3
"""Metadata-only monthly slice crawler (no video bytes).

For each (month window x query x order x page) it runs a pubtime-windowed
Bilibili search, applies the same cheap title/duration/tag filters as the M0
crawler, and resolves each surviving candidate with ONE /view call, which
gives argue_info (label), pubdate, duration, dimension and state. Every
resolved candidate is appended to --out (resumable); nothing is downloaded.

Labels (same evidence rules as M0 Tier 2):
  fake  <- argue_msg contains a Bilibili AI-disclosure string
  real  <- no argue_msg AND query_kind == generic AND title has no AI keyword
  other <- no argue_msg on an AI-keyword query (kept, never used as real)

Usage:
  python scripts/crawl_meta_slices.py --kind ai      --out data/meta/ai.jsonl
  python scripts/crawl_meta_slices.py --kind generic --out data/meta/generic.jsonl
  python scripts/crawl_meta_slices.py --liveness manifests/M0/metadata.jsonl --n 1000 --out data/meta/live.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import requests  # noqa: E402

from server.crawler import bilibili as B  # noqa: E402

log = logging.getLogger("meta")
CST = timezone(timedelta(hours=8))

MONTHS = [(2026, m) for m in range(5, 12)]  # May .. Nov 2026

AI_QUERIES = ['AI小品', 'AI文生视频', 'AI让老照片动起来', 'AI老照片修复', 'AI远古', 'AI末日', 'AI玄幻', 'AI图生视频', 'AI科普', 'AI蒸汽朋克', 'AI河南话', 'AI修真', 'AI复原', 'AI换脸', 'AI影像大赛', 'AI品牌TVC', 'AI创意广告', 'AI片头', 'AI纪录片', 'AI史前', 'AI深海', 'AI戏曲', 'AI京剧', 'AI封神', 'AI宋朝', 'AI民国', '老上海AI', '民国AI', 'AI国风', 'AI ACG', 'AI粤语', 'AI偷感', 'AI退退退', 'AI显眼包', 'AI脑洞', 'AI脑洞大开', '咕咕嘎嘎AI', 'AI葫芦娃', 'AI孙悟空', 'AI机甲少女', 'AI整活', 'AI土狗', 'AI复活偶像', 'AI东北话', 'AI恶搞', 'AI换脸搞笑', '科目三AI', 'AI科目三', '挖呀挖AI', 'AI土味', 'AI抽象', 'AI鬼畜', 'AI老照片动起来', 'AI水墨动画', '赛博朋克AI', 'AI科幻', 'AI机甲', 'AI古风', 'AI古装', 'AI汉服', 'AI水墨', 'AI武侠', 'AI仙侠', 'AI玄幻', 'AI美女', 'AI美少女', 'AI美妆', 'AI时尚', 'AI动物', 'AI猫咪', 'AI熊猫', 'AI恐龙', 'AI儿童', 'AI萌娃', 'AI宇宙', 'AI太空', 'AI星球', 'AI星空', 'AI梦境', 'AI幻想', 'AI童话', 'AI海底', 'AI丛林', 'AI火山', 'AI沙漠', 'AI僵尸', 'AI怪物', 'AI机器人', 'AI短剧', 'AI剧情', 'AI故事', 'AI小说改编', 'AI影视剪辑', 'AI剪辑', 'AI混剪', 'AI广告', 'AI宣传片', 'AI品牌', 'AI纪录片', 'AI科普', 'AI视频', 'AI视频作品', 'AI短视频', 'AI短片', 'AI动画', 'AI动漫', 'AI电影', 'AI电影短片', 'AIGC', 'AIGC视频', 'AIGC作品', 'AIGC短片', 'AI创作', 'AI生成视频', 'AI合成视频', 'AI主播', 'AI绘画动画', 'AI特效', 'Sora2', 'veo3', 'PikaAI', 'LumaAI', 'Seedance', '可灵AI', '可灵AI视频', 'KlingAI', '即梦AI', '即梦AI视频', '智影AI', '豆包AI视频', '海螺AI视频', 'ViduAI', 'B站AI创作大赛', 'B站AI', 'AI春晚', 'AIGC创作大赛', 'AI短视频大赛', 'AI视频创作大赛', 'AI神话', 'AI传说', 'AI神兽', 'AI灵兽', 'AI妖怪', 'AI佛', 'AI菩萨', 'AI神明', 'AI龙王', 'AI山海经', 'AI皮克斯', 'AI迪士尼', 'AI宫崎骏', 'AI吉卜力', 'AI国产动画', 'AI水彩动画', 'AI油画风', 'AI素描风', 'AI 3D动画', 'AI春秋', 'AI战国', 'AI秦始皇', 'AI汉武帝', 'AI李白', 'AI诗词', 'AI古诗', 'AI宋词', 'AI元曲', 'AI外星人', 'AI深空', 'AI空间站', 'AI宇航员', 'AI机甲战士', 'AI机器人战争', 'AI银河', 'AI黑洞', 'AI治愈', 'AI感动', 'AI催泪', 'AI温情', 'AI回忆', 'AI父爱', 'AI母爱', 'AI亲情', 'AI广告创意', 'AI产品宣传', 'AI电商', 'AI周杰伦', 'AI刘亦菲', 'AI李荣浩', 'AI成龙', 'AI邓丽君', 'AI张国荣', 'AI王家卫', 'AI篮球', 'AI足球', 'AI武林', 'AI美食AI烹饪', 'AI吃播', 'AI惊艳瞬间', 'AI名场面', 'AI震撼', 'AI视觉盛宴', 'AI混剪盛宴', 'AI神级', 'AI高燃', 'AI段子', 'AI趣味', 'AI奇遇', 'AI反转']

GENERIC_QUERIES = [
    "vlog", "日常", "旅行", "美食", "做饭", "宠物", "猫咪", "狗狗", "风景", "城市",
    "街拍", "健身", "跑步", "篮球", "足球", "露营", "钓鱼", "开箱", "手工", "摄影",
    "航拍", "校园", "毕业", "婚礼", "演唱会", "现场", "舞蹈", "翻唱", "乐器", "吉他",
    "搞笑", "沙雕", "整活", "生活记录", "上班", "下班", "通勤", "地铁", "夜景", "日出",
    "海边", "爬山", "骑行", "自驾", "火车", "飞机", "动物园", "公园", "菜市场", "早餐",
]
AI_TITLE_RE = re.compile(
    r"(ai|aigc|人工智能|生成|sora|可灵|kling|即梦|veo|runway|pika|海螺|vidu|"
    r"通义|万相|seedance|豆包|数字人|虚拟人|midjourney|stable ?diffusion|comfyui)",
    re.I)

LOGIN_COOKIES: dict = {}


def load_cookies(path: str | None) -> None:
    if not path or not os.path.exists(path):
        return
    m = {"BILIBILI_SESSDATA": "SESSDATA", "BILIBILI_BILI_JCT": "bili_jct",
         "BILIBILI_DEDEUSERID": "DedeUserID", "BILIBILI_BUVID3": "buvid3"}
    for line in open(path):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k in m and v:
                LOGIN_COOKIES[m[k]] = v
            elif k in m.values() and v:
                LOGIN_COOKIES[k] = v


def apply_login(s: requests.Session) -> None:
    for k, v in LOGIN_COOKIES.items():
        s.cookies.set(k, v, domain=".bilibili.com", path="/")


def new_session() -> requests.Session:
    s = B._bootstrap_session()
    apply_login(s)
    return s


def rebootstrap(s: requests.Session) -> None:
    B._rebootstrap(s)
    apply_login(s)


def get_json(s, url, params, tries=3):
    for a in range(tries):
        B._rate_limit_wait()
        try:
            r = s.get(url, params=params, timeout=12)
            if r.status_code == 412:
                log.warning("412 -> rebootstrap")
                rebootstrap(s)
                time.sleep(8 * (a + 1))
                continue
            return r.json()
        except (requests.RequestException, ValueError) as e:
            time.sleep(2 ** a + random.random())
            last = e
    return None


def search(s, q, page, begin, end, order):
    params = {"search_type": "video", "keyword": q, "order": order, "page": page,
              "page_size": 50, "pubtime_begin_s": begin, "pubtime_end_s": end}
    body = get_json(s, B._SEARCH_URL, B._get_signer(s).sign(params))
    if not body or body.get("code") != 0:
        return None
    d = body.get("data") or {}
    if d.get("v_voucher") and not d.get("result"):
        return None  # flagged
    return d.get("result") or []


def view(s, bvid):
    body = get_json(s, B._VIEW_URL, {"bvid": bvid}, tries=2)
    if body is None:
        return None
    return body


def iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None


def month_bounds(y, m):
    b = datetime(y, m, 1, tzinfo=CST)
    e = datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=CST)
    now = datetime.now(tz=CST)
    return int(b.timestamp()), int(min(e, now).timestamp()) - 1


def _f(name):
    return getattr(B, name, lambda *_a, **_k: False)


def cheap_filter(it) -> bool:
    t = it.get("title", "")
    for fn in ("_has_excluded_term", "_is_ai_music_title", "_is_deepfake_title",
               "_is_vtuber_title", "_is_ai_anchor_title"):
        if _f(fn)(t):
            return False
    d = it.get("duration_sec")
    if d is None or d < 3 or d > 600:
        return False
    tags = [x.strip() for x in (it.get("tag_str") or "").split(",") if x.strip()]
    low = {x.lower() for x in tags}
    if low & getattr(B, "_GAMING_TAG_KEYWORDS", set()):
        return False
    for fn in ("_is_vtuber_tags", "_is_tutorial_tags", "_is_ai_anchor_tags"):
        if _f(fn)(tags):
            return False
    return True


def crawl(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if args.seen:
        for p in args.seen.split(","):
            for line in open(p):
                try:
                    seen.add(json.loads(line).get("source_id") or "")
                except ValueError:
                    pass
    done_pages = set()
    if out.exists():
        for line in open(out):
            r = json.loads(line)
            if r.get("_type") == "page":
                done_pages.add(tuple(r["key"]))
            else:
                seen.add(r["source_id"])
    log.info("seen=%d done_pages=%d", len(seen), len(done_pages))

    queries = list(GENERIC_QUERIES if args.kind == "generic" else AI_QUERIES)
    rnd = random.Random(args.seed)
    rnd.shuffle(queries)
    s = new_session()
    deadline = time.time() + args.minutes * 60
    fh = open(out, "a")
    n_res = 0
    flagged = 0
    # round-robin: for each query, sweep all months (keeps months balanced)
    for q in queries:
        for (y, m) in MONTHS:
            for order in args.orders.split(","):
                for page in range(1, args.pages + 1):
                    if time.time() > deadline:
                        log.info("deadline reached; resolved=%d", n_res)
                        return
                    key = (q, y, m, order, page)
                    if key in done_pages:
                        continue
                    b, e = month_bounds(y, m)
                    items = search(s, q, page, b, e, order)
                    if items is None:
                        flagged += 1
                        log.warning("search flagged (%d) q=%s %d-%02d p%d", flagged, q, y, m, page)
                        rebootstrap(s)
                        time.sleep(min(120, 20 * flagged))
                        continue
                    flagged = 0
                    cands = []
                    for raw in items:
                        it = B._normalize_search_item(raw)
                        if not it["bvid"] or it["bvid"] in seen:
                            continue
                        seen.add(it["bvid"])
                        if cheap_filter(it):
                            cands.append(it)
                    rnd.shuffle(cands)
                    cands = cands[: args.per_page]
                    for it in cands:
                        body = view(s, it["bvid"])
                        if not body or body.get("code") != 0:
                            continue
                        d = body.get("data") or {}
                        msg = (d.get("argue_info") or {}).get("argue_msg", "") or ""
                        is_ai = any(x in msg for x in B._AI_ARGUE_MSGS) or (
                            "AI" in msg.upper() and ("生成" in msg or "合成" in msg))
                        title = d.get("title") or it["title"]
                        if is_ai:
                            label, src = "fake", "tier2_platform_tag"
                        elif args.kind == "generic" and not AI_TITLE_RE.search(title):
                            label, src = "real", "tier2_absence_of_tag"
                        else:
                            label, src = "other", "ai_query_untagged"
                        dim = d.get("dimension") or {}
                        rec = {
                            "source_platform": "bilibili",
                            "source_url": f"https://www.bilibili.com/video/{it['bvid']}",
                            "source_id": it["bvid"],
                            "label": label, "label_source": src,
                            "argue_msg": msg,
                            "claimed_generator": B._extract_generator(
                                f"{title} {d.get('desc','')} {it.get('tag_str','')}"),
                            "duration_sec": d.get("duration"),
                            "resolution_w": dim.get("width"), "resolution_h": dim.get("height"),
                            "tname": d.get("tname"),
                            "published_at": iso(d.get("pubdate")),
                            "crawled_at": datetime.now(tz=timezone.utc).isoformat(),
                            "query": q, "query_kind": args.kind,
                            "window": f"{y}-{m:02d}", "order": order,
                            "title": title,
                        }
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fh.flush()
                        n_res += 1
                    fh.write(json.dumps({"_type": "page", "key": list(key),
                                         "n_items": len(items), "n_cand": len(cands)}) + "\n")
                    fh.flush()
                    if len(items) < 50:
                        break  # no more pages in this window
                    time.sleep(args.search_gap)
    log.info("all queries exhausted; resolved=%d", n_res)


def liveness(args):
    rows = [json.loads(l) for l in open(args.liveness)]
    rows = [r for r in rows if r["source_platform"] == "bilibili"]
    gap = Path(args.liveness).parent / "splits" / "gap_test.jsonl"
    if gap.exists():  # prefer clips that have detector scores
        gid = {json.loads(l)["id"] for l in open(gap)}
        rows = [r for r in rows if r["id"] in gid]
        log.info("liveness restricted to gap_test bilibili rows: %d", len(rows))
    rnd = random.Random(args.seed)
    by = {}
    for r in rows:
        by.setdefault(r["label"], []).append(r)
    sample = []
    for lab, rs in by.items():
        rnd.shuffle(rs)
        sample += rs[: args.n // len(by)]
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in open(out)}
    s = new_session()
    deadline = time.time() + args.minutes * 60
    with open(out, "a") as fh:
        for r in sample:
            if r["id"] in done:
                continue
            if time.time() > deadline:
                break
            body = view(s, r["source_id"])
            code = None if body is None else body.get("code")
            state = None
            if body and body.get("code") == 0:
                state = (body.get("data") or {}).get("state")
            fh.write(json.dumps({"id": r["id"], "source_id": r["source_id"], "label": r["label"],
                                 "code": code, "state": state,
                                 "checked_at": datetime.now(tz=timezone.utc).isoformat()}) + "\n")
            fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["ai", "generic"], default="ai")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cookies", default=str(_root / "data/bilibili_cookies.env"))
    ap.add_argument("--seen", default="")
    ap.add_argument("--pages", type=int, default=2)
    ap.add_argument("--per-page", type=int, default=20)
    ap.add_argument("--orders", default="click,pubdate")
    ap.add_argument("--search-gap", type=float, default=8.0)
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--liveness", default=None)
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_cookies(args.cookies)
    log.info("login cookies loaded: %s", sorted(LOGIN_COOKIES))
    if args.liveness:
        liveness(args)
    else:
        crawl(args)


if __name__ == "__main__":
    main()
