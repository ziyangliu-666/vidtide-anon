"""Build metadata-only monthly slices M1..M7 (publication months 2026-05 .. 2026-11).

Inputs : new retrospective crawl (ai + generic jsonl), rolling-crawler rows exported from the
         off-site crawler DB, M0 manifest (for de-dup).
Outputs: <repo>/manifests/M{k}/metadata.jsonl, eval_balanced.jsonl, README.md,
         <repo>/manifests/rolling_crawl_log.jsonl, <repo>/manifests/RELEASES.md, SHA256SUMS
"""
import hashlib, json, random, sys, collections
from datetime import datetime, timezone, timedelta
from pathlib import Path

CST = timezone(timedelta(hours=8))
FREEZE = "2026-11-07"
REPO = Path(sys.argv[1])
AI, GEN, ROLL, M0 = sys.argv[2], sys.argv[3], sys.argv[4], REPO / "manifests/M0/metadata.jsonl"
MONTHS = [f"2026-{m:02d}" for m in range(5, 12)]
SLICE = {m: f"M{i+1}" for i, m in enumerate(MONTHS)}

m0 = {json.loads(l)["source_id"] for l in open(M0)}


def pub_month(iso):
    if not iso:
        return None
    return datetime.fromisoformat(iso).astimezone(CST).strftime("%Y-%m")


def rid(sid):
    return hashlib.sha1(f"bilibili:{sid}".encode()).hexdigest()[:32]


rows = {}
for path in (AI, GEN):
    for l in open(path):
        r = json.loads(l)
        if r.get("_type") or r["label"] not in ("fake", "real") or r["source_id"] in m0:
            continue
        rows[r["source_id"]] = dict(
            id=rid(r["source_id"]), source_platform="bilibili", source_url=r["source_url"],
            source_id=r["source_id"], label=r["label"], label_source=r["label_source"],
            label_evidence=r["argue_msg"] or None, claimed_generator=r["claimed_generator"],
            duration_sec=r["duration_sec"], resolution_w=r["resolution_w"],
            resolution_h=r["resolution_h"], published_at=r["published_at"],
            crawled_at=r["crawled_at"], collection="retrospective_resolve")

roll = json.load(open(ROLL))
log_rows = []
for r in roll:
    log_rows.append(dict(source_id=r["source_id"], source_url=r["source_url"], label=r["label"],
                         label_source=r["label_source"], published_at=r["published_at"],
                         crawled_at=r["crawled_at"]))
    sid = r["source_id"]
    if sid in m0 or sid in rows:
        if sid in rows:  # keep contemporaneous crawl time as provenance
            rows[sid]["first_seen_at"] = r["crawled_at"]
            rows[sid]["collection"] = "rolling_crawler"
        continue
    pub = (r["published_at"] or "").replace(" ", "T") or None
    if pub and not pub.endswith("Z") and "+" not in pub[10:]:
        pub = pub + "+00:00"
    r["crawled_at"] = r["crawled_at"].replace(" ", "T")
    rows[sid] = dict(
        id=rid(sid), source_platform="bilibili", source_url=r["source_url"], source_id=sid,
        label=r["label"], label_source=r["label_source"], label_evidence=None,
        claimed_generator=r["claimed_generator"], duration_sec=r["duration_sec"],
        resolution_w=r["resolution_w"], resolution_h=r["resolution_h"], published_at=pub,
        crawled_at=r["crawled_at"], first_seen_at=r["crawled_at"], collection="rolling_crawler")

by = collections.defaultdict(list)
for r in rows.values():
    pm = pub_month(r["published_at"])
    if pm in SLICE:
        r["slice"] = SLICE[pm]
        by[pm].append(r)

out = REPO / "manifests"
summary = []
rnd = random.Random(0)
for pm in MONTHS:
    s = SLICE[pm]
    d = out / s
    d.mkdir(parents=True, exist_ok=True)
    rs = sorted(by[pm], key=lambda r: r["id"])
    with open(d / "metadata.jsonl", "w") as fh:
        for r in rs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    f = [r for r in rs if r["label"] == "fake"]
    re_ = [r for r in rs if r["label"] == "real"]
    k = min(len(f), len(re_))
    bal = rnd.sample(f, k) + rnd.sample(re_, k)
    with open(d / "eval_balanced.jsonl", "w") as fh:
        for r in sorted(bal, key=lambda r: r["id"]):
            fh.write(json.dumps({"id": r["id"], "label": r["label"], "source_url": r["source_url"]}) + "\n")
    roll_n = sum(r["collection"] == "rolling_crawler" for r in rs)
    summary.append((s, pm, len(rs), len(f), len(re_), 2 * k, roll_n))
    (d / "README.md").write_text(f"""# {s} — publication month {pm} (metadata-only, frozen {FREEZE})

| | Clips |
|---|---|
| Total | {len(rs)} ({len(f)} generated / {len(re_)} real) |
| Balanced evaluation subset (`eval_balanced.jsonl`) | {2*k} |
| Rows first seen by the rolling crawler at collection time | {roll_n} |

Slice membership = Bilibili publication month ({pm}, UTC+8). Metadata only; no video bytes are
redistributed. Labels follow the M0 Tier-2 rules: `fake` = Bilibili AI-disclosure string present
(`label_evidence` stores it verbatim); `real` = no disclosure on a non-AI-keyword query and no AI
keyword in the title (`tier2_absence_of_tag`). See `../RELEASES.md` for how this slice was frozen.
""")

with open(out / "rolling_crawl_log.jsonl", "w") as fh:
    for r in sorted(log_rows, key=lambda r: r["crawled_at"]):
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

tbl = "\n".join(f"| {s} | {pm} | {n} | {nf} | {nr} | {nb} | {rn} |" for s, pm, n, nf, nr, nb, rn in summary)
crawl_months = collections.Counter(r["crawled_at"][:7] for r in log_rows)
cm = ", ".join(f"{k}: {v}" for k, v in sorted(crawl_months.items()))
(out / "RELEASES.md").write_text(f"""# Release log

| Slice | Publication month | Clips | Generated | Real | Balanced eval | Rolling-crawler rows |
|---|---|---|---|---|---|---|
| M0 | <= 2026-04 | 21,504 | 11,502 | 10,002 | 9,956 (gap_test) | — |
{tbl}

## How M1–M7 were frozen (read before comparing slices)

* **Frozen on {FREEZE}.** M1–M7 are defined by Bilibili publication month. Monthly freezes were
  **not** executed on schedule between May and October 2026. We do not backdate them: every row
  carries the time its metadata was resolved (`crawled_at`).
* **Continuous collection did run.** The off-site rolling crawler ran every 72 h from May 2026 on
  (fake side only, ~100–150 rows per month; per-month counts by crawl time: {cm}).
  `rolling_crawl_log.jsonl` lists every row it collected with its original timestamp; rows that
  fall into M1–M7 keep `first_seen_at` / `collection = rolling_crawler`.
* **Retrospective resolution and survivorship.** Rows with `collection = retrospective_resolve`
  were resolved on {FREEZE}. Videos deleted before that date cannot appear, so older slices are
  survivor-filtered more than newer ones; `../docs` reports measured URL survival per label.
* **Label rule change on the platform.** Since late 2025 Bilibili also shows the disclosure string
  `含AI生成内容` (mandatory-labeling era). It is accepted as Tier-2 positive evidence and is kept
  verbatim in `label_evidence`.
* From M8 on, slices are frozen at the end of each month by the same script
  (`scripts/crawl_meta_slices.py` + `scripts/rebuttal_iclr/build_meta_release.py`).

Removal requests: see `../docs/OPT_OUT.md`. Manifests contain no uploader names or IDs.
""")

# checksums
lines = []
for p in sorted(out.rglob("*.jsonl")):
    lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}")
(out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
for row in summary:
    print(row)
