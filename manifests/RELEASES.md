# Release log

| Slice | Publication month | Clips | Generated | Real | Balanced eval | Rolling-crawler rows |
|---|---|---|---|---|---|---|
| M0 | <= 2026-04 | 21,504 | 11,502 | 10,002 | 9,956 (gap_test) | — |
| M1 | 2026-05 | 771 | 398 | 373 | 746 | 112 |
| M2 | 2026-06 | 632 | 334 | 298 | 596 | 28 |
| M3 | 2026-07 | 587 | 310 | 277 | 554 | 1 |
| M4 | 2026-08 | 619 | 339 | 280 | 560 | 2 |
| M5 | 2026-09 | 623 | 352 | 271 | 542 | 1 |
| M6 | 2026-10 | 620 | 369 | 251 | 502 | 0 |
| M7 | 2026-11 (1–7 Nov, partial) | 528 | 241 | 287 | 482 | 1 |

## How M1–M7 were frozen (read before comparing slices)

* **Frozen on 2026-11-07.** M1–M7 are defined by Bilibili publication month. Monthly freezes were
  **not** executed on schedule between May and October 2026. We do not backdate them: every row
  carries the time its metadata was resolved (`crawled_at`).
* **Continuous collection did run.** The off-site rolling crawler ran every 72 h from May 2026 on
  (fake side only, ~100–150 rows per month; per-month counts by crawl time: 2026-05: 92, 2026-06: 146, 2026-07: 127, 2026-08: 124, 2026-09: 131, 2026-10: 102, 2026-11: 17).
  `rolling_crawl_log.jsonl` lists every row it collected with its original timestamp; rows that
  fall into M1–M7 keep `first_seen_at` / `collection = rolling_crawler`.
* **Retrospective resolution and survivorship.** Rows with `collection = retrospective_resolve`
  were resolved on 2026-11-07. Videos deleted before that date cannot appear, so older slices are
  survivor-filtered more than newer ones; `../docs/URL_SURVIVAL.md` reports measured URL survival per label.
* **Label rule change on the platform.** Since late 2025 Bilibili also shows the disclosure string
  `含AI生成内容` (mandatory-labeling era). It is accepted as Tier-2 positive evidence and is kept
  verbatim in `label_evidence`.
* **Frozen-detector check.** `manifests/evidence/newslice_eval_2026-11-07/` lists the 196 clips from M1–M2
  and M5–M7 that were scored by two frozen detectors, with per-clip scores and AUROCs.
* From M8 on, slices are frozen at the end of each month by the same script
  (`scripts/crawl_meta_slices.py` + `scripts/rebuttal_iclr/build_meta_release.py`).

Removal requests: see `../docs/OPT_OUT.md`. Manifests contain no uploader names or IDs.
