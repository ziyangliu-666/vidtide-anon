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

## How M1–M7 were frozen

* **Frozen on 2026-11-07** as one release covering publication months May–November 2026, with
  both classes per month. Every row records when its metadata was resolved (`crawled_at`).
* **Continuous collection.** The rolling crawler has run every 72 h since May 2026.
  `rolling_crawl_log.jsonl` lists every row it collected with its original timestamp. Rows that
  fall into M1–M7 keep `first_seen_at` / `collection = rolling_crawler`.
* **Survival.** `../docs/URL_SURVIVAL.md` reports measured URL survival per label.
* **Label rule change on the platform.** Since late 2025 Bilibili also shows the disclosure string
  `含AI生成内容` (mandatory-labeling era). It is accepted as Tier-2 positive evidence and is kept
  verbatim in `label_evidence`.
* **Frozen-detector check.** `manifests/evidence/newslice_eval_2026-11-07/` lists the 196 clips from M1–M2
  and M5–M7 that were scored by two frozen detectors, with per-clip scores and AUROCs.
* From M8 on, slices are frozen at the end of each month by the same script
  (`scripts/crawl_meta_slices.py` + `scripts/build_meta_release.py`).

Removal requests: see `../docs/OPT_OUT.md`. Manifests contain no uploader names or IDs.
