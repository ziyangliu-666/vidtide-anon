# URL survival and deletion bias

## 2026-11-07 check (M0, about seven months after collection)

- **Sample.** 791 Bilibili rows from the detector evaluation pool (`manifests/M0/splits/gap_test.jsonl`):
  400 generated and 391 real. Nine requests failed and are excluded.
- **Survival by label** (Wilson 95% CIs):

  | Label | Checked | Alive | Survival |
  |---|---|---|---|
  | Generated | 400 | 370 | 92.5% [89.5, 94.7] |
  | Real | 391 | 387 | 99.0% [97.4, 99.6] |

- **Effect on evaluation.** Rescoring ten zero-shot detectors on survivors only changes AUROC by
  +0.24 pp on average, and by at most 0.87 pp.

Deletion is label-correlated, so every release reports survival per label, and cross-release
comparisons use the shared surviving subset. Per-row results:
`manifests/evidence/url_survival_2026-11-07/`.
