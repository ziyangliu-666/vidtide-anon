# Reproducibility recipe

This document gives the **command sequence** required to reproduce every quantitative claim in the paper, once the corresponding scripts are uploaded to this anonymous repository (see top-level `README.md` for the staged-release plan).

---

## 0. Environment

```bash
git clone <this-repo>
cd vidtide-anon
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Hardware tested: single A100 80 GB container for FT runs; everything else fits on a 2× shared CPU box.

## 1. Recover the M0 evaluation slice

```bash
python scripts/download_videos.py \
    --manifest manifests/M0/splits/test.jsonl \
    --out data/M0/test/ \
    --workers 8
```

This uses each record's `source_url` to fetch the original clip from the platform of origin. The script reports:
- successful re-downloads,
- `sha256` mismatches (typically caused by platform re-encoding — see paper Section `sec:exp_platform`),
- and URLs that have rotted since the M0 freeze.

## 2. Table 1 — static-benchmark gap

```bash
python scripts/compute_gap.py \
    --bench-root data/M0/test/ \
    --detectors demamba_pika,demamba_crafter,nsgvd_pika,nsgvd_crafter,stil_pika,stil_crafter,npr_pika,npr_crafter,tall_pika,tall_crafter \
    --out results/gap.csv
```

Each detector adapter (in `eval/`) wraps the upstream detector's official inference path; we do **not** redistribute upstream weights — the adapters fetch them from the upstream GitHub releases the first time they run.

## 3. Table 2 — fine-tune recovery

```bash
# Linear-probe heads on K400-init backbones
for B in swin_t tsm i3d slowfast; do
    python scripts/ft_train_lp.py --backbone $B --slice M0 --epochs 20
    python scripts/ft_eval_all_slices.py --backbone $B --eval-slice M0
done

# Full fine-tune (DeMamba, B1 row of paper Table 2)
python scripts/demamba_full_ft.py --slice M0 --epochs 10
```

Hyperparameters are pinned in `scripts/ft_train_lp.py` and match Appendix `tab:appendix_hparams`.

## 4. Per-generator / per-platform analyses

```bash
python scripts/per_generator_full_pool.py --slice M0 --out results/per_generator/
python scripts/analyze_predict_results.py --results-dir results/ --by platform --out results/per_platform/
```

These produce the heatmaps in paper Section `sec:exp_generator` and `sec:exp_platform`.

## 5. Label-quality audit

```bash
python scripts/label_audit.py --slice M0 --report results/audit/M0.json
```

Reports the Cohen's $\kappa = 0.93$ figure in Section `sec:gtquality`.

---

> **Note.** During the double-blind review window, all of the above commands point at scripts that are being uploaded to this anonymous mirror in batches. The full set will be in place well before the discussion period closes; reviewers can track progress via the commit history of this repository.
