# Detector wrappers — Table 1 evaluation

The static-benchmark gap experiment (paper Table~`tab:gap`) evaluates seven detector configurations:

| Detector | Architecture family | Checkpoints used | Source |
|----------|--------------------|------------------|--------|
| DeMamba | Mamba | Pika & Crafter | Official GitHub |
| NSG-VD | Diffusion-prior + MMD | Pika (released); Crafter (re-trained by us per Appendix `sec:appendix_nsgvd_crafter`) | Official GitHub + our re-train |
| STIL | Spatio-temporal CNN | Pika & Crafter | Official GitHub |
| NPR | Frequency-domain CNN | Pika & Crafter | Official GitHub |
| TALL | Latent-space ViT | Pika & Crafter | Official GitHub |

This directory will host **thin adapter modules** that wrap each detector's official inference code so that `scripts/compute_gap.py` can call them through a uniform interface.

> **Skeleton notice.** Adapters land alongside `scripts/compute_gap.py`. We do **not** redistribute upstream weights; the adapters fetch each detector's official checkpoints from the upstream GitHub releases.
