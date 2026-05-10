# Detector wrappers — Table 1 evaluation

Thin adapter modules around each of the seven detector configurations
benchmarked in paper Table 1. Implementations live under
[`server/detection/detectors/`](../server/detection/detectors/) and are
exposed through `server.detection.registry.get_detector(name)`.

| Detector | Architecture family | Module(s) | Source |
|----------|--------------------|-----------|--------|
| DeMamba | Mamba | `demamba_pika.py`, `demamba_crafter.py` | Official GitHub |
| NSG-VD  | Diffusion-prior + MMD discriminator | `nsgvd_pika.py`, `nsgvd_crafter.py` | Official GitHub + our re-train (Crafter; see Appendix `sec:appendix_nsgvd_crafter`) |
| STIL    | Spatio-temporal CNN | `stil_pika.py`, `stil_crafter.py` | Official GitHub |
| NPR     | Frequency-domain CNN | `npr_pika.py`, `npr_crafter.py` | Official GitHub |
| TALL    | Latent-space ViT | `tall_pika.py`, `tall_crafter.py` | Official GitHub |
| CLIP zero-shot | CLIP ViT-B/32 prompt | `clip_zero_shot.py` | Open CLIP |
| GPT-4o vision | API | `gpt4o_vision.py` | OpenAI |

We do **not** redistribute upstream weights; each adapter fetches its
official checkpoint from the upstream GitHub release on first use.

The driver `scripts/compute_gap.py` calls these adapters through the
uniform interface to produce paper Table 1; cached per-clip scores end
up in `results/bench_5k_*_scores.jsonl`. NSG-VD's Crafter checkpoint
(not released by the original authors) is re-trained by
`scripts/nsgvd_train_discriminator.py` following the protocol in
Appendix `sec:appendix_nsgvd_crafter`.
