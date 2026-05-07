# Pipeline architecture

```
                    ┌─────────────────────────┐
                    │     Platform Crawlers    │
                    │  YouTube · Reddit · 站B  │
                    │   + Official galleries   │
                    └────────────┬─────────────┘
                                 │ candidate clips with provenance tier
                                 ▼
                    ┌─────────────────────────┐
                    │      Quality filter      │
                    │  resolution · duration   │
                    │            · fps         │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      Tag-tier filter     │
                    │      T1 / T2 / T3        │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     LLM verification     │
                    │   GPT-4o title / tags    │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   CLIP cross-platform    │
                    │      deduplication       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  Cleanlab label audit    │
                    │       (κ = 0.93)         │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  Monthly immutable slice │
                    │   M0, M1, … (metadata)   │
                    └─────────────────────────┘
```

## Tier-source taxonomy

- **T1 — official showcase.** Per-vendor scrapers of model-developer galleries (Pika, Kling, Runway, Dreamina, ...). Definitionally AI-generated.
- **T2 — platform AI tag.** YouTube AI-disclosure (mandatory since 2024); Bilibili `argue_info` (China AI labelling regulation, Sept 2025); TikTok C2PA (covering 47+ generators).
- **T3 — keyword + LLM verification.** Title/description keyword match, then GPT-4o semantic check to separate *actual AI clips* from *tutorials about AI*.

## Cost breakdown

The pipeline is designed to run on a single small box. Standard-mode operation is approximately **\$156/month**:

| Component | Standard mode |
|-----------|---------------|
| Compute (single shared-CPU box, autostop) | ~\$5 |
| Storage (5 GB volume — metadata + thumbnails only, no video files) | ~\$1 |
| Bandwidth (under free-tier) | \$0 |
| LLM verification (GPT-4o-mini, ~50K title classifications/month) | ~\$150 |
| **Total** | **~\$156/mo** |

This is roughly an order of magnitude below the per-month cost of industry-curated video datasets (paper Section `sec:method`).
