# Cross-platform deduplication

CLIP-based perceptual deduplicator that catches cross-platform reposts (e.g. the same Sora 2 demo appearing on YouTube, Bilibili, and Reddit).

Pipeline:

1. Sample N keyframes per clip (default N=8, evenly spaced).
2. Encode with CLIP ViT-B/32 image tower.
3. Mean-pool to a single 512-d clip embedding; L2-normalise.
4. Approximate-nearest-neighbour query against the existing index (FAISS).
5. Match if cosine similarity ≥ `dedup_threshold` (default 0.92).

The duplicate with the **highest-tier provenance** (T1 > T2 > T3) is kept; lower-tier duplicates are dropped from the manifest but their `source_url`s are recorded in an audit log.

> **Skeleton notice.** `clip_dedup.py` is documented here and will be released in the next batch alongside the audit-log schema.
