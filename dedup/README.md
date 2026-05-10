# Cross-platform deduplication

CLIP-based perceptual deduplicator that catches cross-platform reposts (e.g.
the same Sora 2 demo appearing on YouTube, Bilibili, and Reddit). Full
implementation in [`server/dedup/`](../server/dedup/).

Pipeline:

1. Sample N keyframes per clip (`server/dedup/captioner.py`; default N=8, evenly spaced).
2. Encode with a CLIP image tower (`server/dedup/image_embedder.py`, ViT-B/32 by default).
3. Mean-pool to a single 512-d clip embedding; L2-normalise.
4. Approximate-nearest-neighbour query against the existing index
   (`server/dedup/vec_index.py`, sqlite-vec).
5. Match if cosine similarity ≥ `dedup_threshold` (default 0.92).

The duplicate with the **highest-tier provenance** (T1 > T2 > T3) is kept;
lower-tier duplicates are dropped from the manifest but the dropped record's
`source_url` is recorded in `dedup_meta` (audit log).

The orchestration entry point is
[`server.dedup.deduplicator.Deduplicator`](../server/dedup/deduplicator.py);
`scripts/recompute_dedup.py` re-runs it over the current DB.
