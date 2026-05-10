"""Monthly benchmark slice publishing.

This module owns the contract between a frozen BenchmarkSlice row in
the local DB and a published HuggingFace Datasets release.

Entry points:
  - `hf_publisher.publish_slice(slice_id, db, hf_token, repo_id)`
    Synchronously pushes a slice to HF Datasets and returns the URL.

See REQUIREMENTS.md R3 for the monthly release cadence contract.
"""
