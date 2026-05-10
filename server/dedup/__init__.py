"""Cross-platform video deduplication via thumbnail captioning + embeddings.

Pipeline shape:
    thumbnail_bytes
      → captioner (img2txt)    — server/dedup/captioner.py
      → embedder (text → vec)  — server/dedup/embedder.py
      → vec_index (sqlite-vec) — server/dedup/vec_index.py
      → deduplicator           — server/dedup/deduplicator.py
          (KNN search → canonical pick → duplicate_of_id write)

All modules are lazily imported. Importing `server.dedup` itself does NOT
pull torch / sentence-transformers / sqlite-vec — those are only loaded
when their respective classes are first instantiated. This matters because
the FastAPI cloud app imports `server.routers.*` which can drag `server.*`
into process and we don't want every request to pay the torch-import cost.
"""
