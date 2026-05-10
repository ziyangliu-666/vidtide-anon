"""Near-duplicate audit using perceptual hashes (pHash) over middle frames.

For each video in a 10% held-out subset, extract one middle frame and compute
its perceptual hash. Group by hash within Hamming distance ≤ 6 to find
near-duplicates. Reports counts within-platform and cross-platform.

Outputs:
  - results/near_dup_audit.json
  - results/near_dup_audit.md
"""

from __future__ import annotations

import json
import random
import sqlite3
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "vidtide.db"
RESULTS_DIR = REPO / "results"
HOLDOUT_FRAC = 0.10
HAMMING_THRESH = 6  # pHash distance ≤ 6 = visually near-identical
SEED = 42


def load_videos() -> list[dict]:
    """Videos that have a local mp4 in data/blobs/videos/{label}/."""
    blob_root = REPO / "data" / "blobs" / "videos"
    on_disk: dict[str, str] = {}
    for sub in ("fake", "real"):
        for p in (blob_root / sub).glob("*.mp4"):
            on_disk[p.stem] = sub
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT id, source_platform, label, COALESCE(NULLIF(claimed_generator, ''), '') AS gen "
        "FROM videos"
    ).fetchall()
    con.close()
    out = []
    for r in rows:
        if r[0] in on_disk:
            out.append({"id": r[0], "plat": r[1], "label": r[2], "gen": r[3] or None})
    return out


def extract_middle_frame(video_path: Path, out_png: Path) -> bool:
    """Extract one frame near the middle of the video as PNG."""
    try:
        # Probe duration
        d = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(video_path)],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode().strip()
        dur = float(d) if d else 5.0
    except Exception:
        dur = 5.0
    mid_t = max(0.5, dur / 2)
    try:
        subprocess.run(
            ["ffmpeg", "-ss", str(mid_t), "-i", str(video_path),
             "-frames:v", "1", "-vf", "scale=128:128", "-y", str(out_png)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=15, check=True,
        )
        return out_png.exists() and out_png.stat().st_size > 0
    except Exception:
        return False


def main() -> None:
    rng = random.Random(SEED)
    all_vids = load_videos()
    rng.shuffle(all_vids)
    n_holdout = int(len(all_vids) * HOLDOUT_FRAC)
    holdout = all_vids[:n_holdout]
    print(f"Total: {len(all_vids)}, held-out 10%: {len(holdout)}")

    # Hash each held-out video's middle frame
    blob_root = REPO / "data" / "blobs" / "videos"
    hashes: list[tuple[int, dict]] = []  # (int_hash, meta)
    misses = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, v in enumerate(holdout, 1):
            label_dir = "fake" if v["label"] == "fake" else "real"
            mp4 = blob_root / label_dir / f"{v['id']}.mp4"
            if not mp4.exists():
                misses += 1
                continue
            png = Path(tmp) / f"{v['id']}.png"
            if not extract_middle_frame(mp4, png):
                misses += 1
                continue
            try:
                ph = imagehash.phash(Image.open(png).convert("RGB"))
                hashes.append((int(str(ph), 16), v))
            except Exception:
                misses += 1
            if i % 200 == 0:
                print(f"  [{i}/{len(holdout)}] hashed={len(hashes)} miss={misses}")

    print(f"Hashed {len(hashes)} videos ({misses} extraction misses)")

    # Pair-wise Hamming distance via brute-force (~N^2/2 ints; ok for ~3K)
    def hamming(a: int, b: int) -> int:
        return bin(a ^ b).count("1")

    # Greedy clustering: each video joins the first existing cluster
    # whose representative is within HAMMING_THRESH; else starts new cluster.
    clusters: list[dict] = []  # {rep_hash, members: [...]}
    for h, meta in hashes:
        joined = False
        for c in clusters:
            if hamming(h, c["rep_hash"]) <= HAMMING_THRESH:
                c["members"].append(meta)
                joined = True
                break
        if not joined:
            clusters.append({"rep_hash": h, "members": [meta]})

    # Stats
    multi_clusters = [c for c in clusters if len(c["members"]) > 1]
    n_dup_videos = sum(len(c["members"]) - 1 for c in multi_clusters)
    cross_plat = sum(
        1 for c in multi_clusters
        if len({m["plat"] for m in c["members"]}) > 1
    )
    cross_label = sum(
        1 for c in multi_clusters
        if len({m["label"] for m in c["members"]}) > 1
    )

    out = {
        "total_videos_in_db": len(all_vids),
        "holdout_size": len(holdout),
        "holdout_hashed": len(hashes),
        "extraction_misses": misses,
        "n_clusters": len(clusters),
        "n_singleton_clusters": sum(1 for c in clusters if len(c["members"]) == 1),
        "n_multi_clusters": len(multi_clusters),
        "n_videos_in_multi_clusters": sum(len(c["members"]) for c in multi_clusters),
        "n_redundant_videos": n_dup_videos,
        "redundancy_rate": round(n_dup_videos / max(len(hashes), 1), 4),
        "n_cross_platform_dup_clusters": cross_plat,
        "n_cross_label_dup_clusters": cross_label,
        "hamming_threshold": HAMMING_THRESH,
        "top_clusters": [
            {
                "size": len(c["members"]),
                "platforms": sorted({m["plat"] for m in c["members"]}),
                "labels": sorted({m["label"] for m in c["members"]}),
                "video_ids": [m["id"] for m in c["members"][:10]],
            }
            for c in sorted(multi_clusters, key=lambda c: -len(c["members"]))[:25]
        ],
    }
    out_path = RESULTS_DIR / "near_dup_audit.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"→ {out_path}")

    md = RESULTS_DIR / "near_dup_audit.md"
    with md.open("w") as f:
        f.write("# Near-duplicate audit (pHash on middle frame)\n\n")
        f.write(f"- Pool: {out['total_videos_in_db']} videos in DB\n")
        f.write(f"- Held-out 10%: {out['holdout_size']}, successfully hashed {out['holdout_hashed']} "
                f"({out['extraction_misses']} extraction failures)\n")
        f.write(f"- Hamming threshold: ≤ {out['hamming_threshold']} bits = visually near-identical\n\n")
        f.write("## Headline\n\n")
        f.write(f"- **Redundancy rate**: {out['redundancy_rate']:.2%} "
                f"({out['n_redundant_videos']} videos are near-duplicates of another)\n")
        f.write(f"- {out['n_multi_clusters']} clusters with >1 member, "
                f"covering {out['n_videos_in_multi_clusters']} videos\n")
        f.write(f"- Cross-platform dup clusters: {out['n_cross_platform_dup_clusters']} "
                f"(reposts across platforms)\n")
        f.write(f"- Cross-label dup clusters: {out['n_cross_label_dup_clusters']} "
                f"(same video labeled differently — **suspect**)\n\n")
        f.write("## Top 15 largest near-duplicate clusters\n\n")
        f.write("| size | platforms | labels | sample IDs |\n|---|---|---|---|\n")
        for c in out["top_clusters"][:15]:
            ids = ", ".join(f"`{i[:10]}`" for i in c["video_ids"][:3])
            if len(c["video_ids"]) > 3:
                ids += f" +{len(c['video_ids']) - 3} more"
            f.write(f"| {c['size']} | {','.join(c['platforms'])} | {','.join(c['labels'])} | {ids} |\n")
    print(f"→ {md}")


if __name__ == "__main__":
    main()
