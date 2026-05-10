"""Cluster the Unknown-generator fake videos to demonstrate >=5 generator families.

Rebuttal experiment for NeurIPS 2026 reviewer Q1.

Usage:
    python scripts/cluster_unknown.py --phase extract   # ~25 min
    python scripts/cluster_unknown.py --phase cluster   # ~5 min
    python scripts/cluster_unknown.py --phase all       # both
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from server.dedup.image_embedder import ClipImageEmbedder

DB_PATH = REPO / "data" / "vidtide.db"
BLOB_DIR = REPO / "data" / "blobs" / "videos" / "fake"
CACHE_DIR = REPO / "data" / "cluster_cache"
FRAMES_DIR = CACHE_DIR / "frames"
RESULTS_DIR = REPO / "results"
FIGS_DIR = CACHE_DIR / "figs"

FRAME_SIZE = 256
N_FRAMES = 8                  # frames per video, evenly spaced
CLIP_BATCH_VIDEOS = 32        # videos per batch (so 32 * 8 = 256 frames)
FFT_BATCH_VIDEOS = 16
FFT_N_BINS = 32
PATCH_SHUFFLE_SIZE = 32       # patch-shuffle CLIP inputs (SemAnti 2024 style)
                              # 32 = CLIP B/32's native grid; destroys global semantics
                              # while preserving local generator artifacts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cluster_unknown")


# ---------------------------- DB / index ----------------------------


def load_index() -> tuple[list[str], list[str | None], list[str]]:
    import sqlite3

    c = sqlite3.connect(DB_PATH)
    rows = c.execute(
        "select id, claimed_generator, source_platform from videos where label='fake'"
    ).fetchall()
    ids, gens, plats = [], [], []
    for vid_id, gen, plat in rows:
        if (BLOB_DIR / f"{vid_id}.mp4").exists():
            ids.append(vid_id)
            gens.append(gen if gen else None)
            plats.append(plat or "unknown")
    return ids, gens, plats


# ---------------------------- frame extraction (cv2) ----------------------------


def extract_frames(vid_id: str) -> bool:
    """Extract N_FRAMES evenly-spaced frames using cv2. Returns True if all succeeded."""
    import cv2

    out_paths = [FRAMES_DIR / f"{vid_id}_{i}.jpg" for i in range(N_FRAMES)]
    if all(p.exists() and p.stat().st_size > 0 for p in out_paths):
        return True

    mp4 = BLOB_DIR / f"{vid_id}.mp4"
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        return False
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if nf < 1:
        cap.release()
        return False

    saved = 0
    for i in range(N_FRAMES):
        idx = max(0, min(nf - 1, int(nf * (i + 0.5) / N_FRAMES)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        scale = FRAME_SIZE / max(h, w)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(frame, (nw, nh))
        canvas = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
        y0, x0 = (FRAME_SIZE - nh) // 2, (FRAME_SIZE - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        cv2.imwrite(str(out_paths[i]), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
        saved += 1
    cap.release()
    return saved == N_FRAMES


def extract_frames_parallel(ids: list[str], workers: int = 8) -> None:
    import concurrent.futures

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    def needs(v):
        return not all((FRAMES_DIR / f"{v}_{i}.jpg").exists()
                       and (FRAMES_DIR / f"{v}_{i}.jpg").stat().st_size > 0
                       for i in range(N_FRAMES))

    todo = [v for v in ids if needs(v)]
    log.info("frames: %d videos cached, %d to extract", len(ids) - len(todo), len(todo))
    if not todo:
        return
    t0 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in ex.map(extract_frames, todo, chunksize=4):
            done += 1
            if done % 200 == 0:
                rate = done / (time.time() - t0)
                eta = (len(todo) - done) / rate
                log.info("  extracted %d/%d videos  (%.1f vid/s, ETA %.0fs)",
                         done, len(todo), rate, eta)
    log.info("  extraction wall time: %.1fs", time.time() - t0)


def filter_ids_with_frames(ids: list[str]) -> list[str]:
    keep = []
    for v in ids:
        if all((FRAMES_DIR / f"{v}_{i}.jpg").exists()
               and (FRAMES_DIR / f"{v}_{i}.jpg").stat().st_size > 0
               for i in range(N_FRAMES)):
            keep.append(v)
    if len(keep) < len(ids):
        log.info("filtered %d -> %d ids (dropped %d incomplete)",
                 len(ids), len(keep), len(ids) - len(keep))
    return keep


# ---------------------------- CLIP features ----------------------------


def _patch_shuffle(img_pil: "Image.Image", patch_size: int, rng: np.random.Generator) -> "Image.Image":
    """Permute non-overlapping patches of an image. Destroys global semantics, preserves local texture."""
    arr = np.asarray(img_pil)
    h, w = arr.shape[:2]
    nh, nw = h // patch_size, w // patch_size
    if nh == 0 or nw == 0:
        return img_pil
    arr = arr[:nh * patch_size, :nw * patch_size]
    if arr.ndim == 2:
        arr = arr[..., None]
    # [nh, P, nw, P, C] -> [nh, nw, P, P, C]
    patches = arr.reshape(nh, patch_size, nw, patch_size, -1).swapaxes(1, 2)
    flat = patches.reshape(nh * nw, patch_size, patch_size, -1)
    perm = rng.permutation(nh * nw)
    flat = flat[perm]
    patches = flat.reshape(nh, nw, patch_size, patch_size, -1).swapaxes(1, 2)
    out = patches.reshape(nh * patch_size, nw * patch_size, -1)
    if out.shape[-1] == 1:
        out = out[..., 0]
    return Image.fromarray(out)


def compute_clip_features(ids: list[str], patch_shuffle: bool = True) -> np.ndarray:
    """Mean-pool 8 frames per video, optionally patch-shuffled. Return [N,512] L2-normalized."""
    suffix = "_shuffled" if patch_shuffle else ""
    cache = CACHE_DIR / f"clip{suffix}_features.npy"
    cache_ids = CACHE_DIR / f"clip{suffix}_ids.json"
    if cache.exists() and cache_ids.exists():
        cached_ids = json.loads(cache_ids.read_text())
        if cached_ids == ids:
            log.info("loading cached CLIP%s features", suffix)
            return np.load(cache)

    import torch

    embedder = ClipImageEmbedder()
    embedder._ensure_loaded()
    feats = np.zeros((len(ids), 512), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(ids), CLIP_BATCH_VIDEOS):
        batch_ids = ids[i:i + CLIP_BATCH_VIDEOS]
        images = []
        for vid in batch_ids:
            for j in range(N_FRAMES):
                img = Image.open(FRAMES_DIR / f"{vid}_{j}.jpg").convert("RGB")
                if patch_shuffle:
                    seed = (hash(f"{vid}_{j}") & 0xFFFFFFFF)
                    rng = np.random.default_rng(seed)
                    img = _patch_shuffle(img, PATCH_SHUFFLE_SIZE, rng)
                images.append(img)
        with torch.no_grad():
            inputs = embedder._processor(images=images, return_tensors="pt").to(embedder._device)
            vision_out = embedder._model.vision_model(pixel_values=inputs["pixel_values"])
            f = embedder._model.visual_projection(vision_out.pooler_output)
            f = f / f.norm(dim=-1, keepdim=True)
        out = f.cpu().numpy().astype(np.float32)
        out = out.reshape(len(batch_ids), N_FRAMES, 512).mean(axis=1)
        out = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
        feats[i:i + len(batch_ids)] = out
        if (i // CLIP_BATCH_VIDEOS) % 5 == 0:
            done = i + len(batch_ids)
            rate = done / (time.time() - t0 + 1e-6)
            eta = (len(ids) - done) / rate
            log.info("  CLIP%s %d/%d videos  (%.1f vid/s, ETA %.0fs)",
                     suffix, done, len(ids), rate, eta)
    np.save(cache, feats)
    cache_ids.write_text(json.dumps(ids))
    log.info("CLIP%s features saved: %s", suffix, cache)
    return feats


# ---------------------------- DCT/FFT radial spectrum ----------------------------


def radial_fft_spectrum_batch(ids: list[str], n_bins: int = FFT_N_BINS) -> np.ndarray:
    """Mean-pool radial FFT spectrum across 8 frames per video. Return [N, n_bins]."""
    cache = CACHE_DIR / "fft_features.npy"
    cache_ids = CACHE_DIR / "fft_ids.json"
    if cache.exists() and cache_ids.exists():
        cached_ids = json.loads(cache_ids.read_text())
        if cached_ids == ids:
            log.info("loading cached FFT features")
            return np.load(cache)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    h, w = FRAME_SIZE, FRAME_SIZE
    cy, cx = h / 2, w / 2
    yy, xx = np.indices((h, w))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_r = min(cy, cx)
    bin_idx = np.clip(np.digitize(rr, np.linspace(0, max_r, n_bins + 1)) - 1, 0, n_bins - 1)
    bin_idx_t = torch.from_numpy(bin_idx).to(device).long().flatten()
    bin_counts = torch.bincount(bin_idx_t, minlength=n_bins).clamp(min=1).float()

    feats = np.zeros((len(ids), n_bins), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(ids), FFT_BATCH_VIDEOS):
        batch_ids = ids[i:i + FFT_BATCH_VIDEOS]
        # load 8 frames per video, stack to [B*8, h, w]
        imgs = []
        for vid in batch_ids:
            for j in range(N_FRAMES):
                try:
                    img = Image.open(FRAMES_DIR / f"{vid}_{j}.jpg").convert("L")
                    imgs.append(np.asarray(img, dtype=np.float32) / 255.0)
                except Exception:
                    imgs.append(np.zeros((h, w), dtype=np.float32))
        x = torch.from_numpy(np.stack(imgs)).to(device)  # [B*8, h, w]
        F = torch.fft.fft2(x)
        F = torch.fft.fftshift(F, dim=(-2, -1))
        mag = torch.log1p(F.abs())
        flat = mag.flatten(1)  # [B*8, h*w]
        out = torch.zeros((flat.shape[0], n_bins), device=device, dtype=torch.float32)
        out.scatter_add_(1, bin_idx_t.unsqueeze(0).expand(flat.shape[0], -1), flat)
        out = out / bin_counts  # [B*8, n_bins]
        out = out.view(len(batch_ids), N_FRAMES, n_bins).mean(dim=1)  # [B, n_bins]
        feats[i:i + out.shape[0]] = out.cpu().numpy()
        if (i // FFT_BATCH_VIDEOS) % 25 == 0:
            done = i + len(batch_ids)
            rate = done / (time.time() - t0 + 1e-6)
            log.info("  FFT %d/%d videos  (%.1f vid/s)", done, len(ids), rate)
    np.save(cache, feats)
    cache_ids.write_text(json.dumps(ids))
    log.info("FFT features saved: %s", cache)
    return feats


# ---------------------------- clustering ----------------------------


def cluster_track(name: str, X: np.ndarray, gen_labels: list[str | None]) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, normalized_mutual_info_score
    from sklearn.preprocessing import StandardScaler

    Xn = StandardScaler().fit_transform(X) if name == "fft" else X.copy()

    labeled_mask = np.array([g is not None for g in gen_labels])
    n_known = len(set(g for g in gen_labels if g is not None))
    log.info("[%s] N=%d, labeled=%d, distinct generators=%d",
             name, len(X), labeled_mask.sum(), n_known)

    sweep = {}
    for k in range(3, 13):
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = km.fit_predict(Xn)
        sample_idx = np.random.RandomState(0).choice(
            len(X), size=min(5000, len(X)), replace=False
        )
        sil = silhouette_score(Xn[sample_idx], labels[sample_idx])
        sweep[k] = {
            "silhouette": float(sil),
            "inertia": float(km.inertia_),
            "cluster_sizes": np.bincount(labels).tolist(),
        }
        log.info("  k=%2d  silhouette=%.3f  inertia=%.0f  sizes=%s",
                 k, sil, km.inertia_, np.bincount(labels).tolist())

    best_k = max(sweep, key=lambda k: sweep[k]["silhouette"])
    log.info("[%s] best k by silhouette = %d", name, best_k)

    km = KMeans(n_clusters=best_k, random_state=0, n_init=10)
    labels = km.fit_predict(Xn)

    if labeled_mask.sum() > 50:
        gen_str = np.array([g if g else "" for g in gen_labels])
        sub_labels = labels[labeled_mask]
        sub_gens = gen_str[labeled_mask]
        purity = 0.0
        for c in np.unique(sub_labels):
            mask_c = sub_labels == c
            if mask_c.sum() == 0:
                continue
            top = np.unique(sub_gens[mask_c], return_counts=True)
            purity += top[1].max()
        purity /= len(sub_labels)
        nmi = normalized_mutual_info_score(sub_gens, sub_labels)
    else:
        purity, nmi = float("nan"), float("nan")

    sanity_purity = float("nan")
    if labeled_mask.sum() > 50 and n_known >= 2:
        Xl = Xn[labeled_mask]
        gl = np.array([g for g in gen_labels if g is not None])
        km2 = KMeans(n_clusters=min(n_known, len(Xl) - 1), random_state=0, n_init=10)
        ll = km2.fit_predict(Xl)
        s = 0.0
        for c in np.unique(ll):
            mask_c = ll == c
            if mask_c.sum() == 0:
                continue
            top = np.unique(gl[mask_c], return_counts=True)
            s += top[1].max()
        sanity_purity = float(s / len(Xl))

    return {
        "track": name,
        "n": int(len(X)),
        "n_labeled": int(labeled_mask.sum()),
        "n_known_generators": int(n_known),
        "best_k": int(best_k),
        "best_silhouette": float(sweep[best_k]["silhouette"]),
        "best_cluster_sizes": sweep[best_k]["cluster_sizes"],
        "purity_labeled": float(purity),
        "nmi_labeled": float(nmi),
        "sanity_purity_k_eq_known": sanity_purity,
        "k_sweep": sweep,
        "labels": labels.tolist(),
    }


def cross_track_ari(labels_a: list[int], labels_b: list[int]) -> float:
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(labels_a, labels_b))


# ---------------------------- visualization ----------------------------


def render_figure(X_clip, X_fft, ids, gens, plats, clip_labels, fft_labels, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    log.info("running t-SNE")
    sub = np.random.RandomState(0).choice(len(ids), size=min(5000, len(ids)), replace=False)
    Xc = X_clip[sub]
    Xf = StandardScaler().fit_transform(X_fft)[sub]
    tsne_clip = TSNE(n_components=2, random_state=0, perplexity=30, init="pca").fit_transform(Xc)
    tsne_fft = TSNE(n_components=2, random_state=0, perplexity=30, init="pca").fit_transform(Xf)

    sub_gens = [gens[i] for i in sub]
    sub_plats = [plats[i] for i in sub]
    sub_clip_lab = np.array(clip_labels)[sub]
    sub_fft_lab = np.array(fft_labels)[sub]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for row, (tsne, lab, name) in enumerate([
        (tsne_clip, sub_clip_lab, "CLIP"),
        (tsne_fft, sub_fft_lab, "FFT"),
    ]):
        ax = axes[row, 0]
        ax.scatter(tsne[:, 0], tsne[:, 1], c=lab, cmap="tab10", s=3, alpha=0.6)
        ax.set_title(f"{name} — by cluster (k={len(set(lab))})")
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[row, 1]
        gen_set = sorted({g for g in sub_gens if g})
        gen_to_idx = {g: i for i, g in enumerate(gen_set)}
        colors = np.array([gen_to_idx.get(g, -1) for g in sub_gens])
        m = colors >= 0
        ax.scatter(tsne[~m, 0], tsne[~m, 1], c="lightgrey", s=2, alpha=0.3, label="unknown")
        ax.scatter(tsne[m, 0], tsne[m, 1], c=colors[m], cmap="tab20", s=4, alpha=0.85)
        ax.set_title(f"{name} — by labeled generator")
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[row, 2]
        plat_set = sorted(set(sub_plats))
        plat_to_idx = {p: i for i, p in enumerate(plat_set)}
        pc = np.array([plat_to_idx[p] for p in sub_plats])
        ax.scatter(tsne[:, 0], tsne[:, 1], c=pc, cmap="Set2", s=3, alpha=0.6)
        handles = [plt.Line2D([], [], marker='o', linestyle='', label=p,
                              markerfacecolor=plt.cm.Set2(i / max(1, len(plat_set) - 1)))
                   for p, i in plat_to_idx.items()]
        ax.legend(handles=handles, fontsize=8, loc="upper right")
        ax.set_title(f"{name} — by platform")
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    log.info("figure saved: %s", out_path)


# ---------------------------- main ----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["extract", "cluster", "all"], default="all")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ids, gens, plats = load_index()
    log.info("indexed %d locally-available fakes", len(ids))
    log.info("  unknown: %d, labeled: %d",
             sum(1 for g in gens if not g), sum(1 for g in gens if g))

    if args.phase in ("extract", "all"):
        extract_frames_parallel(ids, workers=args.workers)

    keep = filter_ids_with_frames(ids)
    keep_set = set(keep)
    gens = [g for v, g in zip(ids, gens) if v in keep_set]
    plats = [p for v, p in zip(ids, plats) if v in keep_set]
    ids = keep

    (CACHE_DIR / "meta.json").write_text(
        json.dumps({"ids": ids, "gens": gens, "plats": plats})
    )

    if args.phase in ("extract", "all"):
        compute_clip_features(ids)
        radial_fft_spectrum_batch(ids)

    if args.phase in ("cluster", "all"):
        clip_path = CACHE_DIR / "clip_shuffled_features.npy"
        if not clip_path.exists():
            clip_path = CACHE_DIR / "clip_features.npy"
        X_clip = np.load(clip_path)
        X_fft = np.load(CACHE_DIR / "fft_features.npy")
        log.info("clip features: %s shape=%s", clip_path.name, X_clip.shape)
        clip_res = cluster_track("clip", X_clip, gens)
        fft_res = cluster_track("fft", X_fft, gens)
        ari = cross_track_ari(clip_res["labels"], fft_res["labels"])
        log.info("cross-track ARI = %.3f", ari)

        out = {
            "n_videos": len(ids),
            "n_unknown": sum(1 for g in gens if not g),
            "n_labeled": sum(1 for g in gens if g),
            "n_frames_per_video": N_FRAMES,
            "clip": {k: v for k, v in clip_res.items() if k != "labels"},
            "fft": {k: v for k, v in fft_res.items() if k != "labels"},
            "ari_clip_vs_fft": ari,
        }
        (RESULTS_DIR / "cluster_unknown.json").write_text(json.dumps(out, indent=2))
        log.info("results saved: %s", RESULTS_DIR / "cluster_unknown.json")

        render_figure(
            X_clip, X_fft, ids, gens, plats,
            clip_res["labels"], fft_res["labels"],
            FIGS_DIR / "cluster_unknown.png",
        )


if __name__ == "__main__":
    main()
