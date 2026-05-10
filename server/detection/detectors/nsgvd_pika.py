"""NSG-VD (NeurIPS'25, Velocity-MMD) detector — Pika-trained MMD-MP from NSG-VD repo."""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from server.detection.base import BaseDetector
from server.detection.registry import register_detector

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[3]
_NSGVD_DIR = _REPO / "vendor" / "NSG-VD"
_DEFAULT_CKPT = _NSGVD_DIR / "ckpts" / "standard-Pika-mp.pth"
_DIFFUSION_CKPT = _REPO / "Checkpoints" / "256x256_diffusion_uncond.pt"
_REF_CACHE_DIR = _REPO / ".nsgvd_ref_cache"
_DIFFUSE_STEPS = 5


@register_detector("nsgvd_pika")
class NSGVDPika(BaseDetector):
    name = "nsgvd_pika"
    expects_frames = 8

    def __init__(
        self,
        device: str = "cuda",
        ckpt_path: str | None = None,
        ref_videos: list[str] | None = None,
        ref_n: int = 100,
        **kwargs,
    ) -> None:
        if str(_NSGVD_DIR) not in sys.path:
            sys.path.insert(0, str(_NSGVD_DIR))
        if not _DIFFUSION_CKPT.exists():
            raise FileNotFoundError(
                f"Diffusion ckpt not found: {_DIFFUSION_CKPT}\n"
                "Download via: wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
            )

        self.device = device if torch.cuda.is_available() else "cpu"
        ckpt = Path(ckpt_path) if ckpt_path else _DEFAULT_CKPT
        if not ckpt.exists():
            raise FileNotFoundError(f"NSG-VD ckpt not found: {ckpt}")

        # NSG-VD uses hardcoded '../Checkpoints/' relative path; chdir for setup.
        cwd_orig = os.getcwd()
        os.chdir(str(_NSGVD_DIR))
        try:
            from models.deep_mmd import deep_MMD
            from models.tall import SingleSwinBlockDiscriminator
            from data.utils import get_score_fn

            logger.info("NSGVDPika: loading %s on %s", ckpt, self.device)
            disc = SingleSwinBlockDiscriminator(num_features=300)
            self.model = deep_MMD(
                discriminator=disc, sigma=1000, sigma0=0.1, epsilon=10,
                img_size=224, is_yy_zero=True, is_smooth=True,
            )
            state = torch.load(str(ckpt), map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self.model = self.model.to(self.device).eval()

            # Recover frozen MMD hyperparams from the ckpt's buffers (set by training).
            self.mmd_sigma = float(self.model.sigma.item())
            self.mmd_sigma0 = float(self.model.sigma0_u.item())
            self.mmd_ep = float(self.model.ep.item())

            logger.info("NSGVDPika: loading diffusion score_fn (%s)", _DIFFUSION_CKPT.name)
            self.score_fn = get_score_fn(device=self.device, process_shape=(3, 224, 224))
        finally:
            os.chdir(cwd_orig)

        # Build/load reference set
        if ref_videos is None:
            ref_videos = self._discover_ref_videos(ref_n)
        self._build_ref_features(ref_videos[:ref_n])

    @staticmethod
    def _discover_ref_videos(ref_n: int) -> list[str]:
        """Reserve the LAST ref_n sorted real videos as MMD reference set.

        Bench scripts shuffle from the full real pool with a small seed; reserving
        from the tail by sorted name minimizes overlap risk for small test sizes.
        """
        real_dir = _REPO / "data" / "blobs" / "videos" / "real"
        all_reals = sorted(real_dir.glob("*.mp4"))
        if len(all_reals) < ref_n:
            raise RuntimeError(f"Only {len(all_reals)} reals; need ≥{ref_n} for ref")
        return [str(p) for p in all_reals[-ref_n:]]

    def _extract_velocity(self, frames: np.ndarray) -> torch.Tensor:
        """frames (T,H,W,3) uint8 -> velocity (1, T, 3, 224, 224)."""
        x = torch.from_numpy(frames).float().to(self.device) / 255.0  # (T,H,W,3)
        x = x.permute(0, 3, 1, 2)  # (T, 3, H, W)
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        T = x.shape[0]
        x = x.unsqueeze(0)  # (1, T, 3, 224, 224)

        # Diffusion score at small noise level t ≈ 6/1000.
        t_value = (_DIFFUSE_STEPS + 1) / 1000
        curr_t = torch.tensor(t_value, device=self.device).expand(T)
        flat = x.view(T, 3, 224, 224)
        with torch.no_grad():
            score_full = self.score_fn(2 * flat - 1, curr_t)  # (T, 6, H, W) for VPSDE
            score, _ = torch.split(score_full, score_full.shape[1] // 2, dim=1)
            if score.shape[-1] != 224 or score.shape[-2] != 224:
                score = F.interpolate(score, size=(224, 224), mode="bilinear", align_corners=False)
        score = score.view(1, T, 3, 224, 224)

        # Time-difference of pixel intensities (velocity field denominator).
        image = x  # (1, T, 3, 224, 224)
        pixel_diff = torch.zeros_like(image)
        for t in range(1, T - 1):
            pixel_diff[:, t] = (image[:, t + 1] - image[:, t - 1]) / 2
        pixel_diff[:, 0] = image[:, 1] - image[:, 0]
        pixel_diff[:, -1] = image[:, -1] - image[:, -2]

        eps = 1e-10
        denom = (score * pixel_diff).sum(dim=(2, 3, 4)).view(1, T, 1, 1, 1) + eps
        velocity = score / denom  # (1, T, 3, 224, 224)
        return velocity

    def _build_ref_features(self, ref_videos: list[str]) -> None:
        from server.detection.dataset import extract_frames

        cache_key = hashlib.md5(
            ("|".join(sorted(ref_videos))).encode()
        ).hexdigest()[:12]
        _REF_CACHE_DIR.mkdir(exist_ok=True)
        cache_file = _REF_CACHE_DIR / f"ref_pika_{cache_key}_n{len(ref_videos)}.pt"

        if cache_file.exists():
            blob = torch.load(cache_file, map_location=self.device, weights_only=False)
            self.feature_ref = blob["feature_ref"].to(self.device)
            self.ref_data = blob["ref_data"].to(self.device)
            logger.info(
                "NSGVDPika: loaded cached ref features from %s (N=%d)",
                cache_file.name, self.feature_ref.shape[0],
            )
            return

        logger.info("NSGVDPika: extracting ref features for %d videos", len(ref_videos))
        velocities = []
        for i, vp in enumerate(ref_videos):
            try:
                frames = extract_frames(Path(vp), num_frames=8, resolution=224)
                v = self._extract_velocity(frames)
                velocities.append(v.cpu())
            except Exception as e:
                logger.warning("ref skip %s: %s", Path(vp).name, e)
            if (i + 1) % 20 == 0:
                logger.info("  ref [%d/%d]", i + 1, len(ref_videos))
        if not velocities:
            raise RuntimeError("Failed to extract any ref features")

        ref_data = torch.cat(velocities, dim=0).to(self.device)
        with torch.no_grad():
            _, feature_ref = self.model.net(ref_data, out_feature=True)
        self.feature_ref = feature_ref.detach()
        self.ref_data = ref_data.detach()
        torch.save(
            {"feature_ref": self.feature_ref.cpu(), "ref_data": self.ref_data.cpu()},
            cache_file,
        )
        logger.info(
            "NSGVDPika: cached ref features to %s (N=%d)",
            cache_file.name, self.feature_ref.shape[0],
        )

    def predict(self, frames: np.ndarray) -> float:
        from utils.mmd_utils import MMD_batch2

        velocity = self._extract_velocity(frames)  # (1, T, 3, 224, 224)
        with torch.no_grad():
            _, feat = self.model.net(velocity, out_feature=True)  # (1, 300)
            n_ref = self.feature_ref.shape[0]
            Fea = torch.cat([self.feature_ref, feat], dim=0)  # (n_ref+1, 300)
            Fea_org = torch.cat([self.ref_data, velocity], dim=0).view(n_ref + 1, -1)
            mmd2 = MMD_batch2(
                Fea, n_ref, Fea_org,
                self.mmd_sigma, self.mmd_sigma0, self.mmd_ep, is_smooth=True,
            )
        # mmd2: (1,) — higher = more fake (per test_dMMD logic).
        return float(mmd2[0].item())

    def close(self) -> None:
        del self.model
        del self.score_fn
        if hasattr(self, "feature_ref"):
            del self.feature_ref
        if hasattr(self, "ref_data"):
            del self.ref_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
