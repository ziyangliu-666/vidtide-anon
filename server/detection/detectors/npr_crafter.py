"""NPR (CVPR'24) detector — VideoCrafter-trained checkpoint from NSG-VD repo."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from server.detection.base import BaseDetector
from server.detection.registry import register_detector

logger = logging.getLogger(__name__)

_NSGVD_DIR = Path(__file__).resolve().parents[3] / "vendor" / "NSG-VD"
_DEFAULT_CKPT = _NSGVD_DIR / "results" / "ckpts" / "baselines" / "standard-Crafter-npr" / "final_ckpt.pth"
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


@register_detector("npr_crafter")
class NPRCrafter(BaseDetector):
    name = "npr_crafter"
    expects_frames = 8

    def __init__(self, device: str = "cuda", ckpt_path: str | None = None, **kwargs) -> None:
        if str(_NSGVD_DIR) not in sys.path:
            sys.path.insert(0, str(_NSGVD_DIR))
        from models.npr import resnet50_npr

        self.device = device if torch.cuda.is_available() else "cpu"
        ckpt = Path(ckpt_path) if ckpt_path else _DEFAULT_CKPT
        if not ckpt.exists():
            raise FileNotFoundError(f"NPR checkpoint not found: {ckpt}")

        logger.info("NPRCrafter: loading %s on %s", ckpt, self.device)
        self.model = resnet50_npr(pretrained=False).to(self.device)
        state = torch.load(str(ckpt), map_location=self.device, weights_only=False)
        if any(k.startswith("module.") for k in state.keys()):
            state = {k[len("module."):]: v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()

        self._mean = _IMAGENET_MEAN.to(self.device)
        self._std = _IMAGENET_STD.to(self.device)

    def predict(self, frames: np.ndarray) -> float:
        x = torch.from_numpy(frames).float().to(self.device) / 255.0  # (T,H,W,3)
        x = x.permute(0, 3, 1, 2).unsqueeze(0)                        # (1,T,3,H,W)
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            B, T = x.shape[0], x.shape[1]
            x = F.interpolate(
                x.view(B * T, 3, x.shape[-2], x.shape[-1]),
                size=(224, 224), mode="bilinear", align_corners=False,
            ).view(B, T, 3, 224, 224)
        x = (x - self._mean) / self._std

        with torch.no_grad():
            logit = self.model(x)            # (1, 1)
            score = logit[:, 0].sigmoid().item()
        return float(score)

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
