"""Direct image → vector embedding for thumbnail-based dedup.

Default: `ClipImageEmbedder` — wraps OpenAI's CLIP ViT-B/32 to project
a JPEG/PNG thumbnail into a 512-dim unit-normalized float32 vector,
skipping the lossy image→text→vector chain (BLIP caption + MiniLM)
that the original architecture used.

Why CLIP instead of BLIP+MiniLM:
Manual testing showed the caption-based pipeline only caught byte-
identical thumbnails. Real cross-platform reuploads compress, rescale,
and extract at different timestamps — all of which shift BLIP's greedy
decode enough that MiniLM cosine similarity drops below any usable
threshold. CLIP embeds the image directly, preserving spatial structure,
and stays > 0.85 cosine similarity through JPEG quality changes
(0.93), resolution halving (0.88), and timestamp shifts (0.94), while
scoring < 0.55 against genuinely different videos. The 0.85 threshold
gives a clean gap with zero false positives on our test data.

Model: `openai/clip-vit-base-patch32` (~600MB, ~150ms/image on 3090).
Outputs 512-dim unit-normalized vectors via the visual_projection head.

Lazy-loaded: constructing an instance is cheap (~0 ms); the model only
loads on the first `embed()` call (~3-5s cold start), and stays in GPU
memory for the rest of the batch.
"""

from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseImageEmbedder(ABC):
    """Contract: take JPEG/PNG bytes, return a fixed-dim unit-length vector."""

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, image_bytes: bytes) -> list[float]:
        ...

    def embed_batch(self, image_bytes_list: list[bytes]) -> list[list[float]]:
        """Default: loop. Override for real batched inference."""
        return [self.embed(b) for b in image_bytes_list]


class ClipImageEmbedder(BaseImageEmbedder):
    """CLIP ViT-B/32 image embedding — 512-dim unit-normalized."""

    name = "clip-vit-base-patch32"
    dim = 512

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("ClipImageEmbedder: loading %s on %s", self.model_id, device)
        model = CLIPModel.from_pretrained(self.model_id).to(device).eval()
        processor = CLIPProcessor.from_pretrained(self.model_id)
        self._model = model
        self._processor = processor
        self._device = device
        logger.info("ClipImageEmbedder: loaded")

    def embed(self, image_bytes: bytes) -> list[float]:
        import torch
        from PIL import Image

        self._ensure_loaded()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        with torch.no_grad():
            inputs = self._processor(images=[img], return_tensors="pt").to(self._device)
            vision_out = self._model.vision_model(pixel_values=inputs["pixel_values"])
            feats = self._model.visual_projection(vision_out.pooler_output)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].cpu().numpy().astype("float32").tolist()

    def embed_batch(self, image_bytes_list: list[bytes]) -> list[list[float]]:
        import torch
        from PIL import Image

        if not image_bytes_list:
            return []
        self._ensure_loaded()
        images = [Image.open(io.BytesIO(b)).convert("RGB") for b in image_bytes_list]
        with torch.no_grad():
            inputs = self._processor(images=images, return_tensors="pt", padding=True).to(self._device)
            vision_out = self._model.vision_model(pixel_values=inputs["pixel_values"])
            feats = self._model.visual_projection(vision_out.pooler_output)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return [f.cpu().numpy().astype("float32").tolist() for f in feats]


def get_image_embedder(name: str = "clip") -> BaseImageEmbedder:
    if name in ("clip", "clip-vit-base-patch32"):
        return ClipImageEmbedder()
    raise ValueError(f"unknown image embedder: {name!r}")
