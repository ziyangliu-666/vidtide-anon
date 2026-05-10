"""Thumbnail image-to-text captioner(s) for dedup.

Three implementations:

1. **BlipCaptioner** — Salesforce BLIP image-captioning-base (~400MB,
   ~1s per image on GPU). Default. Chosen over Moondream2 because BLIP
   is a first-class `transformers` model with no dynamic-module imports
   and no system-level deps (Moondream2's latest revisions require the
   libvips system library via pyvips). Interface is pluggable so the
   user can switch back once libvips is installed and they add a
   MoondreamCaptioner subclass.

2. **StubCaptioner** — returns a deterministic hash-based pseudo-caption.
   Zero dependencies beyond stdlib. Used by CI and GPU-less development
   so the pipeline's shape is testable without downloading any model.

All implementations satisfy `BaseCaptioner.caption(image_bytes: bytes) -> str`.
The output string is the only contract — downstream the embedder turns
it into a 384-dim vector.

Why not OpenAI / cloud vision APIs: cost scales with crawl rate (30k
videos/year → ~$30/year at $0.001/image) and privacy leaks every
thumbnail to a third party. Local captioners are small enough that
thumbnail decode dominates inference time.
"""

from __future__ import annotations

import hashlib
import io
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseCaptioner(ABC):
    """Contract: take JPEG/PNG bytes, return a caption string."""

    # Identifier stamped into Video.caption_model so we can refuse to
    # compare embeddings across captioner generations. Override in
    # subclasses.
    name: str = "base"

    @abstractmethod
    def caption(self, image_bytes: bytes) -> str:
        ...

    def caption_batch(self, image_bytes_list: list[bytes]) -> list[str]:
        """Default: loop. Subclasses can override for real batching."""
        return [self.caption(b) for b in image_bytes_list]


# ---------------------------------------------------------------------------
# Stub — no model, deterministic output. Used by CI / GPU-less dev.
# ---------------------------------------------------------------------------


class StubCaptioner(BaseCaptioner):
    """Returns a deterministic pseudo-caption derived from the bytes hash.

    Two identical images produce identical captions, so the downstream
    embedding + KNN pipeline still exercises its real code path. Different
    images produce different captions, so no false duplicates will ever
    appear. This is the point: it lets the pipeline shape be tested
    without loading a 4GB model.
    """

    name = "stub-v1"

    def caption(self, image_bytes: bytes) -> str:
        digest = hashlib.sha1(image_bytes).hexdigest()[:12]
        return f"stub caption for image {digest}"


# ---------------------------------------------------------------------------
# BLIP — default local captioner
# ---------------------------------------------------------------------------


class BlipCaptioner(BaseCaptioner):
    """Salesforce BLIP-base image captioning.

    Uses `Salesforce/blip-image-captioning-base` (~400MB). First-class
    transformers model: AutoProcessor + BlipForConditionalGeneration,
    no dynamic module loading, no pyvips / einops / other transitive
    system deps. Runs in ~1s per image on a 3090 (less if batched).

    Lazy-loads the model on first `caption()` call. Construction itself
    is cheap so callers can instantiate at the top of a dedup batch
    and pay the ~5-10s model load cost only when there's actual work.

    Model/preprocessor pinning: transformers defaults to the latest
    snapshot; for reproducibility we pin a known-good revision. Bumping
    it is a captioner-generation change and should be paired with
    re-captioning existing rows (see scripts/recompute_dedup.py, follow-
    up to commit 4).
    """

    name = "blip-base-20250101"

    def __init__(
        self,
        model_id: str = "Salesforce/blip-image-captioning-base",
        revision: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 32,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = device  # None -> auto (cuda if available)
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, BlipForConditionalGeneration

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("BlipCaptioner: loading %s on %s", self.model_id, device)
        load_kwargs: dict = {}
        if self.revision:
            load_kwargs["revision"] = self.revision
        processor = AutoProcessor.from_pretrained(self.model_id, **load_kwargs)
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = BlipForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=dtype, **load_kwargs
        ).to(device)
        model.eval()
        self._model = model
        self._processor = processor
        self._device = device
        logger.info("BlipCaptioner: loaded")

    def caption(self, image_bytes: bytes) -> str:
        from PIL import Image
        import torch

        self._ensure_loaded()
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = self._processor(img, return_tensors="pt").to(self._device)
            if self._device == "cuda":
                inputs = {k: v.to(dtype=torch.float16) if v.dtype == torch.float32 else v
                          for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            text = self._processor.decode(out[0], skip_special_tokens=True)
            return text.strip()
        except Exception as exc:
            logger.warning("BlipCaptioner: caption failed: %s", exc)
            return ""

    def caption_batch(self, image_bytes_list: list[bytes]) -> list[str]:
        from PIL import Image
        import torch

        self._ensure_loaded()
        if not image_bytes_list:
            return []
        try:
            images = [Image.open(io.BytesIO(b)).convert("RGB") for b in image_bytes_list]
            inputs = self._processor(images, return_tensors="pt", padding=True).to(self._device)
            if self._device == "cuda":
                inputs = {k: v.to(dtype=torch.float16) if v.dtype == torch.float32 else v
                          for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            return [self._processor.decode(tok, skip_special_tokens=True).strip() for tok in out]
        except Exception as exc:
            logger.warning("BlipCaptioner: batch caption failed, falling back to loop: %s", exc)
            return super().caption_batch(image_bytes_list)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_captioner(name: str = "blip") -> BaseCaptioner:
    """Return a captioner by config name. Keeps the pipeline YAML simple."""
    if name == "stub":
        return StubCaptioner()
    if name in ("blip", "blip-base"):
        return BlipCaptioner()
    raise ValueError(
        f"unknown captioner: {name!r} (supported: 'blip', 'stub')"
    )
