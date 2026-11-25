"""The real image encoders and their image preparation (P5).

Both encoders load only from a folder that `tasterank.models.verify_model_dir`
has checked (models.lock): SigLIP through open_clip `local-dir:`, fashion-clip
through transformers with `local_files_only=True` and `use_safetensors=True`.
Never `trust_remote_code`, never `hf-hub:`.

Preprocessing is split in two, so the image cache can hold uint8 pixels:

- geometry (`prepare_image`): resize to the model's input size from the full
  image, PIL bicubic. SigLIP squashes to 224x224 (open_clip_config.json
  `resize_mode: squash`); fashion-clip resizes the short side to 224 and takes
  the centre 224x224 crop (preprocessor_config.json).
- normalization: inside the encoder module, so lanes keep seeing pixels in
  [0, 1] (contracts/seams.md, `Encoder`).

On CUDA the tower runs under bf16 autocast and returns float32.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from tasterank.models import model_entry, verify_model_dir

SIGLIP_ID = "Marqo/marqo-fashionSigLIP"
FCLIP_ID = "patrickjohncyh/fashion-clip"
SQUASH = "squash"
CROP = "crop"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def prepare_image(path: Path | str, size: int, geometry: str) -> np.ndarray:
    """uint8 [3, size, size] from an image file: `squash` or short-side resize plus centre `crop`."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        if geometry == SQUASH:
            im = im.resize((size, size), Image.Resampling.BICUBIC)
        elif geometry == CROP:
            w, h = im.size
            s = size / min(w, h)
            nw, nh = max(size, round(w * s)), max(size, round(h * s))
            im = im.resize((nw, nh), Image.Resampling.BICUBIC)
            left, top = (nw - size) // 2, (nh - size) // 2
            im = im.crop((left, top, left + size, top + size))
        else:
            raise ValueError(f"geometry must be {SQUASH!r} or {CROP!r}, got {geometry!r}")
        arr = np.asarray(im, dtype=np.uint8)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


class NormalizedTower(nn.Module):
    """pixels [B, 3, S, S] in [0, 1] -> (x - mean) / std -> tower -> float32 embeddings [B, D]."""

    def __init__(self, tower: nn.Module, image_size: int, mean, std, geometry: str, autocast: bool = True):
        super().__init__()
        self.tower = tower
        self.image_size = int(image_size)
        self.geometry = geometry
        self.autocast = autocast
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        x = (pixels.to(self.mean.device, torch.float32) - self.mean) / self.std
        if self.autocast and x.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.tower(x)
        else:
            out = self.tower(x)
        return out.float()

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        fn = getattr(self.tower, "set_grad_checkpointing", None)
        if fn is not None:
            fn(enable)


class _ClipVision(nn.Module):
    """transformers CLIP vision tower plus its projection: pixel_values -> image_embeds."""

    def __init__(self, clip: nn.Module):
        super().__init__()
        self.vision_model = clip.vision_model
        self.visual_projection = clip.visual_projection

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = self.vision_model(pixel_values=pixel_values).pooler_output
        return self.visual_projection(pooled)


def load_siglip(folder: Path | str, device: str = "cpu") -> NormalizedTower:
    """The SigLIP image tower from a verified folder (open_clip `local-dir:`)."""
    import open_clip

    entry = model_entry(SIGLIP_ID)
    verify_model_dir(folder, entry)
    model = open_clip.create_model(f"local-dir:{Path(folder).resolve()}", device="cpu")
    tower = model.visual
    cfg = getattr(tower, "preprocess_cfg", {}) or {}
    size = cfg.get("size", (224, 224))
    size = size[0] if isinstance(size, (tuple, list)) else int(size)
    mean, std = tuple(cfg.get("mean", (0.5, 0.5, 0.5))), tuple(cfg.get("std", (0.5, 0.5, 0.5)))
    if cfg.get("resize_mode", SQUASH) != SQUASH:
        raise ValueError(f"{SIGLIP_ID}: expected resize_mode squash, got {cfg.get('resize_mode')}")
    del model  # the text tower is not used
    return NormalizedTower(tower, size, mean, std, SQUASH).to(device).eval()


def load_fclip(folder: Path | str, device: str = "cpu") -> NormalizedTower:
    """The fashion-clip image tower from a verified folder (transformers, local files, safetensors only)."""
    import json

    from transformers import CLIPModel

    entry = model_entry(FCLIP_ID)
    files = verify_model_dir(folder, entry)
    pre = json.loads(Path(files["preprocessor_config.json"]).read_text())
    clip = CLIPModel.from_pretrained(str(Path(folder).resolve()), local_files_only=True, use_safetensors=True)
    size = int(pre.get("crop_size", 224) if not isinstance(pre.get("crop_size"), dict) else pre["crop_size"]["height"])
    mean, std = tuple(pre.get("image_mean", CLIP_MEAN)), tuple(pre.get("image_std", CLIP_STD))
    return NormalizedTower(_ClipVision(clip), size, mean, std, CROP).to(device).eval()


def runtime_label() -> str:
    """Library versions for the results row `runtime` field."""
    parts = [f"torch {torch.__version__}"]
    if torch.version.cuda:
        parts.append(f"cuda {torch.version.cuda}")
    for mod in ("open_clip", "transformers", "peft", "polars"):
        try:
            m = __import__(mod)
            parts.append(f"{mod} {m.__version__}")
        except Exception:  # noqa: BLE001 - a missing optional library is just not listed
            pass
    return ", ".join(parts)


def revision_of(model_id: str) -> str:
    return f"{model_id}@{model_entry(model_id)['revision']}"

