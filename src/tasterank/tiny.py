"""A tiny ViT image tower for CPU tests (contracts/seams.md, "Tiny encoder").

Written in P2; no lane edits it. Module names follow timm's VisionTransformer
(blocks.N.attn.qkv, blocks.N.attn.proj), which is the trunk the headline base
model's open_clip config names (models.lock), so LoRA target names carry over.
The patch embedding is called patch_embed.conv so that "attn.proj" matches
only attention projections.
"""

from __future__ import annotations

import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = nn.functional.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(b, n, d))


class Mlp(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(nn.functional.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, patch: int, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).flatten(2).transpose(1, 2)


class TinyViT(nn.Module):
    """16x16 images, 4x4 patches, 2 blocks, width 32, 16-d output."""

    def __init__(self, image_size: int = 16, patch: int = 4, dim: int = 32, depth: int = 2, heads: int = 2, out_dim: int = 16) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_embed = PatchEmbed(patch, dim)
        self.pos = nn.Parameter(torch.zeros(1, (image_size // patch) ** 2, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_dim)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(pixels) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x).mean(dim=1))


def tiny_encoder(seed: int = 1234) -> TinyViT:
    """A TinyViT with weights fixed by seed (independent of the global RNG)."""
    gen_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        model = TinyViT()
    finally:
        torch.random.set_rng_state(gen_state)
    return model.eval()
