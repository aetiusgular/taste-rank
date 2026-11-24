"""Train lane (P3b): customer vectors, InfoNCE, LoRA, the training loop.

Signatures are frozen in contracts/seams.md; the rules are in contracts/eval.md
("Customer vector and score", "Training pairs (E1)"). Everything here runs on
CPU against tasterank.tiny in the tests; the real encoders run on the box (P5).
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from tasterank.seams import REPO_ROOT, ImageSource, TrainConfig, TrainPair, TrainResult

CHECKPOINT_NAME = "lora.pt"
META_NAME = "lora.json"


def set_seed(seed: int) -> None:
    """Seeds Python, numpy and torch, and asks torch for deterministic algorithms."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)


def _weights(ages_days: torch.Tensor, half_life_days: float | None) -> torch.Tensor:
    if half_life_days is None or math.isinf(half_life_days):
        return torch.ones_like(ages_days)
    if not half_life_days > 0:  # also refuses NaN
        raise ValueError(f"half_life_days must be positive or inf, got {half_life_days}")
    return torch.pow(torch.full_like(ages_days, 0.5), ages_days / half_life_days)


def customer_vector(embeddings: Any, ages_days: Any, half_life_days: float) -> Any:
    """Recency-weighted mean of L2-normalized embeddings, L2-normalized (eval.md "Customer vector").

    embeddings [n, d], ages_days [n] (days before the window start or target day);
    half_life_days math.inf (or None) gives equal weights. Returns a unit tensor [d].
    """
    e = torch.as_tensor(embeddings)
    if e.ndim != 2 or e.shape[0] == 0:
        raise ValueError(f"customer_vector needs embeddings [n>=1, d], got shape {tuple(e.shape)}")
    ages = torch.as_tensor(ages_days, dtype=e.dtype, device=e.device).reshape(-1)
    if ages.shape[0] != e.shape[0]:
        raise ValueError(f"{e.shape[0]} embeddings but {ages.shape[0]} ages")
    if bool((ages < 0).any()):
        raise ValueError("ages_days must be >= 0 (history is before the reference day)")
    w = _weights(ages, half_life_days)
    v = (F.normalize(e, dim=-1) * w.unsqueeze(-1)).sum(dim=0)
    return F.normalize(v, dim=-1)


def info_nce(customer_vecs: Any, target_vecs: Any, temperature: float) -> Any:
    """In-batch InfoNCE: cross_entropy(C . T^T / temperature, arange(B)), inputs normalized here."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    c = F.normalize(torch.as_tensor(customer_vecs), dim=-1)
    t = F.normalize(torch.as_tensor(target_vecs), dim=-1)
    if c.shape != t.shape or c.ndim != 2:
        raise ValueError(f"customer and target batches must both be [B, d], got {tuple(c.shape)} and {tuple(t.shape)}")
    logits = c @ t.T / temperature
    return F.cross_entropy(logits, torch.arange(c.shape[0], device=c.device))


def _lora_config(cfg: TrainConfig):
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
    )


def apply_lora(encoder: Any, cfg: TrainConfig) -> Any:
    """Wraps the modules whose names end in cfg.target_modules with LoRA; freezes everything else.

    peft matches a target when the module name equals it or ends in "." + target,
    so "attn.proj" hits blocks.N.attn.proj and not attn_pool.proj or patch_embed.
    The encoder is modified in place (peft injects the adapters).
    """
    from peft import get_peft_model

    for p in encoder.parameters():
        p.requires_grad_(False)
    model = get_peft_model(encoder, _lora_config(cfg))
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n)
    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError(f"no module matched target_modules {cfg.target_modules}")
    return model


def _check_out_dir(out_dir: Any) -> Path:
    if out_dir is None:
        raise ValueError("cfg.out_dir is required: a box-local path outside the repo")
    p = Path(out_dir).expanduser().resolve()
    root = REPO_ROOT.resolve()
    if p == root or root in p.parents:
        raise ValueError(f"out_dir {p} is inside the repo ({root}); LoRA weights are never published, write them to a box-local path")
    return p


def _half_life_for_meta(half_life_days: float | None) -> float | None:
    """Checks cfg.half_life_days before any work; None (split.json `null`) and inf both mean equal weights."""
    if half_life_days is None:
        return None
    hl = float(half_life_days)
    if math.isinf(hl) and hl > 0:
        return None
    if not hl > 0:  # 0, negative, NaN, -inf
        raise ValueError(f"half_life_days must be positive, inf or None, got {half_life_days}")
    return hl


def _check_pairs(pairs: Sequence[TrainPair], cfg: TrainConfig) -> None:
    if not pairs:
        raise ValueError("no training pairs")
    for pr in pairs:
        if pr.target_date >= cfg.train_cutoff:
            raise ValueError(f"pair for {pr.customer_id} dated {pr.target_date} is on or after train_cutoff {cfg.train_cutoff}")
        if not pr.history:
            raise ValueError(f"pair for {pr.customer_id} on {pr.target_date} has an empty history (eval.md drops those)")
        for h in pr.history:
            if h.t_dat >= pr.target_date:
                raise ValueError(f"pair for {pr.customer_id}: history purchase {h.t_dat} is not before target day {pr.target_date}")


def _batches(pairs: Sequence[TrainPair], batch_size: int, steps: int, gen: torch.Generator) -> Iterator[list[int]]:
    """Yields `steps` batches of pair indices, each with distinct target articles.

    Walks seeded permutations; a pair whose target is already in the batch waits
    for the next batch, so in-batch negatives are never the positive itself.
    """
    n_targets = len({p.target_article for p in pairs})
    size = min(batch_size, n_targets)
    if size < 2:
        raise ValueError("in-batch InfoNCE needs at least two distinct target articles")
    queue: list[int] = []
    made = 0
    while made < steps:
        if len(queue) < 4 * size:
            queue.extend(torch.randperm(len(pairs), generator=gen).tolist())
        batch, seen, rest = [], set(), []
        for i in queue:
            t = pairs[i].target_article
            if len(batch) < size and t not in seen:
                batch.append(i)
                seen.add(t)
            else:
                rest.append(i)
        if len(batch) < size:
            # Not enough distinct targets queued: add a fresh permutation and retry.
            # One permutation holds every target, so the retry always fills.
            queue = queue + torch.randperm(len(pairs), generator=gen).tolist()
            continue
        queue = rest
        made += 1
        yield batch


def _step_loss(model: Any, batch: list[TrainPair], images: ImageSource, cfg: TrainConfig, device: torch.device) -> torch.Tensor:
    """One forward pass over every image the batch needs, then customer vectors and InfoNCE."""
    order: dict[str, int] = {}
    for pr in batch:
        for a in [pr.target_article, *(h.article_id for h in pr.history[: cfg.n_last])]:
            order.setdefault(a, len(order))
    pixels = torch.stack([torch.as_tensor(images(a), dtype=torch.float32) for a in order]).to(device)
    emb = model(pixels)
    cust, targ = [], []
    for pr in batch:
        hist = pr.history[: cfg.n_last]
        idx = torch.tensor([order[h.article_id] for h in hist], device=device)
        ages = torch.tensor([float((pr.target_date - h.t_dat).days) for h in hist], device=device)
        cust.append(customer_vector(emb[idx], ages, cfg.half_life_days))
        targ.append(emb[order[pr.target_article]])
    return info_nce(torch.stack(cust), torch.stack(targ), cfg.temperature)


def lora_state_dict(model: Any) -> dict[str, torch.Tensor]:
    """The trainable (LoRA) parameters, detached on CPU, keyed by their peft names."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def train(encoder: Any, pairs: Sequence[TrainPair], images: ImageSource, cfg: TrainConfig) -> TrainResult:
    """LoRA-fine-tunes `encoder` with in-batch InfoNCE on `pairs` (eval.md "Training pairs (E1)").

    Refuses pairs on or after cfg.train_cutoff, an out_dir inside the repo and a
    half_life_days that is not positive, inf or None, all before any work. Same seed, same weights and losses on CPU. Writes
    <out_dir>/lora.pt (the LoRA tensors) and <out_dir>/lora.json (config, losses).
    """
    out_dir = _check_out_dir(cfg.out_dir)
    pairs = list(pairs)
    _check_pairs(pairs, cfg)
    half_life_meta = _half_life_for_meta(cfg.half_life_days)
    if cfg.steps < 1 or cfg.batch_size < 2:
        raise ValueError(f"steps must be >= 1 and batch_size >= 2, got {cfg.steps} and {cfg.batch_size}")

    set_seed(cfg.seed)
    model = apply_lora(encoder, cfg)
    device = next(model.parameters()).device
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)

    losses: list[float] = []
    for idx in _batches(pairs, cfg.batch_size, cfg.steps, gen):
        loss = _step_loss(model, [pairs[i] for i in idx], images, cfg, device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    model.eval()

    state = lora_state_dict(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / CHECKPOINT_NAME
    torch.save(state, ckpt)
    meta = {
        "seed": cfg.seed,
        "steps": cfg.steps,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "rank": cfg.rank,
        "alpha": cfg.alpha,
        "dropout": cfg.dropout,
        "temperature": cfg.temperature,
        "n_last": cfg.n_last,
        "half_life_days": half_life_meta,
        "target_modules": list(cfg.target_modules),
        "train_cutoff": cfg.train_cutoff.isoformat(),
        "n_pairs": len(pairs),
        "losses": losses,
    }
    (out_dir / META_NAME).write_text(json.dumps(meta, indent=1) + "\n")
    return TrainResult(losses=losses, checkpoint=ckpt, lora_state=state)


def load_lora(encoder: Any, checkpoint: Path | str, cfg: TrainConfig) -> Any:
    """Rebuilds a trained encoder: apply_lora with cfg, then the saved LoRA tensors (strict)."""
    model = apply_lora(encoder, cfg)
    state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    own = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(state) != set(own):
        raise ValueError(f"checkpoint keys do not match the LoRA layout: {sorted(set(state) ^ set(own))[:4]}")
    with torch.no_grad():
        for n, p in own.items():
            p.copy_(state[n].to(p.device, p.dtype))
    return model.eval()
