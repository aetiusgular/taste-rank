"""Lane P3b tests beyond the red tests: edge cases, refusals, batching, checkpoint round trip.

CPU only, against tasterank.tiny and the synthetic fixture images.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from tasterank import train
from tasterank.seams import REPO_ROOT, Purchase, TrainConfig, TrainPair
from tasterank.tiny import tiny_encoder

FIX = REPO_ROOT / "fixtures" / "shop_like"
CUTOFF = dt.date.fromisoformat(json.loads((FIX / "split.json").read_text())["train_cutoff"])


def _cfg(out_dir, **kw) -> TrainConfig:
    base = dict(out_dir=out_dir, train_cutoff=CUTOFF, seed=0, steps=3, batch_size=4, lr=1e-3, rank=4, alpha=8.0, n_last=5, half_life_days=30.0, temperature=0.1)
    base.update(kw)
    return TrainConfig(**base)


def _images():
    import numpy as np
    from PIL import Image

    def load(article_id: str):
        arr = np.asarray(Image.open(FIX / "images" / article_id[:3] / f"{article_id}.jpg").convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    return load


def _articles() -> list[str]:
    return sorted({r["article_id"] for r in csv.DictReader((FIX / "articles.csv").open())})


def _pairs(n: int = 12) -> list[TrainPair]:
    arts = _articles()
    day = CUTOFF - dt.timedelta(days=10)
    out = []
    for i in range(n):
        hist = (Purchase(arts[(i + 1) % len(arts)], day - dt.timedelta(days=1)), Purchase(arts[(i + 2) % len(arts)], day - dt.timedelta(days=5)))
        out.append(TrainPair(f"c{i:03d}", arts[i % len(arts)], day, hist))
    return out


# --------------------------------------------------------------- customer_vector


def test_customer_vector_none_half_life_is_equal_weights():
    e = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    a = train.customer_vector(e, torch.tensor([0.0, 999.0]), None)
    b = train.customer_vector(e, torch.tensor([0.0, 999.0]), math.inf)
    assert torch.allclose(a, b) and torch.allclose(a, torch.tensor([1.0, 1.0]) / math.sqrt(2))


def test_customer_vector_unit_norm_and_scale_free():
    g = torch.Generator().manual_seed(0)
    e = torch.randn(7, 16, generator=g)
    ages = torch.arange(7, dtype=torch.float32) * 3
    v = train.customer_vector(e, ages, 30.0)
    assert v.shape == (16,) and abs(float(v.norm()) - 1.0) < 1e-6
    scaled = e * torch.tensor([[1.0], [10.0], [0.1], [3.0], [1.0], [7.0], [2.0]])
    assert torch.allclose(train.customer_vector(scaled, ages, 30.0), v, atol=1e-6)


def test_customer_vector_refuses_bad_input():
    with pytest.raises(ValueError):
        train.customer_vector(torch.zeros(0, 4), torch.zeros(0), 30.0)
    with pytest.raises(ValueError):
        train.customer_vector(torch.ones(2, 4), torch.zeros(3), 30.0)
    with pytest.raises(ValueError):
        train.customer_vector(torch.ones(1, 4), torch.tensor([-1.0]), 30.0)
    with pytest.raises(ValueError):
        train.customer_vector(torch.ones(1, 4), torch.tensor([1.0]), 0.0)


def test_customer_vector_has_gradient():
    e = torch.randn(3, 4, requires_grad=True)
    train.customer_vector(e, torch.tensor([0.0, 1.0, 2.0]), 30.0).sum().backward()
    assert e.grad is not None and bool(e.grad.abs().sum() > 0)


# --------------------------------------------------------------------- info_nce


def test_info_nce_normalizes_inputs():
    eye = torch.eye(3)
    assert torch.allclose(train.info_nce(eye * 5, eye * 0.2, 0.5), train.info_nce(eye, eye, 0.5))


def test_info_nce_matches_formula_on_random_batch():
    g = torch.Generator().manual_seed(3)
    c, t = torch.randn(6, 8, generator=g), torch.randn(6, 8, generator=g)
    cn, tn = c / c.norm(dim=1, keepdim=True), t / t.norm(dim=1, keepdim=True)
    logits = cn @ tn.T / 0.07
    want = -(logits.diag() - logits.logsumexp(dim=1)).mean()
    assert torch.allclose(train.info_nce(c, t, 0.07), want, atol=1e-5)


def test_info_nce_refuses_bad_input():
    with pytest.raises(ValueError):
        train.info_nce(torch.eye(2), torch.eye(2), 0.0)
    with pytest.raises(ValueError):
        train.info_nce(torch.eye(2), torch.eye(3), 1.0)


# ------------------------------------------------------------------- apply_lora


class _Pool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)


class _WithPool(nn.Module):
    """Mimics timm's SigLIP trunk: an attn_pool.proj that must not be wrapped."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = tiny_encoder()
        self.attn_pool = _Pool()

    def forward(self, x):
        return self.inner(x)


def test_lora_skips_attn_pool_and_patch_embed():
    model = train.apply_lora(_WithPool(), _cfg(None))
    wrapped = {n.split(".lora_")[0] for n, p in model.named_parameters() if p.requires_grad}
    assert not any("attn_pool" in n or "patch_embed" in n or "mlp" in n for n in wrapped)
    assert len(wrapped) == 4


def test_lora_starts_as_identity():
    x = torch.rand(2, 3, 16, 16)
    with torch.no_grad():
        before = tiny_encoder()(x)
        after = train.apply_lora(tiny_encoder(), _cfg(None))(x)
    assert torch.allclose(before, after, atol=1e-6)  # lora_B starts at zero


def test_lora_refuses_no_match():
    with pytest.raises(ValueError):
        train.apply_lora(tiny_encoder(), _cfg(None, target_modules=("nothing.here",)))


# ---------------------------------------------------------------------- refusals


def test_refuses_repo_root_and_symlink_into_repo(tmp_path):
    with pytest.raises(ValueError, match="inside the repo"):
        train.train(tiny_encoder(), _pairs(), _images(), _cfg(REPO_ROOT))
    link = tmp_path / "sneaky"
    link.symlink_to(REPO_ROOT / "results")
    with pytest.raises(ValueError, match="inside the repo"):
        train.train(tiny_encoder(), _pairs(), _images(), _cfg(link / "ckpt"))
    assert not (REPO_ROOT / "results" / "ckpt").exists()


def test_refuses_missing_out_dir():
    with pytest.raises(ValueError, match="out_dir"):
        train.train(tiny_encoder(), _pairs(), _images(), _cfg(None))


def test_refusal_writes_nothing(tmp_path):
    late = dataclasses.replace(_pairs()[0], target_date=CUTOFF + dt.timedelta(days=3))
    with pytest.raises(ValueError, match="train_cutoff"):
        train.train(tiny_encoder(), _pairs() + [late], _images(), _cfg(tmp_path / "x"))
    assert not (tmp_path / "x").exists()


def test_refuses_empty_pairs_empty_history_and_history_not_before_target(tmp_path):
    with pytest.raises(ValueError, match="no training pairs"):
        train.train(tiny_encoder(), [], _images(), _cfg(tmp_path / "a"))
    p = _pairs()[0]
    with pytest.raises(ValueError, match="empty history"):
        train.train(tiny_encoder(), [dataclasses.replace(p, history=())] + _pairs(), _images(), _cfg(tmp_path / "b"))
    bad = dataclasses.replace(p, history=(Purchase(p.history[0].article_id, p.target_date),))
    with pytest.raises(ValueError, match="not before target day"):
        train.train(tiny_encoder(), [bad] + _pairs(), _images(), _cfg(tmp_path / "c"))


def test_refuses_single_target(tmp_path):
    p = _pairs()[0]
    same = [dataclasses.replace(p, customer_id=f"c{i}") for i in range(4)]
    with pytest.raises(ValueError, match="two distinct target"):
        train.train(tiny_encoder(), same, _images(), _cfg(tmp_path / "s"))


# ---------------------------------------------------------------------- batching


def test_batches_have_distinct_targets_and_are_seeded():
    pairs = _pairs(12)
    pairs += [dataclasses.replace(p, customer_id=p.customer_id + "x") for p in pairs[:6]]  # repeated targets
    a = list(train._batches(pairs, 5, 20, torch.Generator().manual_seed(7)))
    b = list(train._batches(pairs, 5, 20, torch.Generator().manual_seed(7)))
    assert a == b and len(a) == 20
    for batch in a:
        assert len(batch) == 5 and len({pairs[i].target_article for i in batch}) == 5


def test_batch_size_caps_at_distinct_targets():
    pairs = _pairs(3)
    assert all(len(b) == 3 for b in train._batches(pairs, 8, 4, torch.Generator().manual_seed(0)))


def test_n_last_caps_history_used(tmp_path):
    p = _pairs(8)
    long = [dataclasses.replace(x, history=x.history + (Purchase("0000000000", x.target_date - dt.timedelta(days=50)),)) for x in p]
    # article 0000000000 has no image; n_last 2 must never ask for it
    r = train.train(tiny_encoder(), long, _images(), _cfg(tmp_path / "n", n_last=2, steps=2))
    assert len(r.losses) == 2


# -------------------------------------------------------------- checkpoint files


def test_checkpoint_files_and_round_trip(tmp_path):
    cfg = _cfg(tmp_path / "ck", steps=5)
    r = train.train(tiny_encoder(), _pairs(), _images(), cfg)
    assert r.checkpoint == (tmp_path / "ck" / "lora.pt").resolve()
    meta = json.loads((tmp_path / "ck" / "lora.json").read_text())
    assert meta["losses"] == r.losses and meta["seed"] == 0 and meta["train_cutoff"] == CUTOFF.isoformat()
    assert sorted(p.name for p in (tmp_path / "ck").iterdir()) == ["lora.json", "lora.pt"]
    assert all(torch.isfinite(torch.tensor(r.losses)))

    loaded = train.load_lora(tiny_encoder(), r.checkpoint, cfg)
    assert set(train.lora_state_dict(loaded)) == set(r.lora_state)
    x = torch.rand(3, 3, 16, 16)
    with torch.no_grad():
        base = tiny_encoder()(x)
        tuned = loaded(x)
    assert not torch.allclose(base, tuned)  # trained weights differ from the zero-init identity


def test_load_lora_refuses_other_layout(tmp_path):
    r = train.train(tiny_encoder(), _pairs(), _images(), _cfg(tmp_path / "k", steps=1))
    with pytest.raises(ValueError, match="do not match"):
        train.load_lora(tiny_encoder(), r.checkpoint, _cfg(None, target_modules=("attn.qkv",)))


def test_infinite_half_life_trains_and_is_recorded(tmp_path):
    r = train.train(tiny_encoder(), _pairs(), _images(), _cfg(tmp_path / "inf", half_life_days=math.inf, steps=2))
    assert json.loads((tmp_path / "inf" / "lora.json").read_text())["half_life_days"] is None
    assert len(r.losses) == 2


def test_base_weights_unchanged_by_training(tmp_path):
    enc = tiny_encoder()
    base = {n: p.detach().clone() for n, p in tiny_encoder().named_parameters()}
    model_r = train.train(enc, _pairs(), _images(), _cfg(tmp_path / "f", steps=3, lr=1e-2))
    assert model_r.lora_state
    for n, p in enc.named_parameters():
        if "lora_" in n:
            continue
        key = n.replace(".base_layer", "")
        assert torch.equal(p, base[key]), n


def test_null_half_life_trains_and_is_recorded(tmp_path):
    # contracts/split.json history_grid has half_life_days null (= infinite); TR-P3b-r1-C1
    r = train.train(tiny_encoder(), _pairs(), _images(), _cfg(tmp_path / "null", half_life_days=None, steps=2))
    assert json.loads((tmp_path / "null" / "lora.json").read_text())["half_life_days"] is None
    assert len(r.losses) == 2


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), -math.inf])
def test_bad_half_life_refused_before_any_write(tmp_path, bad):
    with pytest.raises(ValueError, match="half_life_days"):
        train.train(tiny_encoder(), _pairs(), _images(), _cfg(tmp_path / "bad", half_life_days=bad))
    assert not (tmp_path / "bad").exists()
