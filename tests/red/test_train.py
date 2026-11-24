"""Red tests, lane P3b (train). One test per corner case; tests/red/manifest.json lists them.

Everything runs on CPU against tasterank.tiny (a 16x16 TinyViT) and the
synthetic fixture images.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from conftest import ROOT, d, fixture_images, fixture_pairs, fixture_rule
from tasterank.seams import TrainConfig


def _torch():
    import torch

    return torch


def _cfg(out_dir, **kw) -> TrainConfig:
    base = dict(out_dir=out_dir, train_cutoff=d(fixture_rule()["train_cutoff"]), seed=0, steps=4, batch_size=4, lr=1e-3, rank=4, alpha=8.0, n_last=5, half_life_days=30.0, temperature=0.1)
    base.update(kw)
    return TrainConfig(**base)


def test_customer_vector_single_purchase():
    torch = _torch()
    from tasterank import train

    e = torch.tensor([[3.0, 4.0]])
    v = train.customer_vector(e, torch.tensor([10.0]), 30.0)
    assert torch.allclose(v, torch.tensor([0.6, 0.8]), atol=1e-6)


def test_customer_vector_recency_weights():
    torch = _torch()
    from tasterank import train

    e = torch.tensor([[2.0, 0.0], [0.0, 5.0]])  # normalized first: e1=(1,0), e2=(0,1)
    v = train.customer_vector(e, torch.tensor([0.0, 30.0]), 30.0)  # weights 1 and 0.5
    want = torch.tensor([1.0, 0.5]) / math.sqrt(1.25)
    assert torch.allclose(v, want, atol=1e-6)


def test_customer_vector_no_decay():
    torch = _torch()
    from tasterank import train

    e = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    v = train.customer_vector(e, torch.tensor([0.0, 100.0, 400.0]), math.inf)  # equal weights
    want = torch.tensor([1.0, 2.0]) / math.sqrt(5.0)
    assert torch.allclose(v, want, atol=1e-6)


def test_info_nce_hand_value():
    torch = _torch()
    from tasterank import train

    eye = torch.eye(2)
    loss = train.info_nce(eye, eye, 1.0)
    assert abs(float(loss) - math.log(1 + math.exp(-1))) < 1e-6  # 0.313262
    loss_t = train.info_nce(eye, eye, 0.5)
    assert abs(float(loss_t) - math.log(1 + math.exp(-2))) < 1e-6


def test_lora_only_attention_trainable():
    from tasterank import train
    from tasterank.tiny import tiny_encoder

    model = train.apply_lora(tiny_encoder(), _cfg(None))
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    wrapped = sorted({n.split(".lora_")[0].removeprefix("base_model.model.") for n in trainable})
    assert wrapped == ["blocks.0.attn.proj", "blocks.0.attn.qkv", "blocks.1.attn.proj", "blocks.1.attn.qkv"]


def test_seed_determinism(outside_repo):
    torch = _torch()
    from tasterank import train
    from tasterank.tiny import tiny_encoder

    pairs, imgs = fixture_pairs(), fixture_images()
    a = train.train(tiny_encoder(), pairs, imgs, _cfg(outside_repo / "a", seed=0))
    b = train.train(tiny_encoder(), pairs, imgs, _cfg(outside_repo / "b", seed=0))
    c = train.train(tiny_encoder(), pairs, imgs, _cfg(outside_repo / "c", seed=1))
    assert a.lora_state.keys() == b.lora_state.keys() and a.lora_state
    assert all(torch.equal(a.lora_state[k], b.lora_state[k]) for k in a.lora_state)
    assert a.losses == b.losses
    assert any(not torch.equal(a.lora_state[k], c.lora_state[k]) for k in a.lora_state)


def test_checkpoint_outside_repo(outside_repo):
    from tasterank import train
    from tasterank.tiny import tiny_encoder

    pairs, imgs = fixture_pairs(8), fixture_images()
    with pytest.raises(ValueError, match="inside the repo"):
        train.train(tiny_encoder(), pairs, imgs, _cfg(ROOT / "results" / "ckpt"))
    r = train.train(tiny_encoder(), pairs, imgs, _cfg(outside_repo / "ok", steps=1))
    assert r.checkpoint.exists() and outside_repo in r.checkpoint.parents


def test_refuses_pairs_after_cutoff(outside_repo):
    from tasterank import train
    from tasterank.tiny import tiny_encoder

    pairs = list(fixture_pairs(8))
    late = dataclasses.replace(pairs[0], target_date=d(fixture_rule()["train_cutoff"]))
    with pytest.raises(ValueError, match="train_cutoff"):
        train.train(tiny_encoder(), pairs + [late], fixture_images(), _cfg(outside_repo / "x", steps=1))


def test_loss_goes_down(outside_repo):
    from tasterank import train
    from tasterank.tiny import tiny_encoder

    r = train.train(tiny_encoder(), fixture_pairs(), fixture_images(), _cfg(outside_repo / "l", steps=40, lr=5e-3, batch_size=8))
    assert len(r.losses) == 40
    first, last = sum(r.losses[:5]) / 5, sum(r.losses[-5:]) / 5
    assert last < first, (first, last)
