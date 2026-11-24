"""Helpers for the red tests (P2). Lanes do not edit tests/red/.

Inputs are built here from the contracts, never through another lane's code,
so each red test reaches the one function it is about.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "fixtures" / "shop_like"


def d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def fixture_rule() -> dict:
    return json.loads((FIX / "split.json").read_text())


def frozen_rule() -> dict:
    return json.loads((ROOT / "contracts" / "split.json").read_text())


def bucket_of(customer_id: str, rule: dict) -> int:
    """The contract's hash rule, written out independently of the data lane."""
    salt = rule["customer_hash"]["salt"]
    return int(hashlib.sha256(f"{salt}:{customer_id}".encode()).hexdigest()[:16], 16) % 100


def ids_in(split: str, rule: dict, n: int, prefix: str = "cust") -> list[str]:
    lo, hi = rule["customer_hash"]["buckets"][split]
    out, i = [], 0
    while len(out) < n:
        cid = f"{prefix}-{i:04d}"
        if lo <= bucket_of(cid, rule) <= hi:
            out.append(cid)
        i += 1
    return out


def tx_frame(rows: list[tuple[str, str, str]]):
    """rows: (t_dat, customer_id, article_id) in file order -> contract-typed polars frame."""
    import polars as pl

    return pl.DataFrame(
        {
            "t_dat": [d(r[0]) for r in rows],
            "customer_id": [r[1] for r in rows],
            "article_id": [r[2] for r in rows],
            "price": [0.01] * len(rows),
            "sales_channel_id": [1] * len(rows),
            "row_idx": list(range(len(rows))),
        },
        schema={"t_dat": pl.Date, "customer_id": pl.String, "article_id": pl.String, "price": pl.Float64, "sales_channel_id": pl.Int8, "row_idx": pl.UInt32},
    )


def fixture_images():
    """ImageSource over fixtures/shop_like/images: article_id -> float tensor [3, 16, 16] in [0, 1]."""
    import numpy as np
    import torch
    from PIL import Image

    def load(article_id: str):
        p = FIX / "images" / article_id[:3] / f"{article_id}.jpg"
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    return load


def fixture_pairs(n: int = 24):
    """Training pairs from the fixture CSV, built from the contract's rule by hand."""
    import csv

    from tasterank.seams import Purchase, TrainPair

    rule = fixture_rule()
    cutoff = d(rule["train_cutoff"])
    rows = list(csv.DictReader((FIX / "transactions_train.csv").open()))
    lo, hi = rule["customer_hash"]["buckets"]["train"]
    by_c: dict[str, list[tuple[dt.date, int, str]]] = {}
    for i, r in enumerate(rows):
        if d(r["t_dat"]) < cutoff and lo <= bucket_of(r["customer_id"], rule) <= hi:
            by_c.setdefault(r["customer_id"], []).append((d(r["t_dat"]), i, r["article_id"]))
    pairs = []
    for cid in sorted(by_c):
        purchases = sorted(by_c[cid])
        for j, (t, _, a) in enumerate(purchases):
            hist = [p for p in purchases[:j] if p[0] < t][::-1][:5]
            if hist:
                pairs.append(TrainPair(cid, a, t, tuple(Purchase(h[2], h[0]) for h in hist)))
    return pairs[:n]


@pytest.fixture
def outside_repo(tmp_path: Path) -> Path:
    assert ROOT not in tmp_path.parents
    return tmp_path
