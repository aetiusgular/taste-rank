"""Lane P3a's own tests: invariants on the synthetic fixture and edge cases the red tests leave out."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from tasterank import data
from tasterank.seams import TRANSACTION_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "fixtures" / "shop_like"


@pytest.fixture(scope="module")
def rule() -> dict:
    return json.loads((FIX / "split.json").read_text())


@pytest.fixture(scope="module")
def tx() -> pl.DataFrame:
    return data.load_transactions(FIX / "transactions_train.csv")


def test_schema_matches_contract(tx):
    assert {k: str(v) for k, v in tx.schema.items()} == TRANSACTION_COLUMNS


def test_lazy_scan_equals_eager(tx):
    assert data.scan_transactions(FIX / "transactions_train.csv").collect().equals(tx)


def test_bad_date_refused(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("t_dat,customer_id,article_id,price,sales_channel_id\n2020/09/01,c,0000000001,0.1,1\n")
    with pytest.raises(pl.exceptions.PolarsError):
        data.load_transactions(p)


def test_articles_missing_group_refused(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("article_id,prod_name\n0000000001,x\n")
    with pytest.raises(ValueError, match="product_group_name"):
        data.load_articles(p)


def test_anchor_min_and_empty(tx, rule):
    with pytest.raises(ValueError, match="min t_dat"):
        data.check_anchor(tx.filter(pl.col("t_dat") > dt.date(2020, 5, 1)), rule)
    with pytest.raises(ValueError, match="empty"):
        data.check_anchor(tx.head(0), rule)


def test_unknown_split_refused(tx, rule):
    with pytest.raises(ValueError, match="dev' or 'heldout"):
        data.build_split(tx, rule, "train")


@pytest.mark.parametrize("which", ["dev", "heldout"])
def test_fixture_split_invariants(tx, rule, which):
    s = data.build_split(tx, rule, which)
    assert s.customers, "the fixture fills every split"
    assert s.customers == sorted(s.customers)
    assert all(data.assign_split(c, rule) == which for c in s.customers)
    assert set(s.relevant) == set(s.customers)
    assert s.history["t_dat"].max() < s.window_start
    assert set(s.history["customer_id"].unique().to_list()) == set(s.customers)
    assert s.pool == sorted(set(s.pool))
    assert set(s.pop_counts) <= set(s.pool) and all(n > 0 for n in s.pop_counts.values())
    # pool comes only from the 84 days before the window
    lo = s.window_start - dt.timedelta(days=rule["pool_days_before_window"])
    sold = set(tx.filter((pl.col("t_dat") >= lo) & (pl.col("t_dat") < s.window_start))["article_id"].to_list())
    assert set(s.pool) == sold


def test_splits_are_disjoint(tx, rule):
    dev = set(data.build_split(tx, rule, "dev").customers)
    ho = set(data.build_split(tx, rule, "heldout").customers)
    train = {p.customer_id for p in data.iter_training_pairs(tx, rule, 5)}
    assert not (dev & ho) and not (train & dev) and not (train & ho)


def test_lazy_input_gives_same_split(tx, rule):
    a = data.build_split(tx, rule, "dev")
    b = data.build_split(tx.lazy(), rule, "dev")
    assert (a.customers, a.pool, a.pop_counts, a.relevant, a.cold_start) == (b.customers, b.pool, b.pop_counts, b.relevant, b.cold_start)
    assert a.history.equals(b.history)


def test_last_n_caps_and_orders(tx, rule):
    s = data.build_split(tx, rule, "heldout")
    for n in (1, 2, 5):
        last = data.last_n_history(s.history, n)
        assert set(last) == set(s.customers)
        for ps in last.values():
            assert 1 <= len(ps) <= n
            assert [p.t_dat for p in ps] == sorted((p.t_dat for p in ps), reverse=True)
    with pytest.raises(ValueError):
        data.last_n_history(s.history, 0)


def test_training_pairs_on_fixture(tx, rule):
    cutoff = dt.date.fromisoformat(rule["train_cutoff"])
    pairs = data.training_pairs(tx, rule, n_last=3)
    assert pairs
    assert pairs == list(data.iter_training_pairs(tx, rule, n_last=3))
    for p in pairs:
        assert data.assign_split(p.customer_id, rule) == "train"
        assert p.target_date < cutoff
        assert 1 <= len(p.history) <= 3
        assert all(h.t_dat < p.target_date for h in p.history)
        assert [h.t_dat for h in p.history] == sorted((h.t_dat for h in p.history), reverse=True)


def test_same_day_targets_share_history(rule):
    lo, hi = rule["customer_hash"]["buckets"]["train"]
    cid = next(f"t-{i}" for i in range(10_000) if lo <= data.customer_bucket(f"t-{i}", rule) <= hi)
    tx = pl.DataFrame(
        {
            "t_dat": [dt.date(2020, 8, 1), dt.date(2020, 8, 2), dt.date(2020, 8, 2), dt.date(2020, 8, 3)],
            "customer_id": [cid] * 4,
            "article_id": ["0000000001", "0000000002", "0000000003", "0000000004"],
            "price": [0.1] * 4,
            "sales_channel_id": [1] * 4,
            "row_idx": [0, 1, 2, 3],
        }
    ).cast({"sales_channel_id": pl.Int8, "row_idx": pl.UInt32})
    got = [(p.target_article, tuple(h.article_id for h in p.history)) for p in data.training_pairs(tx, rule, n_last=2)]
    assert got == [
        ("0000000002", ("0000000001",)),
        ("0000000003", ("0000000001",)),
        ("0000000004", ("0000000003", "0000000002")),
    ]


def test_split_ids_sha256_changes_with_salt(tx, rule):
    a = data.split_ids_sha256(tx, rule)
    assert len(a) == 64 and int(a, 16) >= 0
    other = json.loads(json.dumps(rule))
    other["customer_hash"]["salt"] = "another-salt"
    assert data.split_ids_sha256(tx, other) != a
    assert a != hashlib.sha256(b"").hexdigest(), "the fixture has customers in every split"


def test_bucket_out_of_ranges_refused(rule):
    bad = json.loads(json.dumps(rule))
    bad["customer_hash"]["buckets"] = {"train": [0, 0], "dev": [1, 1], "heldout": [2, 2]}
    cid = next(f"x-{i}" for i in range(10_000) if data.customer_bucket(f"x-{i}", bad) > 2)
    with pytest.raises(ValueError, match="no split range"):
        data.assign_split(cid, bad)
