"""Red tests, lane P3a (data). One test per corner case; tests/red/manifest.json lists them."""

from __future__ import annotations

import datetime as dt
import hashlib

import pytest

from conftest import FIX, bucket_of, d, fixture_rule, frozen_rule, ids_in, tx_frame
from tasterank import data


def test_ids_keep_leading_zeros():
    tx = data.load_transactions(FIX / "transactions_train.csv")
    assert tx.columns == ["t_dat", "customer_id", "article_id", "price", "sales_channel_id", "row_idx"]
    assert str(tx.schema["t_dat"]) == "Date" and str(tx.schema["article_id"]) == "String"
    assert all(a.startswith("0") and len(a) == 10 for a in tx["article_id"].to_list())
    assert tx["row_idx"].to_list() == list(range(tx.height))
    arts = data.load_articles(FIX / "articles.csv")
    assert arts["article_id"][0] == "0100000000" and "product_group_name" in arts.columns


def test_empty_transactions_file(tmp_path):
    p = tmp_path / "transactions_train.csv"
    p.write_text("t_dat,customer_id,article_id,price,sales_channel_id\n")
    tx = data.load_transactions(p)
    assert tx.height == 0
    assert str(tx.schema["article_id"]) == "String" and str(tx.schema["t_dat"]) == "Date"


def test_bucket_follows_frozen_rule():
    rule = frozen_rule()
    for cid in ["a", "0" * 64, "customer-1", hashlib.sha256(b"x").hexdigest()]:
        assert data.customer_bucket(cid, rule) == bucket_of(cid, rule)
    lo, hi = rule["customer_hash"]["buckets"]["heldout"]
    cid = ids_in("heldout", rule, 1)[0]
    assert lo <= bucket_of(cid, rule) <= hi
    assert data.assign_split(cid, rule) == "heldout"
    assert data.assign_split(ids_in("dev", rule, 1)[0], rule) == "dev"
    assert data.assign_split(ids_in("train", rule, 1)[0], rule) == "train"


def test_anchor_mismatch_refused():
    rule = fixture_rule()
    tx = data.load_transactions(FIX / "transactions_train.csv")
    data.check_anchor(tx, rule)  # the fixture matches its rule
    short = tx.filter(tx["t_dat"] < d("2020-09-22"))
    with pytest.raises(ValueError, match="max t_dat"):
        data.check_anchor(short, rule)


def test_window_boundaries():
    rule = fixture_rule()
    (c,) = ids_in("heldout", rule, 1)
    tx = tx_frame([
        ("2020-09-01", c, "0000000001"),
        ("2020-09-15", c, "0000000002"),  # day before the heldout window: history
        ("2020-09-16", c, "0000000003"),  # first day of the window: relevant
        ("2020-09-22", c, "0000000004"),  # last day: relevant
    ])
    s = data.build_split(tx, rule, "heldout")
    assert (s.window_start, s.window_end) == (d("2020-09-16"), d("2020-09-22"))
    assert s.relevant[c] == frozenset({"0000000003", "0000000004"})
    assert sorted(s.history.filter(s.history["customer_id"] == c)["article_id"].to_list()) == ["0000000001", "0000000002"]


def test_dev_split_ignores_later_days():
    rule = fixture_rule()
    (c,) = ids_in("dev", rule, 1)
    tx = tx_frame([
        ("2020-09-01", c, "0000000001"),
        ("2020-09-10", c, "0000000002"),  # in the dev window
        ("2020-09-16", c, "0000000003"),  # heldout window: not dev history, not dev relevant
    ])
    s = data.build_split(tx, rule, "dev")
    assert s.relevant[c] == frozenset({"0000000002"})
    assert s.history["article_id"].to_list() == ["0000000001"]
    assert "0000000003" not in s.pool


def test_eligibility_and_cold_start():
    rule = fixture_rule()
    ok, cold, quiet = ids_in("heldout", rule, 3)
    (wrong_bucket,) = ids_in("dev", rule, 1)
    tx = tx_frame([
        ("2020-09-01", ok, "0000000001"),
        ("2020-09-17", ok, "0000000002"),
        ("2020-09-18", cold, "0000000002"),  # window only: cold start, not evaluated
        ("2020-09-01", quiet, "0000000001"),  # history only: not evaluated
        ("2020-09-01", wrong_bucket, "0000000001"),
        ("2020-09-17", wrong_bucket, "0000000002"),
    ])
    s = data.build_split(tx, rule, "heldout")
    assert s.customers == [ok]
    assert set(s.relevant) == {ok}
    assert s.cold_start == 1


def test_pool_has_no_leakage():
    rule = fixture_rule()
    (c,) = ids_in("heldout", rule, 1)
    start = d("2020-09-16")
    tx = tx_frame([
        ((start - dt.timedelta(days=85)).isoformat(), c, "0000000085"),  # 85 days before: out of pool
        ((start - dt.timedelta(days=84)).isoformat(), c, "0000000084"),  # 84 days before: in pool
        ("2020-09-15", c, "0000000001"),
        ("2020-09-17", c, "0000000099"),  # sold only in the window: never in the pool
    ])
    s = data.build_split(tx, rule, "heldout")
    assert s.pool == ["0000000001", "0000000084"]
    assert "0000000099" in s.relevant[c]  # still counts as relevant (a miss)


def test_relevant_dedups_repeats():
    rule = fixture_rule()
    (c,) = ids_in("heldout", rule, 1)
    tx = tx_frame([
        ("2020-09-01", c, "0000000001"),
        ("2020-09-17", c, "0000000002"),
        ("2020-09-17", c, "0000000002"),
        ("2020-09-19", c, "0000000002"),
    ])
    s = data.build_split(tx, rule, "heldout")
    assert s.relevant[c] == frozenset({"0000000002"})


def test_pop_counts_last_week_only():
    rule = fixture_rule()
    a, b = ids_in("train", rule, 2)
    tx = tx_frame([
        ("2020-09-01", a, "0000000001"),  # before the pop week
        ("2020-09-01", b, "0000000001"),
        ("2020-09-09", a, "0000000002"),  # first day of the pop week (start - 7)
        ("2020-09-15", b, "0000000002"),
        ("2020-09-15", a, "0000000003"),
        ("2020-09-16", a, "0000000003"),  # window day: not counted
    ])
    s = data.build_split(tx, rule, "heldout")
    assert s.pop_counts == {"0000000002": 2, "0000000003": 1}


def test_history_order_and_ties():
    rule = fixture_rule()
    (c,) = ids_in("heldout", rule, 1)
    tx = tx_frame([
        ("2020-09-01", c, "0000000001"),
        ("2020-09-03", c, "0000000002"),
        ("2020-09-03", c, "0000000003"),  # same day, later in the file: more recent
        ("2020-09-02", c, "0000000004"),
        ("2020-09-17", c, "0000000009"),
    ])
    s = data.build_split(tx, rule, "heldout")
    last = data.last_n_history(s.history, 3)
    assert [p.article_id for p in last[c]] == ["0000000003", "0000000002", "0000000004"]
    assert last[c][0].t_dat == d("2020-09-03")


def test_training_pairs_train_bucket_before_cutoff():
    rule = fixture_rule()
    tr = ids_in("train", rule, 1)[0]
    dv = ids_in("dev", rule, 1)[0]
    ho = ids_in("heldout", rule, 1)[0]
    rows = []
    for c in (tr, dv, ho):
        rows += [("2020-08-01", c, "0000000001"), ("2020-08-01", c, "0000000005"), ("2020-08-05", c, "0000000002"), ("2020-09-10", c, "0000000003")]
    pairs = data.training_pairs(tx_frame(rows), rule, n_last=5)
    assert {p.customer_id for p in pairs} == {tr}
    assert all(p.target_date < d(rule["train_cutoff"]) for p in pairs)
    for p in pairs:
        assert all(h.t_dat < p.target_date for h in p.history)
    assert {(p.target_article, tuple(h.article_id for h in p.history)) for p in pairs} == {
        ("0000000002", ("0000000005", "0000000001")),
    }


def test_split_ids_sha256_canonical():
    rule = fixture_rule()
    tr, = ids_in("train", rule, 1)
    dv1, dv2 = ids_in("dev", rule, 2)
    ho, = ids_in("heldout", rule, 1)
    tx = tx_frame([
        ("2020-08-01", tr, "0000000001"), ("2020-09-10", tr, "0000000002"),
        ("2020-08-01", dv2, "0000000001"), ("2020-09-10", dv2, "0000000002"),
        ("2020-08-01", dv1, "0000000001"), ("2020-09-11", dv1, "0000000002"),
        ("2020-08-01", ho, "0000000001"), ("2020-09-20", ho, "0000000002"),
    ])
    # contracts/eval.md: lines "<split>\t<customer_id>\n", splits in the order
    # train, dev, heldout, ids sorted; train = every train-bucket customer with a
    # purchase before the cutoff; dev and heldout = the eligible customers.
    text = f"train\t{tr}\n" + "".join(f"dev\t{c}\n" for c in sorted([dv1, dv2])) + f"heldout\t{ho}\n"
    assert data.split_ids_sha256(tx, rule) == hashlib.sha256(text.encode()).hexdigest()
