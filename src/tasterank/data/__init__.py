"""Data lane (P3a): ingest, split, pool, histories, training pairs.

Signatures are frozen in contracts/seams.md; the rules are in contracts/eval.md
and contracts/data.md. Tables are read with polars, never pandas.

Every function that takes `transactions` accepts a polars DataFrame or
LazyFrame with the `TRANSACTION_COLUMNS` schema (what `load_transactions`
returns). Split functions read only rows dated on or before their window's
end, so a dev split never sees heldout days.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import polars as pl

from tasterank.seams import (
    ARTICLE_COLUMNS,
    TRANSACTION_COLUMNS,
    EvalSplit,
    EvalSplitName,
    Purchase,
    SplitName,
    TrainPair,
    parse_date,
)

_SPLIT_ORDER: tuple[SplitName, ...] = ("train", "dev", "heldout")
_HISTORY_COLUMNS = ["t_dat", "customer_id", "article_id", "row_idx"]


def _dtype(name: str) -> pl.DataType:
    return getattr(pl, name)


_TX_SCHEMA = {k: _dtype(v) for k, v in TRANSACTION_COLUMNS.items()}


def _eager(transactions: Any) -> pl.DataFrame:
    if isinstance(transactions, pl.LazyFrame):
        return transactions.collect()
    if isinstance(transactions, pl.DataFrame):
        return transactions
    raise TypeError(f"transactions must be a polars DataFrame or LazyFrame, got {type(transactions).__name__}")


# ---------------------------------------------------------------- ingest


def scan_transactions(path: Path | str) -> pl.LazyFrame:
    """Lazy form of `load_transactions`, for the 31M-row file on the box."""
    raw = pl.scan_csv(
        path,
        schema_overrides={
            "t_dat": pl.String,
            "customer_id": pl.String,
            "article_id": pl.String,
            "price": pl.Float64,
            "sales_channel_id": pl.Int8,
        },
        infer_schema=False,
    )
    return (
        raw.with_row_index("row_idx")
        .select(
            pl.col("t_dat").str.to_date("%Y-%m-%d", strict=True),
            pl.col("customer_id"),
            pl.col("article_id"),
            pl.col("price").cast(pl.Float64, strict=True),
            pl.col("sales_channel_id").cast(pl.Int8, strict=True),
            pl.col("row_idx").cast(pl.UInt32),
        )
    )


def load_transactions(path: Path | str) -> pl.DataFrame:
    """transactions_train.csv -> DataFrame with the contract's columns and dtypes.

    Ids stay strings (leading zeros kept); row_idx is the 0-based file row.
    A header-only file gives zero rows with the same schema.
    """
    df = scan_transactions(path).collect()
    return df.cast(_TX_SCHEMA)


def load_articles(path: Path | str) -> pl.DataFrame:
    """articles.csv -> DataFrame, every column read as a string (article_id keeps its zeros)."""
    df = pl.read_csv(path, infer_schema=False)
    missing = [c for c in ARTICLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: articles file lacks columns {missing}")
    return df


# ---------------------------------------------------------------- buckets


def customer_bucket(customer_id: str, rule: dict[str, Any]) -> int:
    """int(first 16 hex digits of sha256("<salt>:<customer_id>")) mod 100 (contracts/eval.md)."""
    h = rule["customer_hash"]
    if h.get("algorithm", "sha256") != "sha256":
        raise ValueError(f"unsupported customer_hash algorithm {h.get('algorithm')!r}")
    digest = hashlib.sha256(f"{h['salt']}:{customer_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % 100


def _split_of_bucket(bucket: int, rule: dict[str, Any]) -> SplitName:
    for name in _SPLIT_ORDER:
        lo, hi = rule["customer_hash"]["buckets"][name]
        if lo <= bucket <= hi:
            return name
    raise ValueError(f"bucket {bucket} is in no split range of the rule")


def assign_split(customer_id: str, rule: dict[str, Any]) -> SplitName:
    return _split_of_bucket(customer_bucket(customer_id, rule), rule)


def _customers_in(transactions: pl.DataFrame, rule: dict[str, Any], split: SplitName) -> pl.DataFrame:
    """One-column frame of the distinct customer ids in `transactions` whose bucket is `split`."""
    lo, hi = rule["customer_hash"]["buckets"][split]
    ids = transactions["customer_id"].unique().to_list()
    keep = [c for c in ids if lo <= customer_bucket(c, rule) <= hi]
    return pl.DataFrame({"customer_id": keep}, schema={"customer_id": pl.String})


# ---------------------------------------------------------------- anchor


def check_anchor(transactions: Any, rule: dict[str, Any]) -> None:
    """Refuses a file whose first or last t_dat differs from the rule (a different file version)."""
    tx = _eager(transactions)
    anchor = rule["anchor"]
    want_min = parse_date(anchor["expected_min_t_dat"])
    want_max = parse_date(anchor["expected_max_t_dat"])
    if tx.height == 0:
        raise ValueError(f"min t_dat and max t_dat: the table is empty, rule expects {want_min} and {want_max}")
    got_min, got_max = tx["t_dat"].min(), tx["t_dat"].max()
    if got_min != want_min:
        raise ValueError(f"min t_dat is {got_min}, rule expects {want_min}")
    if got_max != want_max:
        raise ValueError(f"max t_dat is {got_max}, rule expects {want_max}")


# ---------------------------------------------------------------- splits


def _window(rule: dict[str, Any], which: EvalSplitName) -> tuple[dt.date, dt.date]:
    if which not in ("dev", "heldout"):
        raise ValueError(f"eval split must be 'dev' or 'heldout', got {which!r}")
    w = rule["windows"][which]
    start, end = parse_date(w["start"]), parse_date(w["end"])
    if end < start:
        raise ValueError(f"{which} window ends before it starts")
    return start, end


def _eligible(tx: pl.DataFrame, rule: dict[str, Any], which: EvalSplitName):
    """(visible rows, eligible ids frame, window-only count) for one eval split."""
    start, end = _window(rule, which)
    visible = tx.filter(pl.col("t_dat") <= end)
    members = _customers_in(visible, rule, which)
    mine = visible.join(members, on="customer_id", how="semi")
    with_hist = mine.filter(pl.col("t_dat") < start).select("customer_id").unique()
    with_win = mine.filter(pl.col("t_dat") >= start).select("customer_id").unique()
    eligible = with_hist.join(with_win, on="customer_id", how="semi").sort("customer_id")
    cold = with_win.join(with_hist, on="customer_id", how="anti").height
    return visible, mine, eligible, cold


def build_split(transactions: Any, rule: dict[str, Any], which: EvalSplitName) -> EvalSplit:
    """One evaluation split (contracts/eval.md: Split, Candidate pool and POP)."""
    tx = _eager(transactions)
    start, end = _window(rule, which)
    visible, mine, eligible, cold = _eligible(tx, rule, which)

    history = (
        mine.filter(pl.col("t_dat") < start)
        .join(eligible, on="customer_id", how="semi")
        .select(_HISTORY_COLUMNS)
        .sort(["customer_id", "t_dat", "row_idx"])
    )
    window_rows = mine.filter(pl.col("t_dat") >= start).join(eligible, on="customer_id", how="semi")
    relevant = {
        cid: frozenset(arts)
        for cid, arts in window_rows.group_by("customer_id")
        .agg(pl.col("article_id").unique())
        .iter_rows()
    }

    pool_from = start - dt.timedelta(days=int(rule["pool_days_before_window"]))
    pool = (
        visible.filter((pl.col("t_dat") >= pool_from) & (pl.col("t_dat") < start))["article_id"]
        .unique()
        .sort()
        .to_list()
    )
    pop_from = start - dt.timedelta(days=int(rule["pop_days_before_window"]))
    pool_frame = pl.DataFrame({"article_id": pool}, schema={"article_id": pl.String})
    pop_counts = {
        a: int(n)
        for a, n in visible.filter((pl.col("t_dat") >= pop_from) & (pl.col("t_dat") < start))
        .join(pool_frame, on="article_id", how="semi")
        .group_by("article_id")
        .len()
        .iter_rows()
    }

    return EvalSplit(
        name=which,
        window_start=start,
        window_end=end,
        customers=eligible["customer_id"].to_list(),
        history=history,
        relevant=relevant,
        pool=pool,
        pop_counts=pop_counts,
        cold_start=cold,
        extra={"pool_start": pool_from, "pop_start": pop_from},
    )


def last_n_history(history: Any, n: int) -> dict[str, list[Purchase]]:
    """customer_id -> last n purchases, most recent first; same day: later file row first."""
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    h = _eager(history)
    top = (
        h.sort(["customer_id", "t_dat", "row_idx"], descending=[False, True, True])
        .group_by("customer_id", maintain_order=True)
        .head(n)
    )
    out: dict[str, list[Purchase]] = {}
    for cid, art, day in top.select("customer_id", "article_id", "t_dat").iter_rows():
        out.setdefault(cid, []).append(Purchase(art, day))
    return out


# ---------------------------------------------------------------- training


def iter_training_pairs(transactions: Any, rule: dict[str, Any], n_last: int) -> Iterator[TrainPair]:
    """Generator form of `training_pairs`, in customer id order then file order.

    Train-bucket customers only, targets dated before train_cutoff, history
    strictly before the target day (most recent first, later row first on a
    tie), at most n_last of it. Pairs with no history are dropped.
    """
    if n_last < 1:
        raise ValueError(f"n_last must be at least 1, got {n_last}")
    tx = _eager(transactions)
    cutoff = parse_date(rule["train_cutoff"])
    before = tx.filter(pl.col("t_dat") < cutoff)
    rows = (
        before.join(_customers_in(before, rule, "train"), on="customer_id", how="semi")
        .select("customer_id", "t_dat", "row_idx", "article_id")
        .sort(["customer_id", "t_dat", "row_idx"])
    )
    for (cid,), group in rows.group_by("customer_id", maintain_order=True):
        seq = [Purchase(a, t) for t, a in group.select("t_dat", "article_id").iter_rows()]
        # newest-first view; history for a target on day t is every purchase on an earlier day.
        newest_first = seq[::-1]
        n = len(seq)
        first_of_day = 0  # index in seq of the first purchase on the current day
        for j, p in enumerate(seq):
            if j > 0 and p.t_dat != seq[j - 1].t_dat:
                first_of_day = j
            if first_of_day == 0:
                continue
            hist = tuple(newest_first[n - first_of_day : n - first_of_day + n_last])
            yield TrainPair(customer_id=cid, target_article=p.article_id, target_date=p.t_dat, history=hist)


def training_pairs(transactions: Any, rule: dict[str, Any], n_last: int) -> list[TrainPair]:
    return list(iter_training_pairs(transactions, rule, n_last))


def split_ids_sha256(transactions: Any, rule: dict[str, Any]) -> str:
    """sha256 of "<split>\\t<customer_id>\\n" lines: train, dev, heldout, ids sorted in each."""
    tx = _eager(transactions)
    cutoff = parse_date(rule["train_cutoff"])
    before = tx.filter(pl.col("t_dat") < cutoff)
    lists = {"train": sorted(_customers_in(before, rule, "train")["customer_id"].to_list())}
    for which in ("dev", "heldout"):
        lists[which] = _eligible(tx, rule, which)[2]["customer_id"].to_list()
    h = hashlib.sha256()
    for name in _SPLIT_ORDER:
        for cid in lists[name]:
            h.update(f"{name}\t{cid}\n".encode("utf-8"))
    return h.hexdigest()
