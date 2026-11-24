"""Data lane (P3a): ingest, split, pool, histories, training pairs.

P2 stubs. Lane P3a replaces each body; signatures are in contracts/seams.md.
Tables are read with polars or DuckDB, never pandas.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from tasterank.seams import EvalSplit, EvalSplitName, SplitName, TrainPair


def load_transactions(path: Path | str) -> Any:
    raise NotImplementedError("P3a: load_transactions")


def load_articles(path: Path | str) -> Any:
    raise NotImplementedError("P3a: load_articles")


def customer_bucket(customer_id: str, rule: dict[str, Any]) -> int:
    raise NotImplementedError("P3a: customer_bucket")


def assign_split(customer_id: str, rule: dict[str, Any]) -> SplitName:
    raise NotImplementedError("P3a: assign_split")


def check_anchor(transactions: Any, rule: dict[str, Any]) -> None:
    raise NotImplementedError("P3a: check_anchor")


def build_split(transactions: Any, rule: dict[str, Any], which: EvalSplitName) -> EvalSplit:
    raise NotImplementedError("P3a: build_split")


def last_n_history(history: Any, n: int) -> dict[str, list[Any]]:
    raise NotImplementedError("P3a: last_n_history")


def training_pairs(transactions: Any, rule: dict[str, Any], n_last: int) -> list[TrainPair]:
    raise NotImplementedError("P3a: training_pairs")


def split_ids_sha256(transactions: Any, rule: dict[str, Any]) -> str:
    raise NotImplementedError("P3a: split_ids_sha256")
