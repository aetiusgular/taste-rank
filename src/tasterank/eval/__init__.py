"""Eval lane (P3c): ranking, metrics, per-category tables, bands, results rows.

P2 stubs. Lane P3c replaces each body; signatures are in contracts/seams.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def rank(customer_vec: Any, candidate_ids: Sequence[str], candidate_embs: Any, k: int) -> list[str]:
    raise NotImplementedError("P3c: rank")


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    raise NotImplementedError("P3c: ndcg_at_k")


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    raise NotImplementedError("P3c: recall_at_k")


def pool_coverage(relevant: Mapping[str, Iterable[str]], pool: Iterable[str]) -> float:
    raise NotImplementedError("P3c: pool_coverage")


def pop_ranking(pop_counts: Mapping[str, int], k: int) -> list[str]:
    raise NotImplementedError("P3c: pop_ranking")


def category_metrics(
    rankings: Mapping[str, Sequence[str]],
    relevant: Mapping[str, Iterable[str]],
    article_group: Mapping[str, str],
) -> dict[str, dict[str, float]]:
    raise NotImplementedError("P3c: category_metrics")


def band(values: Sequence[float]) -> dict[str, float]:
    raise NotImplementedError("P3c: band")


def lift_band(e1_by_seed: Mapping[int, float], e0: float) -> dict[str, Any]:
    raise NotImplementedError("P3c: lift_band")


def read_machine() -> str:
    raise NotImplementedError("P3c: read_machine")


def make_row(**fields: Any) -> dict[str, Any]:
    raise NotImplementedError("P3c: make_row")


def write_rows(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> None:
    raise NotImplementedError("P3c: write_rows")
