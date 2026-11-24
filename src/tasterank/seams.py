"""The seams the lanes build against (contracts/seams.md).

Written in P2. No lane edits this file; a change goes under "Requests for
integration" in the lane's PR and P4 applies it. Lanes import these types and
implement the functions named in contracts/seams.md inside their own folders.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_RULE_PATH = REPO_ROOT / "contracts" / "split.json"

SplitName = Literal["train", "dev", "heldout"]
EvalSplitName = Literal["dev", "heldout"]

# Column names and polars dtypes the data lane returns (contracts/data.md).
TRANSACTION_COLUMNS = {
    "t_dat": "Date",
    "customer_id": "String",
    "article_id": "String",
    "price": "Float64",
    "sales_channel_id": "Int8",
    "row_idx": "UInt32",
}
ARTICLE_COLUMNS = {"article_id": "String", "product_group_name": "String"}

# Keys a results row may never carry (contracts/eval.md, "Results rows").
FORBIDDEN_RESULT_KEYS = frozenset(
    {"customer_id", "customer_ids", "customers", "article_id", "article_ids", "articles", "ids", "id_list"}
)
# Fields every results row carries (BUILD_OUTLINE.md section 6, item 7).
SECTION6_FIELDS = ("run_id", "git_sha", "machine", "model_revision", "quant", "runtime", "date")


def load_split_rule(path: Path | str = SPLIT_RULE_PATH) -> dict[str, Any]:
    """Reads a split rule (contracts/split.json format). The frozen rule is the default."""
    return json.loads(Path(path).read_text())


def parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


@dataclass(frozen=True)
class Purchase:
    article_id: str
    t_dat: dt.date


@dataclass(frozen=True)
class TrainPair:
    """One training example: a train-bucket customer's purchase and what came before it.

    history: the customer's purchases strictly before target_date, most recent
    first, at most n_last of them (contracts/eval.md, "Customer vector").
    """

    customer_id: str
    target_article: str
    target_date: dt.date
    history: tuple[Purchase, ...]


@dataclass
class EvalSplit:
    """One evaluation split (dev or heldout) as the data lane builds it.

    customers: eligible customers of this split's bucket, sorted.
    history: polars DataFrame, columns t_dat, customer_id, article_id, row_idx;
        those customers' purchases before window_start.
    relevant: customer_id -> distinct articles bought in the window (pool or not).
    pool: candidate article ids, sorted.
    pop_counts: article_id -> purchases in the pop window (pool articles only).
    cold_start: customers of this bucket with a window purchase and no history.
    """

    name: EvalSplitName
    window_start: dt.date
    window_end: dt.date
    customers: list[str]
    history: Any
    relevant: dict[str, frozenset[str]]
    pool: list[str]
    pop_counts: dict[str, int]
    cold_start: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class Encoder(Protocol):
    """An image tower: pixels [B, 3, H, W] in [0, 1] -> embeddings [B, D] (not normalized)."""

    image_size: int

    def __call__(self, pixels: Any) -> Any: ...


# article_id -> float tensor [3, image_size, image_size] in [0, 1]
ImageSource = Callable[[str], Any]


@dataclass
class TrainConfig:
    out_dir: Path
    train_cutoff: dt.date
    seed: int = 0
    steps: int = 100
    batch_size: int = 8
    lr: float = 1e-4
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    temperature: float = 0.07
    n_last: int = 20
    half_life_days: float = 30.0
    target_modules: tuple[str, ...] = ("attn.qkv", "attn.proj")


@dataclass
class TrainResult:
    losses: list[float]
    checkpoint: Path
    lora_state: dict[str, Any]
