"""Train lane (P3b): customer vectors, InfoNCE, LoRA, the training loop.

P2 stubs. Lane P3b replaces each body; signatures are in contracts/seams.md.
"""

from __future__ import annotations

from typing import Any, Sequence

from tasterank.seams import ImageSource, TrainConfig, TrainPair, TrainResult


def set_seed(seed: int) -> None:
    raise NotImplementedError("P3b: set_seed")


def customer_vector(embeddings: Any, ages_days: Any, half_life_days: float) -> Any:
    raise NotImplementedError("P3b: customer_vector")


def info_nce(customer_vecs: Any, target_vecs: Any, temperature: float) -> Any:
    raise NotImplementedError("P3b: info_nce")


def apply_lora(encoder: Any, cfg: TrainConfig) -> Any:
    raise NotImplementedError("P3b: apply_lora")


def train(encoder: Any, pairs: Sequence[TrainPair], images: ImageSource, cfg: TrainConfig) -> TrainResult:
    raise NotImplementedError("P3b: train")
