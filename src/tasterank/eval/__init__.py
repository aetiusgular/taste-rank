"""Eval lane (P3c): ranking, metrics, per-category tables, bands, results rows.

Signatures are frozen in contracts/seams.md; the rules are in contracts/eval.md
("Candidate pool and POP", "Customer vector and score", "Metrics", "Rows and
bands", "Results rows"). `tasterank.eval.run` wires these into
`scripts/eval.sh --fixture`.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import secrets
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tasterank.seams import FORBIDDEN_RESULT_KEYS, REPO_ROOT, SECTION6_FIELDS

METRIC_KS = {"ndcg@10": 10, "recall@12": 12, "recall@50": 50}
LIFT_SEEDS = (0, 1, 2, 3, 4)
REQUIRED_ROW_FIELDS = (*SECTION6_FIELDS, "row", "split", "table", "metrics", "n_customers")
ROW_NAMES = frozenset({"POP", "E0-siglip", "E0-fclip", "E1-siglip", "E0-tiny", "E1-tiny"})
EVAL_SPLITS = frozenset({"dev", "heldout"})


# ---------------------------------------------------------------- ranking


def _order(candidate_ids: Sequence[str]) -> list[int]:
    """Indices of candidate_ids sorted by id ascending; refuses repeated ids."""
    ids = list(candidate_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("candidate_ids repeat an id")
    return sorted(range(len(ids)), key=ids.__getitem__)


def rank_many(customer_vecs: Any, candidate_ids: Sequence[str], candidate_embs: Any, k: int | None = None) -> list[list[str]]:
    """`rank` for a batch of customers [C, d]: one top-k list per row.

    Cosine similarity; ties by article id ascending (a stable sort over
    candidates already in id order). k None, or above the pool size, keeps
    the whole pool.
    """
    import torch
    from torch.nn import functional as F

    ids = list(candidate_ids)
    embs = torch.as_tensor(candidate_embs, dtype=torch.float64)
    vecs = torch.as_tensor(customer_vecs, dtype=torch.float64)
    if vecs.ndim == 1:
        vecs = vecs.unsqueeze(0)
    if embs.ndim != 2 or embs.shape[0] != len(ids):
        raise ValueError(f"{len(ids)} candidate ids but embeddings of shape {tuple(embs.shape)}")
    if vecs.ndim != 2 or vecs.shape[1] != embs.shape[1]:
        raise ValueError(f"customer vectors {tuple(vecs.shape)} do not match candidate dim {embs.shape[1]}")
    if k is not None and k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    order = _order(ids)
    sorted_ids = [ids[i] for i in order]
    scores = F.normalize(vecs, dim=-1) @ F.normalize(embs[order], dim=-1).T  # [C, N], columns in id order
    if bool(torch.isnan(scores).any()):
        raise ValueError("a score is NaN (non-finite embedding)")
    keep = len(ids) if k is None else min(k, len(ids))
    idx = torch.sort(scores, dim=1, descending=True, stable=True).indices[:, :keep]
    return [[sorted_ids[j] for j in row] for row in idx.tolist()]


def rank(customer_vec: Any, candidate_ids: Sequence[str], candidate_embs: Any, k: int) -> list[str]:
    """Top-k candidate ids by cosine similarity to customer_vec; ties by id ascending."""
    import torch

    v = torch.as_tensor(customer_vec)
    if v.ndim != 1:
        raise ValueError(f"customer_vec must be 1-d, got shape {tuple(v.shape)}")
    return rank_many(v, candidate_ids, candidate_embs, k)[0]


def pop_ranking(pop_counts: Mapping[str, int], k: int) -> list[str]:
    """Count descending, ties by article id ascending; the top k."""
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    return [a for a, _ in sorted(pop_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


# ---------------------------------------------------------------- metrics


def _relevant_set(relevant: Iterable[str]) -> frozenset[str]:
    r = frozenset(relevant)
    if not r:
        raise ValueError("relevant set is empty: a customer with no window purchase is not evaluated")
    return r


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """DCG/IDCG with binary gains; IDCG counts min(|R|, k) items, out-of-pool ones included."""
    r = _relevant_set(relevant)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    seen: set[str] = set()
    dcg = 0.0
    for i, a in enumerate(list(ranked)[:k]):
        if a in r and a not in seen:
            dcg += 1.0 / math.log2(i + 2)
            seen.add(a)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(r), k)))
    return dcg / idcg


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """|top-k ∩ R| / |R|."""
    r = _relevant_set(relevant)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return len(r.intersection(list(ranked)[:k])) / len(r)


def pool_coverage(relevant: Mapping[str, Iterable[str]], pool: Iterable[str]) -> float:
    """Σ |R ∩ pool| / Σ |R| over the customers in `relevant`."""
    p = frozenset(pool)
    hit = total = 0
    for arts in relevant.values():
        r = frozenset(arts)
        hit += len(r & p)
        total += len(r)
    if total == 0:
        raise ValueError("pool_coverage over no relevant purchases")
    return hit / total


def table_metrics(rankings: Mapping[str, Sequence[str]], relevant: Mapping[str, Iterable[str]]) -> dict[str, float]:
    """nDCG@10, recall@12 and recall@50, each a mean over customers, plus n_customers."""
    if set(rankings) != set(relevant):
        raise ValueError("rankings and relevant must cover the same customers")
    if not rankings:
        raise ValueError("no customers to evaluate")
    sums = dict.fromkeys(METRIC_KS, 0.0)
    for cid, ranked in rankings.items():
        r = relevant[cid]
        for name, k in METRIC_KS.items():
            sums[name] += (ndcg_at_k if name.startswith("ndcg") else recall_at_k)(ranked, r, k)
    n = len(rankings)
    return {**{name: s / n for name, s in sums.items()}, "n_customers": n}


def category_metrics(
    rankings: Mapping[str, Sequence[str]],
    relevant: Mapping[str, Iterable[str]],
    article_group: Mapping[str, str],
) -> dict[str, dict[str, float]]:
    """One table per product group g (contracts/eval.md, "Per-category tables").

    Customers with at least one window purchase in g; their ranking filtered to
    g articles; R filtered to g (pool or not). A purchased article with no group
    belongs to no table.
    """
    if set(rankings) != set(relevant):
        raise ValueError("rankings and relevant must cover the same customers")
    by_group: dict[str, dict[str, frozenset[str]]] = {}
    for cid in sorted(relevant):
        for a in relevant[cid]:
            g = article_group.get(a)
            if g is not None:
                by_group.setdefault(g, {}).setdefault(cid, frozenset())
    out: dict[str, dict[str, float]] = {}
    for g in sorted(by_group):
        rel_g = {cid: frozenset(a for a in relevant[cid] if article_group.get(a) == g) for cid in by_group[g]}
        rank_g = {cid: [a for a in rankings[cid] if article_group.get(a) == g] for cid in rel_g}
        out[g] = table_metrics(rank_g, rel_g)
    return out


# ---------------------------------------------------------------- bands


def band(values: Sequence[float]) -> dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("band of no values")
    if any(math.isnan(v) for v in vals):
        raise ValueError("band of a NaN value")
    return {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals), "n": len(vals)}


def lift_band(e1_by_seed: Mapping[int, float], e0: float) -> dict[str, Any]:
    """Per-seed lift E1_s − E0 and its band; verdict lift (min > 0), worse (max < 0), else no lift."""
    seeds = sorted(e1_by_seed)
    if tuple(seeds) != LIFT_SEEDS:
        raise ValueError(f"lift_band needs five seeds {list(LIFT_SEEDS)} (contracts/eval.md), got {seeds}")
    if math.isnan(float(e0)):
        raise ValueError("lift_band: E0 is NaN")
    out: dict[str, Any] = band([float(e1_by_seed[s]) - float(e0) for s in seeds])
    out["verdict"] = "lift" if out["min"] > 0 else "worse" if out["max"] < 0 else "no lift"
    return out


# ---------------------------------------------------------------- results rows


def _read_proc(path: str) -> str:
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def read_gpu() -> tuple[str | None, float | None]:
    """(name, power limit in W) of the first GPU from nvidia-smi, or (None, None) when there is none."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,power.limit", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None, None
    if not out:
        return None, None
    name, _, power = out[0].partition(",")
    try:
        watts: float | None = float(power.strip())
    except ValueError:
        watts = None
    return name.strip() or None, watts


def read_machine() -> str:
    """The machine label, read now (contracts/eval.md, "Results rows")."""
    mf = os.environ.get("FORGE_MACHINE_FILE", "")
    if mf:
        text = Path(mf).read_text().strip()
        if not text:
            raise ValueError(f"FORGE_MACHINE_FILE {mf} is empty")
        return text
    cpuinfo = _read_proc("/proc/cpuinfo")
    models = [l.split(":", 1)[1].strip() for l in cpuinfo.splitlines() if l.startswith("model name")]
    parts = [f"cpu: {models[0]} x{len(models)}" if models else "cpu: unknown"]
    mem = next((l.split()[1] for l in _read_proc("/proc/meminfo").splitlines() if l.startswith("MemTotal:")), None)
    parts.append(f"mem: {int(mem) / 1024**2:.1f} GiB" if mem else "mem: unknown")
    gpu, watts = read_gpu()
    if gpu:
        parts.append(f"gpu: {gpu}" + (f", power limit {watts:g} W" if watts is not None else ""))
    return "; ".join(parts)


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=30, check=True).stdout


def new_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(4)


def make_row(**fields: Any) -> dict[str, Any]:
    """A results row: the caller's fields plus run_id, git_sha, git_dirty, machine and date.

    git_sha (HEAD) and machine are always read here, never taken from the
    caller; run_id and date are kept when given, so one run's rows share them.
    """
    for key in ("git_sha", "git_dirty", "machine"):
        if key in fields:
            raise ValueError(f"make_row reads {key} itself; do not pass it")
    sha = _git("rev-parse", "HEAD").strip()
    if len(sha) != 40:
        raise RuntimeError(f"git rev-parse HEAD gave {sha!r}")
    row = dict(fields)
    row.setdefault("run_id", new_run_id())
    row.setdefault("date", dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    row["git_sha"] = sha
    row["git_dirty"] = bool(_git("status", "--porcelain", "--untracked-files=no").strip())
    row["machine"] = read_machine()
    return row


def _forbidden_keys(obj: Any, where: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            path = f"{where}.{k}" if where else str(k)
            if k in FORBIDDEN_RESULT_KEYS:
                found.append(path)
            found += _forbidden_keys(v, path)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found += _forbidden_keys(v, f"{where}[{i}]")
    return found


def check_row(row: Mapping[str, Any]) -> None:
    """ValueError unless the row is safe to publish and has the contract's required fields."""
    bad = _forbidden_keys(row)
    if bad:
        raise ValueError(f"results row has forbidden key(s) {bad}: rows never carry ids (contracts/eval.md)")
    missing = [f for f in REQUIRED_ROW_FIELDS if f not in row]
    if missing:
        raise ValueError(f"results row lacks {missing}")
    if row["row"] not in ROW_NAMES:
        raise ValueError(f"results row name {row['row']!r} is not one of {sorted(ROW_NAMES)}")
    if row["split"] not in EVAL_SPLITS:
        raise ValueError(f"results row split {row['split']!r} is not dev or heldout")
    if not isinstance(row["metrics"], Mapping) or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in row["metrics"].values()):
        raise ValueError("results row metrics must map names to numbers")
    n = row["n_customers"]
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError(f"results row n_customers must be an integer >= 0, got {n!r}")
    for key, kind in (("git_dirty", bool), ("dataset", str)):
        if key in row and not isinstance(row[key], kind):
            raise ValueError(f"results row {key} must be a {kind.__name__}")
    json.dumps(row, allow_nan=False)  # ValueError on NaN or inf


def write_rows(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> None:
    """Appends rows to a JSON lines file. Checks every row before writing any."""
    rows = list(rows)
    for r in rows:
        check_row(r)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, allow_nan=False) + "\n")
