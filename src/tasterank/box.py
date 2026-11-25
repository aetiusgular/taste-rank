"""The box pipeline behind `scripts/eval.sh` with no arguments (P5).

    python -m tasterank.box [--root DIR] [--data DIR] [--work DIR] [--out DIR] [--stages ...]

Order (contracts/eval.md, BUILD_OUTLINE.md section 6):

1. preflight: data files, verified model folders, CUDA, a work folder outside
   the repo. Anything missing exits 3 and names it; nothing is written.
2. load the transactions (polars), `check_anchor`, build dev and heldout,
   `split_ids_sha256`; sample the E1 training customers.
3. image caches: every needed article image, prepared once per geometry, as
   uint8 in a memmap under the work folder (never in the repo).
4. POP on dev and heldout.
5. E0 grid on dev with the headline base model only, before any E1 run:
   one overall row per grid point; the best `ndcg@10` picks (n_last,
   half-life), ties to the default and then grid order.
6. E0 rows (base and second model) on dev and heldout at that point, plus one
   repeat of the base heldout row from a fresh load, which must match to 1e-6.
7. E1: five seeds, LoRA trained on train customers only, then dev and heldout.
8. Lift band on heldout overall nDCG@10, summary.md and the failure-case table.

Rows go to <out>/box-<run_id>.jsonl as each stage finishes, so a later crash
keeps the earlier rows. No row holds an id (write_rows refuses them).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tasterank.seams import REPO_ROOT, SPLIT_RULE_PATH, EvalSplit, Purchase, TrainConfig, TrainPair, load_split_rule

DEFAULT_ROOT = Path("/data/models/taste-rank")
E1_CONFIG_PATH = REPO_ROOT / "configs" / "e1.json"
DATASET_LABEL = "H&M Personalized Fashion Recommendations (Kaggle)"
TOP_K = 50
REPEAT_TOLERANCE = 1e-6
EXIT_NOT_PROVISIONED = 3
DATA_FILES = ("transactions_train.csv", "articles.csv")


# ---------------------------------------------------------------- paths


@dataclass
class BoxPaths:
    root: Path
    data: Path
    work: Path
    out: Path

    @property
    def siglip(self) -> Path:
        return self.root / "models" / "marqo-fashionSigLIP"

    @property
    def fclip(self) -> Path:
        return self.root / "models" / "fashion-clip"


def default_paths(root: Path | None = None, data: Path | None = None, work: Path | None = None, out: Path | None = None) -> BoxPaths:
    root = Path(root or os.environ.get("TASTE_RANK_ROOT") or DEFAULT_ROOT)
    data = Path(data or os.environ.get("TASTE_RANK_DATA") or root / "hm")
    if work is None:
        tmp = os.environ.get("FORGE_TMP")
        work = Path(tmp) / "taste-rank" if tmp else None
    return BoxPaths(root=root, data=data, work=work, out=Path(out or REPO_ROOT / "results"))  # type: ignore[arg-type]


def _inside(path: Path, parent: Path) -> bool:
    p, r = path.expanduser().resolve(), parent.resolve()
    return p == r or r in p.parents


def preflight(paths: BoxPaths, need_cuda: bool = True) -> list[str]:
    """What the box lacks for a run; empty when it can start. Reads, never writes."""
    from tasterank.models import load_lock, verify_model_dir

    problems: list[str] = []
    folders = {"Marqo/marqo-fashionSigLIP": paths.siglip, "patrickjohncyh/fashion-clip": paths.fclip}
    for name in DATA_FILES:
        if not (paths.data / name).is_file():
            problems.append(f"data: {paths.data / name} missing (scripts/box-setup.sh data, needs the Kaggle token)")
    if not (paths.data / "images").is_dir():
        problems.append(f"data: {paths.data / 'images'} missing")
    for entry in load_lock()["models"]:
        try:
            verify_model_dir(folders[entry["id"]], entry)
        except ValueError as e:
            problems.append(f"model: {e} (scripts/box-setup.sh models)")
    if paths.work is None:
        problems.append("work: no work folder (FORGE_TMP unset and no --work)")
    elif _inside(paths.work, REPO_ROOT):
        problems.append(f"work: {paths.work} is inside the repo; caches and LoRA weights stay out of it")
    if need_cuda:
        try:
            import torch

            if not torch.cuda.is_available():
                problems.append("cuda: torch sees no GPU (srt-gpu profile, driver, venv torch build)")
        except ImportError:
            problems.append("python: torch is not installed in this interpreter")
    for mod in ("numpy", "PIL", "open_clip", "transformers", "peft", "polars"):
        try:
            __import__(mod)
        except ImportError:
            problems.append(f"python: {mod} is not installed (requirements-box.txt)")
    return problems


# ---------------------------------------------------------------- E1 config


def load_e1_config(path: Path | str = E1_CONFIG_PATH) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text())
    for key in ("steps", "lr", "rank", "alpha", "dropout", "temperature", "max_images_per_step", "batch_size_max", "batch_size_min", "train_customers", "sample_salt", "grad_checkpointing", "embed_batch"):
        if key not in cfg:
            raise ValueError(f"{path}: missing {key}")
    return cfg


def e1_batch_size(cfg: Mapping[str, Any], n_last: int) -> int:
    """Pairs per step: as many as fit `max_images_per_step` images (each pair needs 1 + n_last), within the bounds."""
    fit = int(cfg["max_images_per_step"]) // (1 + int(n_last))
    return max(int(cfg["batch_size_min"]), min(int(cfg["batch_size_max"]), fit))


# ---------------------------------------------------------------- image cache


def _prepare_one(args: tuple[str, str, int, str]) -> Any:
    from tasterank.encoders import prepare_image

    path, _aid, size, geometry = args
    try:
        return prepare_image(path, size, geometry)
    except (OSError, ValueError):
        return None


@dataclass
class ImageCache:
    """uint8 [N, 3, S, S] in a memmap file; `source(article_id)` -> float tensor in [0, 1] or None."""

    index: dict[str, int]
    array: Any
    size: int
    geometry: str
    missing: frozenset[str] = frozenset()

    def source(self, article_id: str) -> Any | None:
        import torch

        i = self.index.get(article_id)
        if i is None:
            return None
        return torch.from_numpy(self.array[i].astype("float32") / 255.0)

    def __contains__(self, article_id: str) -> bool:
        return article_id in self.index


def image_path(images_dir: Path, article_id: str) -> Path:
    from tasterank.eval.run import ARTICLE_ID_RE

    if not isinstance(article_id, str) or not ARTICLE_ID_RE.fullmatch(article_id):
        raise ValueError(f"article id {article_id!r} is not [0-9A-Za-z_-]+; it would name a path outside the image folder")
    return images_dir / article_id[:3] / f"{article_id}.jpg"


def build_image_cache(
    images_dir: Path,
    article_ids: Sequence[str],
    size: int,
    geometry: str,
    cache_dir: Path,
    workers: int | None = None,
    log: Callable[[str], None] = print,
) -> ImageCache:
    """Prepares each id's image once (missing or unreadable files are listed, not fatal)."""
    import numpy as np

    ids = sorted(set(article_ids))
    have = [a for a in ids if image_path(images_dir, a).is_file()]
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256(("\n".join(have) + f"|{size}|{geometry}").encode()).hexdigest()[:16]
    arr_path = cache_dir / f"images-{geometry}-{size}-{tag}.u8"
    ok_path = cache_dir / f"images-{geometry}-{size}-{tag}.ok.json"
    shape = (len(have), 3, size, size)
    if ok_path.is_file() and arr_path.is_file():
        done = json.loads(ok_path.read_text())
        arr = np.memmap(arr_path, dtype=np.uint8, mode="r", shape=shape)
        bad = set(done["unreadable"])
        index = {a: i for i, a in enumerate(have) if a not in bad}
        log(f"image cache {geometry}/{size}: reused {len(index)} images")
        return ImageCache(index, arr, size, geometry, frozenset(set(ids) - set(index)))
    arr = np.memmap(arr_path, dtype=np.uint8, mode="w+", shape=shape) if have else np.zeros(shape, dtype=np.uint8)
    unreadable: list[str] = []
    t0 = time.time()
    jobs = [(str(image_path(images_dir, a)), a, size, geometry) for a in have]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers == 1 or len(jobs) < 64:
        results = map(_prepare_one, jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(_prepare_one, jobs, chunksize=64)
    for i, img in enumerate(results):
        if img is None:
            unreadable.append(have[i])
        else:
            arr[i] = img
        if i and i % 20000 == 0:
            log(f"image cache {geometry}/{size}: {i}/{len(jobs)} in {time.time() - t0:.0f}s")
    if workers != 1 and len(jobs) >= 64:
        pool.shutdown()
    if have:
        arr.flush()
    ok_path.write_text(json.dumps({"n": len(have), "unreadable": unreadable}))
    bad = set(unreadable)
    index = {a: i for i, a in enumerate(have) if a not in bad}
    log(f"image cache {geometry}/{size}: {len(index)} images, {len(ids) - len(index)} missing or unreadable, {time.time() - t0:.0f}s")
    return ImageCache(index, arr, size, geometry, frozenset(set(ids) - set(index)))


# ---------------------------------------------------------------- embeddings and rankings


def embed(encoder: Any, ids: Sequence[str], cache: ImageCache, batch: int = 256) -> dict[str, Any]:
    """article_id -> float32 embedding [D] on CPU, for every id the cache holds; streams in batches."""
    import torch

    have = [a for a in dict.fromkeys(ids) if a in cache]
    device = next(encoder.parameters()).device
    out: dict[str, Any] = {}
    with torch.no_grad():
        for i in range(0, len(have), batch):
            chunk = have[i : i + batch]
            px = torch.from_numpy(cache.array[[cache.index[a] for a in chunk]].astype("float32") / 255.0).to(device)
            embs = encoder(px).float().cpu()
            out.update(zip(chunk, embs))
    return out


def customer_vectors(split: EvalSplit, hist: Mapping[str, Sequence[Purchase]], embs: Mapping[str, Any], half_life_days: float | None) -> tuple[Any, int]:
    """[C, D] vectors in split.customers order (eval.md "Customer vector"; missing images as in encoder_rankings)."""
    import torch

    from tasterank.train import customer_vector

    dim = next(iter(embs.values())).shape[0]
    hl = math.inf if half_life_days is None else half_life_days
    vecs, none = [], 0
    for c in split.customers:
        items = [p for p in hist.get(c, []) if p.article_id in embs]
        if not items:
            none += 1
            vecs.append(torch.zeros(dim))
            continue
        e = torch.stack([embs[p.article_id] for p in items])
        ages = torch.tensor([(split.window_start - p.t_dat).days for p in items], dtype=e.dtype)
        vecs.append(customer_vector(e, ages, hl))
    return torch.stack(vecs), none


def truncated_rankings(
    customers: Sequence[str],
    vecs: Any,
    pool: Sequence[str],
    embs: Mapping[str, Any],
    customer_groups: Mapping[str, set[str]],
    article_group: Mapping[str, str],
    k: int = TOP_K,
    device: str = "cpu",
    chunk: int = 512,
) -> dict[str, list[str]]:
    """Per customer: the overall top-k plus the top-k of each product group they bought, merged in rank order.

    Same order as `rank_many` over the whole pool (float64 cosine, ties by id
    ascending, articles without an image after every scored one, by id), cut
    to what the metrics read: `evaluate` on these lists gives the same tables
    as on the full rankings, since each table reads at most its top 50.
    """
    import torch
    from torch.nn import functional as F

    if list(pool) != sorted(pool):
        raise ValueError("pool must be sorted by id")
    scored = [a for a in pool if a in embs]
    unscored = [a for a in pool if a not in embs]
    if not scored:
        raise ValueError("no pool article has an image")
    groups = sorted({g for gs in customer_groups.values() for g in gs})
    gcode = {g: i for i, g in enumerate(groups)}
    code = torch.tensor([gcode.get(article_group.get(a, ""), -1) for a in scored], device=device)
    cand = F.normalize(torch.stack([embs[a] for a in scored]).to(device, torch.float64), dim=-1)
    tail_all = unscored[:k]
    tail_by_group: dict[str, list[str]] = {}
    for a in unscored:
        g = article_group.get(a)
        if g in gcode and len(tail_by_group.setdefault(g, [])) < k:
            tail_by_group[g].append(a)
    out: dict[str, list[str]] = {}
    vecs = torch.as_tensor(vecs, dtype=torch.float64)
    first = min(k, len(scored))
    for s in range(0, len(customers), chunk):
        cs = list(customers[s : s + chunk])
        v = F.normalize(vecs[s : s + chunk].to(device), dim=-1)
        scores = v @ cand.T
        if bool(torch.isnan(scores).any()):
            raise ValueError("a score is NaN (non-finite embedding)")
        order = torch.sort(scores, dim=1, descending=True, stable=True).indices  # [c, P], ties by id
        codes_sorted = code[order]
        keep = torch.zeros_like(order, dtype=torch.bool)
        keep[:, :first] = True
        for row, c in enumerate(cs):
            for g in customer_groups.get(c, ()):
                if g in gcode:
                    mask = codes_sorted[row] == gcode[g]
                    keep[row] |= mask & (torch.cumsum(mask, dim=0) <= k)
        for row, c in enumerate(cs):
            idx = order[row][keep[row]].cpu().tolist()
            tail = set(tail_all)
            for g in customer_groups.get(c, ()):
                tail.update(tail_by_group.get(g, ()))
            out[c] = [scored[j] for j in idx] + sorted(tail)
    return out


def customer_groups_of(split: EvalSplit, article_group: Mapping[str, str]) -> dict[str, set[str]]:
    return {c: {article_group[a] for a in split.relevant[c] if a in article_group} for c in split.customers}


def missing_count(split: EvalSplit, hist: Mapping[str, Sequence[Purchase]], embs: Mapping[str, Any]) -> int:
    need = set(split.pool) | {p.article_id for c in split.customers for p in hist.get(c, [])}
    return len(need - set(embs))


# ---------------------------------------------------------------- training pairs


def sample_train_customers(transactions: Any, rule: Mapping[str, Any], n: int, salt: str) -> list[str]:
    """The first n train customers (purchase before train_cutoff) by sha256(salt:id): the same set for every seed."""
    import polars as pl

    from tasterank.data import assign_split
    from tasterank.seams import parse_date

    cutoff = parse_date(rule["train_cutoff"])
    tx = transactions.lazy() if hasattr(transactions, "lazy") else transactions
    ids = tx.filter(pl.col("t_dat") < cutoff).select("customer_id").unique().collect()["customer_id"].to_list()
    train = [c for c in ids if assign_split(c, dict(rule)) == "train"]
    train.sort(key=lambda c: hashlib.sha256(f"{salt}:{c}".encode()).hexdigest())
    return sorted(train[:n])


def train_pairs_for(transactions: Any, rule: Mapping[str, Any], customers: Sequence[str], n_last: int) -> list[TrainPair]:
    import polars as pl

    from tasterank.data import iter_training_pairs

    frame = pl.DataFrame({"customer_id": list(customers)}, schema={"customer_id": pl.String})
    tx = transactions.lazy() if hasattr(transactions, "lazy") else transactions
    sub = tx.join(frame.lazy(), on="customer_id", how="semi").collect()
    return list(iter_training_pairs(sub, dict(rule), n_last))


def drop_pairs_without_images(pairs: Sequence[TrainPair], cache: ImageCache, n_last: int) -> tuple[list[TrainPair], int]:
    """eval.md "Missing images": a pair whose target or a kept history item has no image is dropped."""
    kept = [p for p in pairs if p.target_article in cache and all(h.article_id in cache for h in p.history[:n_last])]
    return kept, len(pairs) - len(kept)


# ---------------------------------------------------------------- the run


@dataclass
class ModelSpec:
    row_e0: str
    load: Callable[[], Any]
    revision: str
    geometry: str
    image_size: int
    row_e1: str | None = None
    quant: str | None = None


@dataclass
class RunContext:
    rule: dict[str, Any]
    rule_sha256: str
    split_ids: str
    splits: dict[str, EvalSplit]
    article_group: dict[str, str]
    out_file: Path
    run_id: str
    runtime: str
    dataset: str
    gpu: str | None
    gpu_power_limit_w: float | None
    rows: list[dict[str, Any]] = field(default_factory=list)

    def common(self, split: str) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "split": split,
            "split_rule_sha256": self.rule_sha256,
            "split_ids_sha256": self.split_ids,
            "cold_start": self.splits[split].cold_start,
            "gpu": self.gpu,
            "gpu_power_limit_w": self.gpu_power_limit_w,
            "dataset": self.dataset,
            "runtime": self.runtime,
        }

    def write(self, rows: list[dict[str, Any]]) -> None:
        from tasterank import eval as ev

        ev.write_rows(self.out_file, rows)
        self.rows.extend(rows)


def _table_rows(ctx: RunContext, split: str, rankings: Mapping[str, Sequence[str]], **fields: Any) -> list[dict[str, Any]]:
    from tasterank import eval as ev
    from tasterank.eval.run import evaluate

    return [ev.make_row(**ctx.common(split), **fields, **t) for t in evaluate(ctx.splits[split], rankings, ctx.article_group)]


def split_embeddings(encoder: Any, cache: ImageCache, split: EvalSplit, n_hist: int, embed_batch: int) -> dict[str, Any]:
    """Embeddings of the split's pool and its customers' last n_hist purchases (one pass per encoder and split)."""
    from tasterank.data import last_n_history

    hist = last_n_history(split.history, n_hist)
    return embed(encoder, list(split.pool) + [p.article_id for c in split.customers for p in hist.get(c, [])], cache, embed_batch)


def rankings_from(
    ctx: RunContext, split: str, embs: Mapping[str, Any], n_last: int, half_life: float | None, device: str, overall_only: bool = False
) -> tuple[dict[str, list[str]], int, int]:
    """(truncated rankings, missing_images, customers with no history image) at one history setting."""
    from tasterank.data import last_n_history

    sp = ctx.splits[split]
    hist = last_n_history(sp.history, n_last)
    vecs, no_img = customer_vectors(sp, hist, embs, half_life)
    groups = {} if overall_only else customer_groups_of(sp, ctx.article_group)
    rankings = truncated_rankings(sp.customers, vecs, sp.pool, embs, groups, ctx.article_group, TOP_K, device)
    return rankings, missing_count(sp, hist, embs), no_img


def choose_history(ctx: RunContext, spec: ModelSpec, cache: ImageCache, device: str, embed_batch: int, log=print) -> tuple[int, float | None]:
    """E0 grid on dev with the base model: one overall row per point; best ndcg@10, ties to the default then grid order."""
    from tasterank import eval as ev

    grid = ctx.rule["history_grid"]
    default = (int(ctx.rule["history_default"]["n_last"]), ctx.rule["history_default"]["half_life_days"])
    points = [default] + [(int(n), h) for n in grid["n_last"] for h in grid["half_life_days"] if (int(n), h) != default]
    encoder = spec.load()
    sp = ctx.splits["dev"]
    embs = split_embeddings(encoder, cache, sp, max(n for n, _ in points), embed_batch)
    del encoder
    _free_gpu()
    rows, scores = [], []
    for n_last, hl in points:
        rankings, missing, _ = rankings_from(ctx, "dev", embs, n_last, hl, device, overall_only=True)
        rel = {c: sp.relevant[c] for c in sp.customers}
        m = ev.table_metrics(rankings, rel)
        n = m.pop("n_customers")
        m["pool_coverage"] = ev.pool_coverage(rel, sp.pool)
        scores.append(m["ndcg@10"])
        rows.append(ev.make_row(**ctx.common("dev"), row=spec.row_e0, model_revision=spec.revision, quant=spec.quant, seed=None,
                                n_last=n_last, half_life_days=hl, missing_images=missing, table="overall", metrics=m,
                                n_customers=n, stage="history_grid"))
        log(f"grid n_last={n_last} half_life={hl}: dev ndcg@10 {m['ndcg@10']:.5f}")
    best = max(range(len(points)), key=lambda i: (scores[i], -i))
    for i, r in enumerate(rows):
        r["grid_choice"] = i == best
    ctx.write(rows)
    return points[best]


def _reset_peak() -> bool:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        return True
    return False


def _peak_gib() -> float:
    import torch

    return round(torch.cuda.max_memory_allocated() / 2**30, 2)


def _free_gpu() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass


def max_metric_diff(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]]) -> float:
    """Largest absolute difference between two row lists' metrics, table by table; inf if the tables differ."""
    ma = {r["table"]: r["metrics"] for r in a}
    mb = {r["table"]: r["metrics"] for r in b}
    if set(ma) != set(mb):
        return math.inf
    worst = 0.0
    for t in ma:
        if set(ma[t]) != set(mb[t]):
            return math.inf
        for k in ma[t]:
            worst = max(worst, abs(float(ma[t][k]) - float(mb[t][k])))
    return worst


def run_box(
    paths: BoxPaths,
    base: ModelSpec,
    second: ModelSpec | None,
    e1cfg: Mapping[str, Any],
    rule_path: Path = SPLIT_RULE_PATH,
    device: str = "cuda",
    dataset: str = DATASET_LABEL,
    seeds: Sequence[int] | None = None,
    stages: Sequence[str] = ("pop", "grid", "e0", "e1"),
    log: Callable[[str], None] = print,
) -> RunContext:
    import polars as pl

    from tasterank import eval as ev
    from tasterank.data import build_split, check_anchor, load_articles, scan_transactions, split_ids_sha256
    from tasterank.eval.run import pop_rankings

    t0 = time.time()
    rule = load_split_rule(rule_path)
    seeds = list(rule["seeds"] if seeds is None else seeds)
    tx = scan_transactions(paths.data / "transactions_train.csv").collect()
    check_anchor(tx, rule)
    articles = load_articles(paths.data / "articles.csv")
    group = {a: g for a, g in articles.select("article_id", "product_group_name").iter_rows() if g is not None}
    splits = {w: build_split(tx, rule, w) for w in ("dev", "heldout")}
    gpu, watts = ev.read_gpu()
    ctx = RunContext(
        rule=rule,
        rule_sha256=hashlib.sha256(Path(rule_path).read_bytes()).hexdigest(),
        split_ids=split_ids_sha256(tx, rule),
        splits=splits,
        article_group=group,
        out_file=paths.out / f"box-{ev.new_run_id()}.jsonl",
        run_id="",
        runtime="",
        dataset=dataset,
        gpu=gpu,
        gpu_power_limit_w=watts,
    )
    ctx.run_id = ctx.out_file.stem.removeprefix("box-")
    from tasterank.encoders import runtime_label

    ctx.runtime = runtime_label()
    log(f"run {ctx.run_id}: {tx.height} transactions; dev {len(splits['dev'].customers)} customers, pool {len(splits['dev'].pool)}; "
        f"heldout {len(splits['heldout'].customers)} customers, pool {len(splits['heldout'].pool)}; {time.time() - t0:.0f}s")

    # Training customers and pairs at the widest n_last, so the cache covers every grid point.
    grid_max = max(int(n) for n in rule["history_grid"]["n_last"])
    train_customers = sample_train_customers(tx, rule, int(e1cfg["train_customers"]), str(e1cfg["sample_salt"])) if "e1" in stages else []

    # Image caches: base geometry for everything E0/E1 reads; second geometry for E0 only.
    from tasterank.data import last_n_history

    eval_ids: set[str] = set()
    for sp in splits.values():
        eval_ids.update(sp.pool)
        for ps in last_n_history(sp.history, grid_max).values():
            eval_ids.update(p.article_id for p in ps)
    pairs_all: list[TrainPair] = []
    if train_customers:
        pairs_all = train_pairs_for(tx, rule, train_customers, grid_max)
    train_ids = {p.target_article for p in pairs_all} | {h.article_id for p in pairs_all for h in p.history}
    del tx
    images = paths.data / "images"
    cache_dir = paths.work / "cache"
    base_cache = build_image_cache(images, sorted(eval_ids | train_ids), base.image_size, base.geometry, cache_dir, log=log)
    second_cache = None
    if second is not None and "e0" in stages:
        second_cache = (base_cache if (second.image_size, second.geometry) == (base.image_size, base.geometry)
                        else build_image_cache(images, sorted(eval_ids), second.image_size, second.geometry, cache_dir, log=log))

    if "pop" in stages:
        for split in ("dev", "heldout"):
            ctx.write(_table_rows(ctx, split, pop_rankings(splits[split]), row="POP", model_revision=None, quant=None, seed=None,
                                  n_last=None, half_life_days=None, missing_images=0))
        log("POP rows written")

    embed_batch = int(e1cfg["embed_batch"])
    if "grid" in stages:
        n_last, half_life = choose_history(ctx, base, base_cache, device, embed_batch, log)
    else:
        n_last, half_life = int(rule["history_default"]["n_last"]), rule["history_default"]["half_life_days"]
    log(f"history: n_last {n_last}, half-life {half_life}")

    def encoder_rows(spec: ModelSpec, encoder: Any, cache: ImageCache, split: str, row: str, **extra: Any) -> list[dict[str, Any]]:
        embs = split_embeddings(encoder, cache, ctx.splits[split], n_last, embed_batch)
        rankings, missing, no_img = rankings_from(ctx, split, embs, n_last, half_life, device)
        return _table_rows(ctx, split, rankings, row=row, model_revision=spec.revision, quant=spec.quant, n_last=n_last,
                           half_life_days=half_life, missing_images=missing, no_history_image_customers=no_img, **extra)

    e0_heldout: list[dict[str, Any]] = []
    if "e0" in stages:
        enc = base.load()
        for split in ("dev", "heldout"):
            rows = encoder_rows(base, enc, base_cache, split, base.row_e0, seed=None)
            ctx.write(rows)
            if split == "heldout":
                e0_heldout = rows
        del enc
        _free_gpu()
        enc = base.load()  # the repeat: a fresh load, fresh embeddings
        rep = encoder_rows(base, enc, base_cache, "heldout", base.row_e0, seed=None, repeat_of=ctx.run_id)
        diff = max_metric_diff(e0_heldout, rep)
        for r in rep:
            r["repeat_max_abs_diff"] = diff
            r["repeat_within_tolerance"] = diff <= REPEAT_TOLERANCE
        ctx.write(rep)
        log(f"E0 repeat: max metric difference {diff:.3g} (tolerance {REPEAT_TOLERANCE:g})")
        del enc
        _free_gpu()
        if second is not None and second_cache is not None:
            enc = second.load()
            for split in ("dev", "heldout"):
                ctx.write(encoder_rows(second, enc, second_cache, split, second.row_e0, seed=None))
            del enc
            _free_gpu()
        log(f"E0 rows written, {time.time() - t0:.0f}s")

    if "e1" in stages and base.row_e1:
        from tasterank.train import load_lora, train

        pairs, dropped = drop_pairs_without_images([dataclasses.replace(p, history=p.history[:n_last]) for p in pairs_all], base_cache, n_last)
        batch = e1_batch_size(e1cfg, n_last)
        log(f"E1: {len(train_customers)} train customers, {len(pairs)} pairs ({dropped} dropped for a missing image), batch {batch}")
        e1_heldout: dict[int, float] = {}
        e1_dev: dict[int, float] = {}
        for seed in seeds:
            cfg = TrainConfig(
                out_dir=paths.work / "lora" / f"{ctx.run_id}-seed{seed}",
                train_cutoff=_date(rule["train_cutoff"]),
                seed=int(seed),
                steps=int(e1cfg["steps"]),
                batch_size=batch,
                lr=float(e1cfg["lr"]),
                rank=int(e1cfg["rank"]),
                alpha=float(e1cfg["alpha"]),
                dropout=float(e1cfg["dropout"]),
                temperature=float(e1cfg["temperature"]),
                n_last=n_last,
                half_life_days=math.inf if half_life is None else float(half_life),
            )
            enc = base.load()
            if e1cfg.get("grad_checkpointing") and hasattr(enc, "set_grad_checkpointing"):
                enc.set_grad_checkpointing(True)
            peak = _reset_peak()
            ts = time.time()
            result = train(enc, pairs, base_cache.source, cfg)
            train_s = time.time() - ts
            peak = _peak_gib() if peak else None
            del enc
            _free_gpu()
            enc = load_lora(base.load(), result.checkpoint, cfg)
            train_meta = {
                "steps": cfg.steps, "batch_size": cfg.batch_size, "lr": cfg.lr, "lora_rank": cfg.rank, "lora_alpha": cfg.alpha,
                "temperature": cfg.temperature, "n_pairs": len(pairs), "pairs_dropped_missing_image": dropped,
                "train_customers": len(train_customers), "loss_first": result.losses[0], "loss_last": result.losses[-1],
                "loss_last100_mean": sum(result.losses[-100:]) / len(result.losses[-100:]), "train_seconds": round(train_s, 1),
                "seconds_per_step": round(train_s / cfg.steps, 3), "peak_gpu_mem_gib": peak,
                "trainable_params": sum(v.numel() for v in result.lora_state.values()),
            }
            for split in ("dev", "heldout"):
                rows = encoder_rows(base, enc, base_cache, split, base.row_e1, seed=int(seed), train=train_meta)
                ctx.write(rows)
                overall = next(r for r in rows if r["table"] == "overall")["metrics"]["ndcg@10"]
                (e1_heldout if split == "heldout" else e1_dev)[int(seed)] = overall
            log(f"E1 seed {seed}: loss {train_meta['loss_first']:.3f} -> {train_meta['loss_last100_mean']:.3f}, "
                f"dev ndcg@10 {e1_dev[int(seed)]:.5f}, {train_s:.0f}s training")
            del enc
            _free_gpu()
        e0_by_split = {r["split"]: r["metrics"]["ndcg@10"] for r in ctx.rows if r["row"] == base.row_e0 and r["table"] == "overall"
                       and r.get("stage") is None and r.get("repeat_of") is None}
        lift_rows = []
        for split, e1 in (("dev", e1_dev), ("heldout", e1_heldout)):
            if sorted(e1) == list(ev.LIFT_SEEDS) and split in e0_by_split:
                lb = ev.lift_band(e1, e0_by_split[split])
                lift_rows.append(ev.make_row(
                    **ctx.common(split), row=base.row_e1, model_revision=base.revision, quant=base.quant, seed=None, n_last=n_last,
                    half_life_days=half_life, missing_images=0, table="overall", n_customers=len(splits[split].customers),
                    metrics={"ndcg@10_lift_mean": lb["mean"], "ndcg@10_lift_min": lb["min"], "ndcg@10_lift_max": lb["max"],
                             "ndcg@10_e0": e0_by_split[split]},
                    stage="lift_band", verdict=lb["verdict"], n_seeds=lb["n"],
                ))
        if lift_rows:
            ctx.write(lift_rows)
    log(f"run {ctx.run_id} done in {time.time() - t0:.0f}s: {len(ctx.rows)} rows in {ctx.out_file}")
    return ctx


def _date(s: str):
    from tasterank.seams import parse_date

    return parse_date(s)


# ---------------------------------------------------------------- summary


def _fmt(x: Any) -> str:
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def summary_markdown(rows: Sequence[Mapping[str, Any]], base_e0: str = "E0-siglip", base_e1: str = "E1-siglip") -> str:
    """Overall tables, the lift band, the E0 repeat check, per-category heldout tables and the failure cases."""
    from tasterank import eval as ev

    if not rows:
        return "No rows.\n"
    run_id = rows[0]["run_id"]
    main = [r for r in rows if r.get("stage") is None and r.get("repeat_of") is None]
    out = [f"## taste-rank box run {run_id}", "", f"git `{rows[0]['git_sha'][:12]}`, machine: {rows[0]['machine']}", ""]
    lift = [r for r in rows if r.get("stage") == "lift_band"]
    for r in lift:
        m = r["metrics"]
        out.append(f"- **{r['split']} nDCG@10 lift, {base_e1} minus {base_e0}, five seeds:** mean {m['ndcg@10_lift_mean']:+.4f}, "
                   f"band [{m['ndcg@10_lift_min']:+.4f}, {m['ndcg@10_lift_max']:+.4f}], verdict **{r['verdict']}** (E0 {m['ndcg@10_e0']:.4f})")
    rep = [r for r in rows if r.get("repeat_of") is not None and r["table"] == "overall"]
    for r in rep:
        out.append(f"- E0 repeat ({r['split']}): max metric difference {r['repeat_max_abs_diff']:.3g}, within 1e-6: {r['repeat_within_tolerance']}")
    grid = [r for r in rows if r.get("stage") == "history_grid"]
    for r in grid:
        if r.get("grid_choice"):
            out.append(f"- History chosen on dev with E0 only: n_last {r['n_last']}, half-life {r['half_life_days']} "
                       f"(dev nDCG@10 {r['metrics']['ndcg@10']:.4f}, {len(grid)} grid points)")
    out += ["", "### Overall", "", "| row | split | seed | n | nDCG@10 | recall@12 | recall@50 | pool coverage | missing images |", "|---|---|---|---|---|---|---|---|---|"]
    for r in main:
        if r["table"] == "overall":
            m = r["metrics"]
            out.append(f"| {r['row']} | {r['split']} | {_fmt(r.get('seed'))} | {r['n_customers']} | {m['ndcg@10']:.4f} | {m['recall@12']:.4f} | "
                       f"{m['recall@50']:.4f} | {m['pool_coverage']:.4f} | {r.get('missing_images', '')} |")
    # Per-category heldout: E1 band mean against E0 and POP, and the failure cases.
    held = [r for r in main if r["split"] == "heldout" and r["table"] != "overall"]
    by: dict[str, dict[str, Any]] = {}
    for r in held:
        t = by.setdefault(r["table"], {"n": r["n_customers"], "e1": []})
        if r["row"] == base_e1:
            t["e1"].append(r["metrics"]["ndcg@10"])
        elif r["row"] in ("POP", base_e0):
            t[r["row"]] = r["metrics"]["ndcg@10"]
    if by:
        out += ["", "### Heldout per product group (nDCG@10)", "", f"| group | n | POP | {base_e0} | {base_e1} mean [min, max] | E1 − E0 |", "|---|---|---|---|---|---|"]
        fails = []
        for g in sorted(by, key=lambda g: -by[g]["n"]):
            t = by[g]
            e1 = ev.band(t["e1"]) if t["e1"] else None
            e0, pop = t.get(base_e0), t.get("POP")
            e1s = f"{e1['mean']:.4f} [{e1['min']:.4f}, {e1['max']:.4f}]" if e1 else ""
            d = f"{e1['mean'] - e0:+.4f}" if e1 and e0 is not None else ""
            out.append(f"| {g} | {t['n']} | {_fmt(pop)} | {_fmt(e0)} | {e1s} | {d} |")
            if e1 and ((e0 is not None and e1["max"] < e0) or (pop is not None and e1["max"] < pop)):
                fails.append((g, t["n"], e1, e0, pop))
        out += ["", "### Failure cases (heldout groups where every E1 seed is below E0 or POP)", ""]
        if fails:
            for g, n, e1, e0, pop in fails:
                out.append(f"- {g} ({n} customers): E1 max {e1['max']:.4f}, E0 {_fmt(e0)}, POP {_fmt(pop)}")
        else:
            out.append("- none in this run")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- main


def default_specs(paths: BoxPaths, device: str) -> tuple[ModelSpec, ModelSpec]:
    from tasterank.encoders import CROP, FCLIP_ID, SIGLIP_ID, SQUASH, load_fclip, load_siglip, revision_of

    quant = "bf16 autocast (weights fp32)"
    base = ModelSpec("E0-siglip", lambda: load_siglip(paths.siglip, device), revision_of(SIGLIP_ID), SQUASH, 224, "E1-siglip", quant)
    second = ModelSpec("E0-fclip", lambda: load_fclip(paths.fclip, device), revision_of(FCLIP_ID), CROP, 224, None, quant)
    return base, second


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tasterank.box")
    ap.add_argument("--root", type=Path)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--work", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--stages", default="pop,grid,e0,e1", help="comma list of pop, grid, e0, e1")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args(argv)
    paths = default_paths(args.root, args.data, args.work, args.out)
    problems = preflight(paths)
    if problems:
        print("box not provisioned; nothing ran:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return EXIT_NOT_PROVISIONED
    if args.preflight_only:
        print("preflight ok")
        return 0
    stages = [s for s in args.stages.split(",") if s]
    bad = sorted(set(stages) - {"pop", "grid", "e0", "e1"})
    if bad:
        print(f"unknown stage(s) {bad}", file=sys.stderr)
        return 2
    base, second = default_specs(paths, "cuda")
    ctx = run_box(paths, base, second, load_e1_config(), stages=stages)
    (paths.out / "summary.md").write_text(summary_markdown(ctx.rows))
    print((paths.out / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
