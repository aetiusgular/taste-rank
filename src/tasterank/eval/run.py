"""The eval pipeline behind scripts/eval.sh.

    python -m tasterank.eval.run --fixture <out_dir>

runs POP and E0-tiny on the fixture's dev split (fixtures/shop_like, rule
fixtures/shop_like/split.json) and appends rows to <out_dir>/fixture-<run_id>.jsonl,
plus <out_dir>/summary.md. Fixture runs never build a heldout split
(contracts/eval.md, "Results rows"). The box path (real encoders, E1, heldout)
is wired in P5 on top of `evaluate`.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tasterank import eval as ev
from tasterank.seams import REPO_ROOT, EvalSplit, ImageSource, load_split_rule

FIXTURE_DIR = REPO_ROOT / "fixtures" / "shop_like"
TINY_SEED = 1234
EMBED_BATCH = 256


def image_dir_source(root: Path) -> Callable[[str], Any | None]:
    """article_id -> float tensor [3, H, W] in [0, 1] from <root>/<first 3 digits>/<id>.jpg, or None if absent."""
    import numpy as np
    import torch
    from PIL import Image

    def load(article_id: str):
        p = root / article_id[:3] / f"{article_id}.jpg"
        if not p.is_file():
            return None
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    return load


def embed_articles(encoder: Any, article_ids: Sequence[str], images: Callable[[str], Any | None]) -> tuple[dict[str, Any], int]:
    """article_id -> embedding [D] for every id with an image; also the count of ids without one."""
    import torch

    have, pixels, missing = [], [], 0
    for a in dict.fromkeys(article_ids):
        img = images(a)
        if img is None:
            missing += 1
            continue
        have.append(a)
        pixels.append(img)
    out: dict[str, Any] = {}
    with torch.no_grad():
        for i in range(0, len(have), EMBED_BATCH):
            embs = encoder(torch.stack(pixels[i : i + EMBED_BATCH])).float()
            out.update(zip(have[i : i + EMBED_BATCH], embs))
    return out, missing


def encoder_rankings(
    split: EvalSplit,
    encoder: Any,
    images: Callable[[str], Any | None],
    n_last: int,
    half_life_days: float | None,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Each eligible customer's full pool ranking under `encoder` (contracts/eval.md, "Customer vector and score").

    Pool articles without an image rank after every scored one, by id. History
    purchases without an image are skipped; a customer left with no history
    image gets a zero vector, so every score ties and the pool ranks by id.
    """
    import math

    import torch

    from tasterank.data import last_n_history
    from tasterank.train import customer_vector

    hist = last_n_history(split.history, n_last)
    need = list(split.pool) + [p.article_id for c in split.customers for p in hist.get(c, [])]
    # missing: distinct articles (pool and history together) with no image, each counted once
    embs, missing = embed_articles(encoder, need, images)
    scored = [a for a in split.pool if a in embs]
    unscored = [a for a in split.pool if a not in embs]
    if not scored:
        raise ValueError("no pool article has an image")
    dim = embs[scored[0]].shape[0]
    hl = math.inf if half_life_days is None else half_life_days
    vecs, no_image_customers = [], 0
    for c in split.customers:
        items = [p for p in hist.get(c, []) if p.article_id in embs]
        if not items:
            no_image_customers += 1
            vecs.append(torch.zeros(dim))
            continue
        e = torch.stack([embs[p.article_id] for p in items])
        ages = torch.tensor([(split.window_start - p.t_dat).days for p in items], dtype=e.dtype)
        vecs.append(customer_vector(e, ages, hl))
    cand = torch.stack([embs[a] for a in scored])
    ranked = ev.rank_many(torch.stack(vecs), scored, cand, None) if vecs else []
    tail = sorted(unscored)
    rankings = {c: r + tail for c, r in zip(split.customers, ranked)}
    return rankings, {"missing_images": missing, "no_history_image_customers": no_image_customers}


def pop_rankings(split: EvalSplit) -> dict[str, list[str]]:
    """POP over the whole pool (articles with no POP-week sale last, by id); the same list for everyone."""
    full = ev.pop_ranking({a: split.pop_counts.get(a, 0) for a in split.pool}, len(split.pool))
    return {c: full for c in split.customers}


def evaluate(
    split: EvalSplit,
    rankings: Mapping[str, Sequence[str]],
    article_group: Mapping[str, str],
) -> list[dict[str, Any]]:
    """[{table, metrics, n_customers}] for the overall table and each product group."""
    relevant = {c: split.relevant[c] for c in split.customers}
    overall = ev.table_metrics(dict(rankings), relevant)
    n = overall.pop("n_customers")
    out = [{"table": "overall", "metrics": {**overall, "pool_coverage": ev.pool_coverage(relevant, split.pool)}, "n_customers": n}]
    for g, m in ev.category_metrics(rankings, relevant, article_group).items():
        m = dict(m)
        n_g = m.pop("n_customers")
        rel_g = {c: [a for a in arts if article_group.get(a) == g] for c, arts in relevant.items()}
        rel_g = {c: arts for c, arts in rel_g.items() if arts}
        pool_g = [a for a in split.pool if article_group.get(a) == g]
        out.append({"table": g, "metrics": {**m, "pool_coverage": ev.pool_coverage(rel_g, pool_g)}, "n_customers": n_g})
    return out


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summary_md(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| row | split | table | n | nDCG@10 | recall@12 | recall@50 | pool coverage |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        m = r["metrics"]
        lines.append(
            f"| {r['row']} | {r['split']} | {r['table']} | {r['n_customers']} | {m['ndcg@10']:.4f} | {m['recall@12']:.4f} | {m['recall@50']:.4f} | {m['pool_coverage']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def run_fixture(out_dir: Path, fixture: Path = FIXTURE_DIR) -> Path:
    """POP and E0-tiny on the fixture's dev split; returns the results file written."""
    import torch

    from tasterank.data import build_split, check_anchor, load_articles, load_transactions, split_ids_sha256
    from tasterank.tiny import tiny_encoder

    rule_path = fixture / "split.json"
    rule = load_split_rule(rule_path)
    tx = load_transactions(fixture / "transactions_train.csv")
    check_anchor(tx, rule)
    articles = load_articles(fixture / "articles.csv")
    group = dict(articles.select("article_id", "product_group_name").iter_rows())
    split = build_split(tx, rule, "dev")
    n_last = int(rule["history_default"]["n_last"])
    half_life = rule["history_default"]["half_life_days"]

    run_id = ev.new_run_id()
    common = {
        "run_id": run_id,
        "split": "dev",
        "quant": None,
        "split_rule_sha256": _sha256_file(rule_path),
        "split_ids_sha256": split_ids_sha256(tx, rule),
        "cold_start": split.cold_start,
        "gpu": None,
        "gpu_power_limit_w": None,
        "dataset": "fixtures/shop_like (synthetic)",
    }
    rows: list[dict[str, Any]] = []
    for t in evaluate(split, pop_rankings(split), group):
        rows.append(ev.make_row(**common, row="POP", model_revision=None, runtime=None, seed=None, n_last=None, half_life_days=None, missing_images=0, **t))

    encoder = tiny_encoder(TINY_SEED)
    rankings, counts = encoder_rankings(split, encoder, image_dir_source(fixture / "images"), n_last, half_life)
    for t in evaluate(split, rankings, group):
        rows.append(
            ev.make_row(
                **common,
                row="E0-tiny",
                model_revision=f"tasterank.tiny seed {TINY_SEED}",
                runtime=f"torch {torch.__version__}",
                seed=None,
                n_last=n_last,
                half_life_days=half_life,
                missing_images=counts["missing_images"],
                **t,
            )
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"fixture-{run_id}.jsonl"
    ev.write_rows(path, rows)
    (out_dir / "summary.md").write_text(f"Fixture run {run_id} (synthetic data, dev split only)\n\n" + summary_md(rows))
    return path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tasterank.eval.run")
    ap.add_argument("--fixture", metavar="OUT_DIR", type=Path, required=True, help="run POP and E0-tiny on the fixture dev split")
    args = ap.parse_args(argv)
    path = run_fixture(args.fixture)
    print(f"wrote {path}")
    print((args.fixture / "summary.md").read_text(), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
