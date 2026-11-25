"""P4 cross-lane tests: data (P3a) -> train (P3b) -> eval (P3c), and the contracts between them.

CPU only, against the synthetic fixture `fixtures/shop_like` and `tasterank.tiny`.
Every number here comes from synthetic data and measures nothing.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
from pathlib import Path

import polars as pl
import pytest
import torch

from tasterank import data, train
from tasterank import eval as ev
from tasterank.eval import run as er
from tasterank.seams import REPO_ROOT, TrainConfig, load_split_rule
from tasterank.tiny import tiny_encoder

FIX = REPO_ROOT / "fixtures" / "shop_like"
RULE = load_split_rule(FIX / "split.json")
CUTOFF = dt.date.fromisoformat(RULE["train_cutoff"])
SCHEMA = json.loads((REPO_ROOT / "contracts" / "results.schema.json").read_text())


@pytest.fixture(scope="module")
def tx():
    t = data.load_transactions(FIX / "transactions_train.csv")
    data.check_anchor(t, RULE)
    return t


@pytest.fixture(scope="module")
def groups():
    a = data.load_articles(FIX / "articles.csv")
    return dict(a.select("article_id", "product_group_name").iter_rows())


def _cfg(out_dir: Path, **kw) -> TrainConfig:
    base = dict(out_dir=out_dir, train_cutoff=CUTOFF, seed=0, steps=4, batch_size=6, lr=5e-3, rank=4, alpha=8.0, n_last=5, half_life_days=30.0, temperature=0.1)
    base.update(kw)
    return TrainConfig(**base)


# ---------------------------------------------------------------- leakage across lanes


def test_training_pairs_never_touch_eval_customers_or_windows(tx):
    pairs = data.training_pairs(tx, RULE, 5)
    assert pairs, "the fixture gives no training pairs"
    dev = data.build_split(tx, RULE, "dev")
    held = data.build_split(tx, RULE, "heldout")
    pair_customers = {p.customer_id for p in pairs}
    assert not pair_customers & set(dev.customers)
    assert not pair_customers & set(held.customers)
    assert all(data.assign_split(c, RULE) == "train" for c in pair_customers)
    assert all(p.target_date < CUTOFF for p in pairs)
    assert all(h.t_dat < p.target_date for p in pairs for h in p.history)


def test_iter_training_pairs_is_training_pairs(tx):
    assert list(data.iter_training_pairs(tx, RULE, 5)) == data.training_pairs(tx, RULE, 5)


def test_dev_split_and_its_rankings_ignore_heldout_days(tx, groups):
    # Rewrite every heldout-window purchase: the dev split, its rankings and its rows must not move.
    start = dt.date.fromisoformat(RULE["windows"]["heldout"]["start"])
    arts = sorted(groups)
    shifted = tx.with_columns(
        pl.when(pl.col("t_dat") >= start).then(pl.lit(arts[0])).otherwise(pl.col("article_id")).alias("article_id")
    )
    a, b = data.build_split(tx, RULE, "dev"), data.build_split(shifted, RULE, "dev")
    assert (a.customers, a.pool, a.pop_counts, a.relevant) == (b.customers, b.pool, b.pop_counts, b.relevant)
    assert er.evaluate(a, er.pop_rankings(a), groups) == er.evaluate(b, er.pop_rankings(b), groups)


# ---------------------------------------------------------------- data -> train -> eval


@pytest.fixture(scope="module")
def trained(tx, tmp_path_factory):
    out = tmp_path_factory.mktemp("lora")
    pairs = data.training_pairs(tx, RULE, 5)
    images = er.image_dir_source(FIX / "images")
    res = train.train(tiny_encoder(er.TINY_SEED), pairs, images, _cfg(out))
    return res, out, images


def test_trained_checkpoint_rebuilds_the_same_encoder(trained):
    res, out, images = trained
    meta = json.loads((out / train.META_NAME).read_text())
    cfg = _cfg(out, rank=meta["rank"], alpha=meta["alpha"], target_modules=tuple(meta["target_modules"]))
    a = train.load_lora(tiny_encoder(er.TINY_SEED), res.checkpoint, cfg)
    b = train.load_lora(tiny_encoder(er.TINY_SEED), res.checkpoint, cfg)
    px = torch.stack([images(x) for x in _some_articles(4)])
    with torch.no_grad():
        ea, eb, e0 = a(px), b(px), tiny_encoder(er.TINY_SEED)(px)
    assert torch.equal(ea, eb)
    assert not torch.allclose(ea, e0), "the LoRA checkpoint did not change the encoder"
    assert all(math_finite(x) for x in res.losses)


def _some_articles(n: int) -> list[str]:
    return sorted(data.load_articles(FIX / "articles.csv")["article_id"].to_list())[:n]


def math_finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def test_e1_rows_from_a_trained_checkpoint_pass_the_contract(trained, tx, groups, tmp_path):
    res, out, images = trained
    meta = json.loads((out / train.META_NAME).read_text())
    cfg = _cfg(out, rank=meta["rank"], alpha=meta["alpha"], target_modules=tuple(meta["target_modules"]))
    enc = train.load_lora(tiny_encoder(er.TINY_SEED), res.checkpoint, cfg)
    split = data.build_split(tx, RULE, "dev")
    rankings, counts = er.encoder_rankings(split, enc, images, meta["n_last"], meta["half_life_days"])
    assert set(rankings) == set(split.customers)
    assert all(sorted(r) == split.pool for r in rankings.values()), "every customer ranks the whole pool once"
    rows = [
        ev.make_row(row="E1-tiny", split="dev", model_revision="tasterank.tiny + LoRA (test)", quant=None, runtime=f"torch {torch.__version__}",
                    seed=meta["seed"], n_last=meta["n_last"], half_life_days=meta["half_life_days"], missing_images=counts["missing_images"],
                    dataset="fixtures/shop_like (synthetic)", **t)
        for t in er.evaluate(split, rankings, groups)
    ]
    path = tmp_path / "e1.jsonl"
    ev.write_rows(path, rows)
    back = [json.loads(l) for l in path.read_text().splitlines()]
    assert {r["table"] for r in back} >= {"overall"}
    for r in back:
        _check_schema(r)


def test_eval_customer_vector_is_the_train_lane_vector(tx):
    # encoder_rankings must build the vector with train.customer_vector at the window start (eval.md).
    split = data.build_split(tx, RULE, "dev")
    images = er.image_dir_source(FIX / "images")
    enc = tiny_encoder(er.TINY_SEED)
    rankings, _ = er.encoder_rankings(split, enc, images, 5, 30.0)
    hist = data.last_n_history(split.history, 5)
    embs, _ = er.embed_articles(enc, split.pool, images)
    cand = torch.stack([embs[a] for a in split.pool])
    c = split.customers[0]
    items = hist[c]
    with torch.no_grad():
        e = enc(torch.stack([images(p.article_id) for p in items])).float()
    v = train.customer_vector(e, torch.tensor([(split.window_start - p.t_dat).days for p in items], dtype=e.dtype), 30.0)
    assert rankings[c] == ev.rank(v, split.pool, cand, len(split.pool))


# ---------------------------------------------------------------- the fixture pipeline end to end


def _check_schema(row: dict) -> None:
    """The parts of contracts/results.schema.json a row can break, checked without a schema library."""
    missing = [k for k in SCHEMA["required"] if k not in row]
    assert not missing, missing
    types = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "null": type(None)}
    for k, spec in SCHEMA["properties"].items():
        if k not in row:
            continue
        want = spec["type"] if isinstance(spec["type"], list) else [spec["type"]]
        v = row[k]
        ok = any(isinstance(v, types[t]) and not (t in ("integer", "number") and isinstance(v, bool)) for t in want)
        assert ok, (k, v)
        if "enum" in spec:
            assert v in spec["enum"], (k, v)
        if "pattern" in spec and isinstance(v, str):
            assert re.fullmatch(spec["pattern"].strip("^$"), v), (k, v)
        if "minimum" in spec and isinstance(v, (int, float)):
            assert v >= spec["minimum"], (k, v)
    for clause in SCHEMA["not"]["anyOf"]:
        assert not all(k in row for k in clause["required"]), clause
    extra = set(row) - set(SCHEMA["properties"])
    assert not extra, f"row fields not declared in the schema: {sorted(extra)}"


def test_eval_sh_fixture_rows_match_the_schema_exactly(tmp_path):
    p = subprocess.run(["bash", "scripts/eval.sh", "--fixture", str(tmp_path)], cwd=REPO_ROOT, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stderr[-500:]
    rows = [json.loads(l) for f in tmp_path.glob("fixture-*.jsonl") for l in f.read_text().splitlines()]
    assert {r["row"] for r in rows} == {"POP", "E0-tiny"}
    assert len({r["run_id"] for r in rows}) == 1
    for r in rows:
        _check_schema(r)
    assert (tmp_path / "summary.md").is_file()


@pytest.mark.parametrize("sub", ["results", "results/fixture-run"])
def test_eval_sh_refuses_fixture_rows_under_results(sub):
    target = REPO_ROOT / sub
    existed = target.exists()
    before = sorted((REPO_ROOT / "results").rglob("*"))
    p = subprocess.run(["bash", "scripts/eval.sh", "--fixture", sub], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    assert p.returncode == 2 and "results/ holds box run rows only" in p.stderr, p.stderr[-300:]
    assert sorted((REPO_ROOT / "results").rglob("*")) == before
    assert target.exists() == existed


def test_image_source_refuses_ids_that_name_other_paths(tmp_path):
    (tmp_path / "img").mkdir()
    load = er.image_dir_source(tmp_path / "img")
    for bad in ["../x", "a/b", "..", "", "0100000000.jpg", "x\x00y"]:
        with pytest.raises(ValueError, match="not \\[0-9A-Za-z_-\\]\\+"):
            load(bad)
    assert load("0100000000") is None


# ---------------------------------------------------------------- contracts list what the lanes ship


def test_every_seam_function_exists():
    text = (REPO_ROOT / "contracts" / "seams.md").read_text()
    sections = {"tasterank.data": data, "tasterank.train": train, "tasterank.eval": ev}
    import tasterank.models as models

    sections["tasterank.models"] = models
    current = None
    names = []
    for line in text.splitlines():
        m = re.match(r"## .*`(tasterank\.\w+)`", line)
        if m:
            current = sections[m.group(1)]
            continue
        m = re.match(r"\| `(\w+)\(", line)
        if m and current is not None:
            names.append((current, m.group(1)))
    assert len(names) >= 30
    for mod, name in names:
        assert callable(getattr(mod, name, None)), f"{mod.__name__}.{name} is in contracts/seams.md but not shipped"


def test_rows_are_rejected_with_wrong_types():
    base = ev.make_row(row="POP", split="dev", table="overall", metrics={"ndcg@10": 0.0}, n_customers=1, model_revision=None, quant=None, runtime=None)
    ev.check_row(base)
    for k, v in (("n_customers", 1.5), ("n_customers", True), ("n_customers", -1), ("dataset", 3)):
        with pytest.raises(ValueError, match=k):
            ev.check_row({**base, k: v})
    with pytest.raises(ValueError, match="git_dirty"):
        ev.check_row({**base, "git_dirty": "no"})
