"""P5 box pipeline (tasterank.box), on CPU against the synthetic fixture and tasterank.tiny.

The box runs the same code with the real encoders on the GPU. Every number
here comes from synthetic data and measures nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from tasterank import box, data
from tasterank import eval as ev
from tasterank.eval import run as er
from tasterank.seams import REPO_ROOT, FORBIDDEN_RESULT_KEYS, load_split_rule
from tasterank.tiny import tiny_encoder

FIX = REPO_ROOT / "fixtures" / "shop_like"
RULE_PATH = FIX / "split.json"
RULE = load_split_rule(RULE_PATH)
E1_TEST = {
    "steps": 2, "lr": 5e-3, "rank": 4, "alpha": 8.0, "dropout": 0.0, "temperature": 0.1, "max_images_per_step": 64,
    "batch_size_min": 4, "batch_size_max": 6, "train_customers": 1000, "sample_salt": "test", "grad_checkpointing": True, "embed_batch": 7,
}


@pytest.fixture(scope="module")
def tx():
    return data.load_transactions(FIX / "transactions_train.csv")


@pytest.fixture(scope="module")
def groups():
    a = data.load_articles(FIX / "articles.csv")
    return dict(a.select("article_id", "product_group_name").iter_rows())


def _cache(tmp: Path, drop: set[str] = frozenset()) -> box.ImageCache:
    ids = sorted({p.name[:-4] for p in (FIX / "images").rglob("*.jpg")} - set(drop))
    c = box.build_image_cache(FIX / "images", ids, 16, "squash", tmp, workers=1, log=lambda _m: None)
    return c


def _ctx(tx, groups, tmp: Path) -> box.RunContext:
    splits = {w: data.build_split(tx, RULE, w) for w in ("dev", "heldout")}
    return box.RunContext(RULE, "0" * 64, "0" * 64, splits, {a: g for a, g in groups.items() if g}, tmp / "rows.jsonl", "r", "t", "fixture", None, None)


# ---------------------------------------------------------------- rankings match the full pipeline


@pytest.mark.parametrize("which", ["dev", "heldout"])
@pytest.mark.parametrize("drop", [set(), {"0100000020", "0100000031"}])
def test_truncated_rankings_give_the_same_tables_as_full_rankings(tx, groups, tmp_path, which, drop):
    split = data.build_split(tx, RULE, which)
    enc = tiny_encoder(er.TINY_SEED)
    cache = _cache(tmp_path, drop)
    full_src = er.image_dir_source(FIX / "images")
    src = lambda a: None if a in drop else full_src(a)  # noqa: E731
    full, counts = er.encoder_rankings(split, enc, src, 5, 30.0)
    want = er.evaluate(split, full, groups)

    ctx = _ctx(tx, groups, tmp_path)
    embs = box.split_embeddings(enc, cache, split, 5, 7)
    trunc, missing, no_img = box.rankings_from(ctx, which, embs, 5, 30.0, "cpu")
    got = er.evaluate(split, trunc, groups)
    assert [t["table"] for t in got] == [t["table"] for t in want]
    for g, w in zip(got, want):
        assert g["n_customers"] == w["n_customers"]
        for k in w["metrics"]:
            assert g["metrics"][k] == pytest.approx(w["metrics"][k], abs=1e-12), (g["table"], k)
    assert missing == counts["missing_images"] and no_img == counts["no_history_image_customers"]


def test_truncated_rankings_keep_rank_order_and_group_tops():
    pool = [f"a{i:02d}" for i in range(12)]
    embs = {a: torch.tensor([1.0, float(i)]) for i, a in enumerate(pool) if a != "a03"}
    group = {a: ("x" if i % 3 == 0 else "y") for i, a in enumerate(pool)}
    vec = torch.tensor([[0.0, 1.0]])
    out = box.truncated_rankings(["c"], vec, pool, embs, {"c": {"x"}}, group, k=2)["c"]
    full = ev.rank_many(vec, [a for a in pool if a in embs], torch.stack([embs[a] for a in pool if a in embs]))[0] + ["a03"]
    assert out == [a for a in full if a in set(out)]  # same order as the full ranking
    assert [a for a in out if group[a] == "x"][:2] == [a for a in full if group[a] == "x"][:2]
    assert out[:2] == full[:2]


# ---------------------------------------------------------------- the whole box run on the fixture


@pytest.fixture(scope="module")
def box_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("box")
    paths = box.BoxPaths(root=tmp, data=FIX, work=tmp / "work", out=tmp / "out")
    load = lambda: tiny_encoder(er.TINY_SEED)  # noqa: E731
    base = box.ModelSpec("E0-tiny", load, "tasterank.tiny", "squash", 16, "E1-tiny", None)
    ctx = box.run_box(paths, base, None, E1_TEST, rule_path=RULE_PATH, device="cpu", dataset="fixtures/shop_like (synthetic)", log=lambda _m: None)
    return ctx, paths


def test_box_run_writes_every_stage_and_valid_rows(box_run):
    ctx, paths = box_run
    rows = [json.loads(l) for l in ctx.out_file.read_text().splitlines()]
    assert rows == ctx.rows
    for r in rows:
        ev.check_row(r)
        assert not FORBIDDEN_RESULT_KEYS & set(r)
    names = {(r["row"], r["split"]) for r in rows}
    assert {("POP", "dev"), ("POP", "heldout"), ("E0-tiny", "dev"), ("E0-tiny", "heldout"), ("E1-tiny", "dev"), ("E1-tiny", "heldout")} <= names
    seeds = {r["seed"] for r in rows if r["row"] == "E1-tiny" and r.get("stage") is None}
    assert seeds == {0, 1, 2, 3, 4}
    assert {r["split_rule_sha256"] for r in rows} == {hashlib.sha256(RULE_PATH.read_bytes()).hexdigest()}
    assert {r["split_ids_sha256"] for r in rows} == {data.split_ids_sha256(data.load_transactions(FIX / "transactions_train.csv"), RULE)}


def test_history_grid_runs_on_dev_with_e0_only_and_picks_one_point(box_run):
    ctx, _ = box_run
    grid = [r for r in ctx.rows if r.get("stage") == "history_grid"]
    g = RULE["history_grid"]
    points = {(n, h) for n in g["n_last"] for h in g["half_life_days"]}
    points.add((RULE["history_default"]["n_last"], RULE["history_default"]["half_life_days"]))
    assert {(r["n_last"], r["half_life_days"]) for r in grid} == points and len(grid) == len(points)
    assert (grid[0]["n_last"], grid[0]["half_life_days"]) == (RULE["history_default"]["n_last"], RULE["history_default"]["half_life_days"])
    assert {r["split"] for r in grid} == {"dev"} and {r["row"] for r in grid} == {"E0-tiny"}
    assert sum(r["grid_choice"] for r in grid) == 1
    chosen = next(r for r in grid if r["grid_choice"])
    assert chosen["metrics"]["ndcg@10"] == max(r["metrics"]["ndcg@10"] for r in grid)
    # every later encoder row uses the chosen point; the grid comes before any E1 row
    later = [r for r in ctx.rows if r["row"] != "POP" and r.get("stage") != "history_grid"]
    assert {(r["n_last"], r["half_life_days"]) for r in later} == {(chosen["n_last"], chosen["half_life_days"])}
    first_e1 = next(i for i, r in enumerate(ctx.rows) if r["row"] == "E1-tiny")
    assert max(i for i, r in enumerate(ctx.rows) if r.get("stage") == "history_grid") < first_e1


def test_e0_repeat_matches_and_lift_band_uses_five_seeds(box_run):
    ctx, _ = box_run
    rep = [r for r in ctx.rows if r.get("repeat_of")]
    assert rep and all(r["repeat_within_tolerance"] and r["repeat_max_abs_diff"] <= 1e-6 for r in rep)
    lift = {r["split"]: r for r in ctx.rows if r.get("stage") == "lift_band"}
    assert set(lift) == {"dev", "heldout"}
    e0 = next(r for r in ctx.rows if r["row"] == "E0-tiny" and r["split"] == "heldout" and r["table"] == "overall" and r.get("stage") is None and not r.get("repeat_of"))
    e1 = {r["seed"]: r["metrics"]["ndcg@10"] for r in ctx.rows if r["row"] == "E1-tiny" and r["split"] == "heldout" and r["table"] == "overall" and r.get("stage") is None}
    want = ev.lift_band(e1, e0["metrics"]["ndcg@10"])
    m = lift["heldout"]["metrics"]
    assert (m["ndcg@10_lift_mean"], m["ndcg@10_lift_min"], m["ndcg@10_lift_max"]) == pytest.approx((want["mean"], want["min"], want["max"]))
    assert lift["heldout"]["verdict"] == want["verdict"]


def test_lora_weights_and_caches_stay_in_the_work_folder(box_run):
    ctx, paths = box_run
    ckpts = sorted(paths.work.rglob("lora.pt"))
    assert len(ckpts) == 5 and all(paths.work in c.parents for c in ckpts)
    assert not list(REPO_ROOT.glob("**/lora.pt"))
    assert sorted(p.name for p in paths.out.iterdir()) == [ctx.out_file.name]


def test_e1_rows_train_on_train_customers_only(box_run, tx):
    ctx, _ = box_run
    custs = box.sample_train_customers(tx, RULE, 1000, "test")
    assert custs and all(data.assign_split(c, RULE) == "train" for c in custs)
    e1 = [r for r in ctx.rows if r["row"] == "E1-tiny" and r.get("stage") is None]
    assert {r["train"]["train_customers"] for r in e1} == {len(custs)}


def test_summary_renders_the_lift_and_tables(box_run):
    ctx, _ = box_run
    md = box.summary_markdown(ctx.rows, "E0-tiny", "E1-tiny")
    assert "heldout nDCG@10 lift" in md and "### Overall" in md and "Failure cases" in md and "E0 repeat" in md
    for r in ctx.rows:
        assert "customer" not in json.dumps(r).replace("n_customers", "").replace("train_customers", "").replace("no_history_image_customers", "")


# ---------------------------------------------------------------- smaller pieces


def test_e1_batch_size_fits_the_image_budget():
    cfg = {"max_images_per_step": 704, "batch_size_min": 8, "batch_size_max": 32}
    assert box.e1_batch_size(cfg, 20) == 32  # 704 // 21 = 33, capped
    assert box.e1_batch_size(cfg, 50) == 13
    assert box.e1_batch_size(cfg, 200) == 8


def test_e1_config_in_the_repo_is_complete():
    cfg = box.load_e1_config()
    assert cfg["steps"] > 0 and cfg["batch_size_min"] >= 2


def test_sample_train_customers_is_deterministic_and_salted(tx):
    a = box.sample_train_customers(tx, RULE, 5, "s1")
    assert a == box.sample_train_customers(tx, RULE, 5, "s1") and len(a) == 5
    assert a != box.sample_train_customers(tx, RULE, 5, "s2")


def test_drop_pairs_without_images(tx, tmp_path):
    pairs = data.training_pairs(tx, RULE, 5)
    gone = pairs[0].target_article
    kept, dropped = box.drop_pairs_without_images(pairs, _cache(tmp_path, {gone}), 5)
    assert dropped == len(pairs) - len(kept) > 0
    assert all(p.target_article != gone and all(h.article_id != gone for h in p.history) for p in kept)


def test_image_cache_refuses_ids_that_name_other_paths(tmp_path):
    with pytest.raises(ValueError, match="not"):
        box.build_image_cache(FIX / "images", ["../etc"], 16, "squash", tmp_path, workers=1)


def test_image_cache_is_reused_and_holds_prepared_pixels(tmp_path):
    from tasterank.encoders import prepare_image

    c1 = _cache(tmp_path)
    c2 = _cache(tmp_path)
    a = sorted(c1.index)[0]
    assert np.array_equal(c1.array[c1.index[a]], c2.array[c2.index[a]])
    assert np.array_equal(c1.array[c1.index[a]], prepare_image(box.image_path(FIX / "images", a), 16, "squash"))
    assert c1.source(a).shape == (3, 16, 16) and float(c1.source(a).max()) <= 1.0


def test_prepare_image_geometries(tmp_path):
    from PIL import Image

    from tasterank.encoders import prepare_image

    img = np.zeros((40, 20, 3), dtype=np.uint8)  # tall: 20 wide, 40 high
    img[:10] = 255  # white top quarter
    p = tmp_path / "x.png"
    Image.fromarray(img).save(p)
    sq = prepare_image(p, 8, "squash")
    cr = prepare_image(p, 8, "crop")
    assert sq.shape == cr.shape == (3, 8, 8)
    assert sq[:, 0].mean() > 200  # squash keeps the white top
    assert cr[:, 0].mean() < 60  # the centre crop cuts it off
    with pytest.raises(ValueError):
        prepare_image(p, 8, "pad")


def test_normalized_tower_scales_pixels():
    from tasterank.encoders import NormalizedTower

    seen = {}

    class Id(torch.nn.Module):
        def forward(self, x):
            seen["x"] = x
            return x.flatten(1)

    t = NormalizedTower(Id(), 2, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), "squash")
    out = t(torch.ones(1, 3, 2, 2))
    assert torch.allclose(seen["x"], torch.ones(1, 3, 2, 2)) and out.dtype == torch.float32


# ---------------------------------------------------------------- preflight and eval.sh


def test_preflight_lists_what_an_empty_box_lacks(tmp_path):
    paths = box.BoxPaths(root=tmp_path, data=tmp_path / "hm", work=tmp_path / "w", out=tmp_path / "o")
    problems = box.preflight(paths, need_cuda=False)
    text = "\n".join(problems)
    assert "transactions_train.csv" in text and "articles.csv" in text and "images" in text
    assert "Marqo/marqo-fashionSigLIP" in text or "marqo-fashionSigLIP" in text
    assert "fashion-clip" in text


def test_preflight_refuses_a_work_folder_in_the_repo(tmp_path):
    paths = box.BoxPaths(root=tmp_path, data=FIX, work=REPO_ROOT / "tmp-work", out=tmp_path / "o")
    assert any("inside the repo" in p for p in box.preflight(paths, need_cuda=False))


def test_eval_sh_on_an_unprovisioned_box_exits_3_and_writes_nothing(tmp_path):
    env = {**os.environ, "FORGE_PHASE": "P5", "FORGE_TMP": str(tmp_path / "tmp"), "TASTE_RANK_ROOT": str(tmp_path / "root")}
    before = sorted(p.name for p in (REPO_ROOT / "results").iterdir())
    p = subprocess.run(["bash", "scripts/eval.sh"], cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 3, p.stderr
    assert "box not provisioned" in p.stderr and "transactions_train.csv" in p.stderr
    assert sorted(p.name for p in (REPO_ROOT / "results").iterdir()) == before


def test_eval_sh_outside_p5_is_still_not_an_eval_phase():
    env = {k: v for k, v in os.environ.items() if k not in ("FORGE_PHASE", "TASTE_RANK_BOX")}
    p = subprocess.run(["bash", "scripts/eval.sh"], cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0 and "not an eval phase" in p.stdout
