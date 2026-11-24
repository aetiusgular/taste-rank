"""Lane P3c tests beyond the red tests: edge cases, refusals, and an independent
re-computation of the fixture run's POP numbers from the CSVs.

CPU only, against tasterank.tiny and the synthetic fixture.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import subprocess

import pytest
import torch

from tasterank import eval as ev
from tasterank.eval import run as er
from tasterank.seams import FORBIDDEN_RESULT_KEYS, REPO_ROOT, EvalSplit

FIX = REPO_ROOT / "fixtures" / "shop_like"


def _row(**kw):
    base = dict(row="POP", split="dev", table="overall", metrics={"ndcg@10": 0.1}, n_customers=1, model_revision=None, quant=None, runtime=None)
    base.update(kw)
    return ev.make_row(**base)


# ---------------------------------------------------------------- ranking


def test_rank_k_larger_than_pool_and_zero():
    ids = ["b", "a"]
    embs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert ev.rank(torch.tensor([0.0, 1.0]), ids, embs, 10) == ["a", "b"]
    assert ev.rank(torch.tensor([0.0, 1.0]), ids, embs, 0) == []


def test_rank_ignores_vector_scale():
    ids = ["x", "y", "z"]
    embs = torch.tensor([[1.0, 0.1], [0.2, 1.0], [5.0, 4.0]])
    a = ev.rank(torch.tensor([1.0, 0.5]), ids, embs, 3)
    b = ev.rank(torch.tensor([10.0, 5.0]), ids, embs * 7, 3)
    assert a == b


def test_rank_refuses_bad_shapes_and_repeated_ids():
    embs = torch.eye(2)
    with pytest.raises(ValueError, match="candidate ids"):
        ev.rank(torch.ones(2), ["a"], embs, 1)
    with pytest.raises(ValueError, match="repeat"):
        ev.rank(torch.ones(2), ["a", "a"], embs, 1)
    with pytest.raises(ValueError, match="do not match"):
        ev.rank(torch.ones(3), ["a", "b"], embs, 1)
    with pytest.raises(ValueError, match="NaN"):
        ev.rank(torch.ones(2), ["a", "b"], torch.tensor([[1.0, 0.0], [float("nan"), 0.0]]), 1)


def test_zero_customer_vector_ranks_by_id():
    ids = ["c", "a", "b"]
    assert ev.rank(torch.zeros(2), ids, torch.randn(3, 2), 3) == ["a", "b", "c"]


def test_rank_many_matches_rank():
    g = torch.Generator().manual_seed(0)
    ids = [f"{i:04d}" for i in range(30)][::-1]
    embs = torch.randn(30, 8, generator=g)
    vecs = torch.randn(5, 8, generator=g)
    many = ev.rank_many(vecs, ids, embs, 12)
    assert many == [ev.rank(v, ids, embs, 12) for v in vecs]


def test_pop_ranking_k_and_negative():
    assert ev.pop_ranking({"a": 1}, 5) == ["a"]
    with pytest.raises(ValueError):
        ev.pop_ranking({"a": 1}, -1)


# ---------------------------------------------------------------- metrics


def test_metrics_refuse_empty_relevant():
    with pytest.raises(ValueError, match="empty"):
        ev.ndcg_at_k(["a"], [], 10)
    with pytest.raises(ValueError, match="empty"):
        ev.recall_at_k(["a"], set(), 12)


def test_repeated_ranked_item_counts_once():
    assert ev.ndcg_at_k(["a", "a"], {"a", "b"}, 10) == pytest.approx(1 / (1 + 1 / math.log2(3)))
    assert ev.recall_at_k(["a", "a"], {"a", "b"}, 12) == 0.5


def test_short_ranking():
    assert ev.ndcg_at_k([], {"a"}, 10) == 0.0
    assert ev.recall_at_k(["a"], {"a"}, 50) == 1.0


def test_pool_coverage_refuses_nothing():
    with pytest.raises(ValueError):
        ev.pool_coverage({}, ["a"])


def test_table_metrics_needs_matching_customers():
    with pytest.raises(ValueError, match="same customers"):
        ev.table_metrics({"c1": ["a"]}, {"c2": {"a"}})


def test_category_skips_ungrouped_articles():
    rankings = {"c1": ["t1", "u1"]}
    relevant = {"c1": {"u1", "t1"}}
    table = ev.category_metrics(rankings, relevant, {"t1": "Tops"})
    assert set(table) == {"Tops"}
    assert table["Tops"]["ndcg@10"] == 1.0 and table["Tops"]["n_customers"] == 1


# ---------------------------------------------------------------- bands


def test_band_refuses_empty_and_nan():
    with pytest.raises(ValueError):
        ev.band([])
    with pytest.raises(ValueError):
        ev.band([0.1, float("nan")])


def test_lift_touching_zero_is_no_lift():
    out = ev.lift_band({0: 0.20, 1: 0.21, 2: 0.22, 3: 0.23, 4: 0.24}, 0.20)
    assert out["min"] == 0.0 and out["verdict"] == "no lift"
    out = ev.lift_band({0: 0.20, 1: 0.19, 2: 0.18, 3: 0.17, 4: 0.16}, 0.20)
    assert out["max"] == 0.0 and out["verdict"] == "no lift"


def test_lift_needs_the_pinned_seeds():
    with pytest.raises(ValueError, match="five seeds"):
        ev.lift_band({1: 0.3, 2: 0.3, 3: 0.3, 4: 0.3, 5: 0.3}, 0.2)
    with pytest.raises(ValueError, match="five seeds"):
        ev.lift_band({s: 0.3 for s in range(6)}, 0.2)


# ---------------------------------------------------------------- rows


def test_machine_file_wins_and_empty_file_refused(tmp_path, monkeypatch):
    mf = tmp_path / "m.txt"
    mf.write_text("  box: RTX 3090 \n")
    monkeypatch.setenv("FORGE_MACHINE_FILE", str(mf))
    assert ev.read_machine() == "box: RTX 3090"
    mf.write_text("\n")
    with pytest.raises(ValueError, match="empty"):
        ev.read_machine()
    monkeypatch.setenv("FORGE_MACHINE_FILE", "")
    assert ev.read_machine().startswith("cpu: ")


def test_machine_lists_memory(monkeypatch):
    monkeypatch.delenv("FORGE_MACHINE_FILE", raising=False)
    assert "mem: " in ev.read_machine() and "GiB" in ev.read_machine()


def test_make_row_reads_sha_and_machine_itself(monkeypatch, tmp_path):
    mf = tmp_path / "m.txt"
    mf.write_text("m")
    monkeypatch.setenv("FORGE_MACHINE_FILE", str(mf))
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    row = _row(run_id="fixed")
    assert row["git_sha"] == head and row["run_id"] == "fixed" and isinstance(row["git_dirty"], bool)
    dt.datetime.fromisoformat(row["date"])
    for key in ("git_sha", "machine", "git_dirty"):
        with pytest.raises(ValueError, match="itself"):
            _row(**{key: "x"})


def test_write_rows_refuses_nested_ids_and_writes_nothing(tmp_path):
    good = _row()
    out = tmp_path / "r.jsonl"
    with pytest.raises(ValueError, match="forbidden"):
        ev.write_rows(out, [good, {**good, "metrics": {"ndcg@10": 0.1}, "extra": {"article_ids": ["0100000000"]}}])
    assert not out.exists()
    with pytest.raises(ValueError, match="forbidden"):
        ev.write_rows(out, [{**good, "per_seed": [{"customer_id": "c"}]}])


def test_write_rows_checks_required_fields_and_values(tmp_path):
    good = _row()
    out = tmp_path / "r.jsonl"
    for bad, msg in (
        ({k: v for k, v in good.items() if k != "quant"}, "lacks"),
        ({**good, "row": "E9"}, "row name"),
        ({**good, "split": "train"}, "split"),
        ({**good, "metrics": {"ndcg@10": "high"}}, "numbers"),
        ({**good, "metrics": {"ndcg@10": float("nan")}}, "JSON|compliant|nan"),
    ):
        with pytest.raises(ValueError, match=msg):
            ev.write_rows(out, [bad])
    assert not out.exists()


def test_write_rows_appends(tmp_path):
    out = tmp_path / "sub" / "r.jsonl"
    ev.write_rows(out, [_row(table="overall")])
    ev.write_rows(out, [_row(table="Tops")])
    assert [json.loads(l)["table"] for l in out.read_text().splitlines()] == ["overall", "Tops"]


# ---------------------------------------------------------------- the fixture run


@pytest.fixture(scope="module")
def fixture_rows(tmp_path_factory):
    out = tmp_path_factory.mktemp("fixture-run")
    path = er.run_fixture(out)
    return [json.loads(l) for l in path.read_text().splitlines()], out


def test_fixture_rows_follow_schema(fixture_rows):
    rows, out = fixture_rows
    schema = json.loads((REPO_ROOT / "contracts" / "results.schema.json").read_text())
    for r in rows:
        assert set(schema["required"]) <= set(r)
        assert r["row"] in schema["properties"]["row"]["enum"] and r["split"] == "dev"
        assert not (FORBIDDEN_RESULT_KEYS & set(r))
        assert len(r["split_rule_sha256"]) == 64 and len(r["split_ids_sha256"]) == 64
    assert len({r["run_id"] for r in rows}) == 1
    assert (out / "summary.md").read_text().count("| POP |") == sum(r["row"] == "POP" for r in rows)


def test_fixture_rows_carry_no_fixture_ids(fixture_rows):
    rows, _ = fixture_rows
    text = json.dumps(rows)
    ids = {r["customer_id"] for r in csv.DictReader((FIX / "transactions_train.csv").open())}
    ids |= {r["article_id"] for r in csv.DictReader((FIX / "articles.csv").open())}
    assert not [i for i in ids if i in text]


def test_fixture_run_is_deterministic(fixture_rows, tmp_path):
    rows, _ = fixture_rows
    again = [json.loads(l) for l in er.run_fixture(tmp_path).read_text().splitlines()]
    strip = lambda rs: [(r["row"], r["table"], r["n_customers"], r["metrics"]) for r in rs]
    assert strip(rows) == strip(again)


def _by_hand_pop(rule):
    """POP on the dev split, from the CSVs with the contract's rules, no tasterank.data."""
    d = dt.date.fromisoformat
    start, end = d(rule["windows"]["dev"]["start"]), d(rule["windows"]["dev"]["end"])
    lo, hi = rule["customer_hash"]["buckets"]["dev"]
    salt = rule["customer_hash"]["salt"]
    bucket = lambda c: int(hashlib.sha256(f"{salt}:{c}".encode()).hexdigest()[:16], 16) % 100
    tx = [(d(r["t_dat"]), r["customer_id"], r["article_id"]) for r in csv.DictReader((FIX / "transactions_train.csv").open())]
    tx = [t for t in tx if t[0] <= end]
    pool = sorted({a for t, _, a in tx if start - dt.timedelta(days=84) <= t < start})
    counts = {a: 0 for a in pool}
    for t, _, a in tx:
        if start - dt.timedelta(days=7) <= t < start and a in counts:
            counts[a] += 1
    ranking = sorted(pool, key=lambda a: (-counts[a], a))
    hist = {c for t, c, _ in tx if t < start and lo <= bucket(c) <= hi}
    rel: dict[str, set[str]] = {}
    for t, c, a in tx:
        if start <= t and c in hist:
            rel.setdefault(c, set()).add(a)
    ndcg = []
    for c, r in rel.items():
        dcg = sum(1 / math.log2(i + 2) for i, a in enumerate(ranking[:10]) if a in r)
        ndcg.append(dcg / sum(1 / math.log2(i + 2) for i in range(min(len(r), 10))))
    rec12 = [len(r & set(ranking[:12])) / len(r) for r in rel.values()]
    cov = sum(len(r & set(pool)) for r in rel.values()) / sum(len(r) for r in rel.values())
    return len(rel), sum(ndcg) / len(ndcg), sum(rec12) / len(rec12), cov


def test_fixture_pop_matches_hand_computation(fixture_rows):
    rows, _ = fixture_rows
    n, ndcg, rec12, cov = _by_hand_pop(json.loads((FIX / "split.json").read_text()))
    pop = next(r for r in rows if r["row"] == "POP" and r["table"] == "overall")
    assert pop["n_customers"] == n
    assert pop["metrics"]["ndcg@10"] == pytest.approx(ndcg, abs=1e-12)
    assert pop["metrics"]["recall@12"] == pytest.approx(rec12, abs=1e-12)
    assert pop["metrics"]["pool_coverage"] == pytest.approx(cov, abs=1e-12)


def test_category_rows_cover_every_customer_group(fixture_rows):
    rows, _ = fixture_rows
    overall = {r["row"]: r for r in rows if r["table"] == "overall"}
    for name in ("POP", "E0-tiny"):
        cats = [r for r in rows if r["row"] == name and r["table"] != "overall"]
        assert cats and all(r["n_customers"] <= overall[name]["n_customers"] for r in cats)


def _toy_split():
    import polars as pl

    hist = pl.DataFrame(
        {"t_dat": [dt.date(2020, 9, 1)], "customer_id": ["c1"], "article_id": ["h1"], "row_idx": [0]},
        schema={"t_dat": pl.Date, "customer_id": pl.String, "article_id": pl.String, "row_idx": pl.UInt32},
    )
    return EvalSplit(
        name="dev", window_start=dt.date(2020, 9, 9), window_end=dt.date(2020, 9, 15), customers=["c1", "c2"],
        history=hist, relevant={"c1": frozenset({"p2"}), "c2": frozenset({"p1"})}, pool=["p1", "p2", "p3"], pop_counts={},
    )


class _Mean(torch.nn.Module):
    image_size = 2

    def forward(self, pixels):
        return pixels.flatten(2).mean(-1)


def test_missing_images_rank_last_and_are_counted():
    imgs = {"h1": torch.zeros(3, 2, 2) + torch.tensor([1.0, 0, 0]).view(3, 1, 1), "p2": torch.ones(3, 2, 2), "p3": torch.zeros(3, 2, 2) + torch.tensor([1.0, 0, 0]).view(3, 1, 1)}
    rankings, counts = er.encoder_rankings(_toy_split(), _Mean(), imgs.get, 5, None)
    assert rankings["c1"] == ["p3", "p2", "p1"]  # p1 has no image: last
    assert rankings["c2"] == ["p2", "p3", "p1"]  # c2 has no history: zero vector, ties by id, p1 last
    assert counts == {"missing_images": 1, "no_history_image_customers": 1}


def test_eval_sh_usage_errors():
    for args in (["--fixture"], ["--heldout"], ["--fixture", "a", "b"]):
        p = subprocess.run(["bash", "scripts/eval.sh", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
        assert p.returncode == 2 and "usage" in p.stderr
    p = subprocess.run(["bash", "scripts/eval.sh"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0 and "not an eval phase" in p.stdout


def test_missing_image_in_pool_and_history_counts_once():
    # TR-P3c-r1-C1: h1 is both a pool article and c1's history item; with no image it is one missing article.
    split = _toy_split()
    split = EvalSplit(
        name=split.name, window_start=split.window_start, window_end=split.window_end, customers=split.customers,
        history=split.history, relevant=split.relevant, pool=["h1", "p2", "p3"], pop_counts={},
    )
    imgs = {"p2": torch.ones(3, 2, 2), "p3": torch.zeros(3, 2, 2) + torch.tensor([1.0, 0, 0]).view(3, 1, 1)}
    rankings, counts = er.encoder_rankings(split, _Mean(), imgs.get, 5, None)
    assert counts == {"missing_images": 1, "no_history_image_customers": 2}
    assert rankings["c1"][-1] == "h1"


def test_eval_sh_fixture_dir_is_relative_to_caller(tmp_path):
    # TR-P3c-r1-C2: a relative --fixture dir resolves against the caller's directory, not the repo root.
    p = subprocess.run(["bash", str(REPO_ROOT / "scripts" / "eval.sh"), "--fixture", "rel"], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    assert list((tmp_path / "rel").glob("fixture-*.jsonl")) and (tmp_path / "rel" / "summary.md").is_file()
    assert not (REPO_ROOT / "rel").exists()
