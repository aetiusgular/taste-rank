"""Red tests, lane P3c (eval). One test per corner case; tests/red/manifest.json lists them."""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from conftest import ROOT
from tasterank.seams import FORBIDDEN_RESULT_KEYS, SECTION6_FIELDS

CASES = json.loads((ROOT / "fixtures" / "metrics_cases.json").read_text())


def test_ndcg_matches_hand_values():
    from tasterank import eval as ev

    for c in CASES["cases"]:
        got = ev.ndcg_at_k(CASES["ranked"], c["relevant"], 10)
        assert got == pytest.approx(c["ndcg@10"], abs=1e-9), c["name"]


def test_recall_matches_hand_values():
    from tasterank import eval as ev

    for c in CASES["cases"]:
        for k in (12, 50):
            got = ev.recall_at_k(CASES["ranked"], c["relevant"], k)
            assert got == pytest.approx(c[f"recall@{k}"], abs=1e-9), (c["name"], k)


def test_rank_ties_by_article_id():
    import torch

    from tasterank import eval as ev

    ids = ["0000000003", "0000000001", "0000000002", "0000000004"]
    embs = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # 3 and 1 tie on cosine
    got = ev.rank(torch.tensor([1.0, 0.0]), ids, embs, 3)
    assert got == ["0000000001", "0000000003", "0000000004"]


def test_pool_coverage():
    from tasterank import eval as ev

    relevant = {"c1": {"a", "b"}, "c2": {"c"}, "c3": {"a", "x", "y"}}
    assert ev.pool_coverage(relevant, ["a", "b", "c"]) == pytest.approx(4 / 6)


def test_pop_ranking_ties():
    from tasterank import eval as ev

    assert ev.pop_ranking({"0000000005": 3, "0000000002": 1, "0000000001": 3, "0000000009": 2}, 3) == ["0000000001", "0000000005", "0000000009"]


def test_category_table():
    from tasterank import eval as ev

    group = {"t1": "Tops", "t2": "Tops", "b1": "Bottoms", "b2": "Bottoms", "x9": "Tops"}
    rankings = {"c1": ["b1", "t1", "t2", "b2"], "c2": ["t2", "b2", "t1", "b1"]}
    relevant = {"c1": {"t2", "b2"}, "c2": {"x9"}}  # x9 is a Tops article outside the pool
    table = ev.category_metrics(rankings, relevant, group)
    assert set(table) == {"Tops", "Bottoms"}
    # Tops: c1 sees [t1, t2], hit at rank 2; c2 sees [t2, t1], misses x9.
    assert table["Tops"]["n_customers"] == 2
    assert table["Tops"]["ndcg@10"] == pytest.approx((0.6309297535714575 + 0.0) / 2)
    assert table["Tops"]["recall@12"] == pytest.approx(0.5)
    # Bottoms: only c1 has a Bottoms purchase; it sees [b1, b2], hit at rank 2.
    assert table["Bottoms"]["n_customers"] == 1
    assert table["Bottoms"]["ndcg@10"] == pytest.approx(0.6309297535714575)


def test_band_and_lift():
    from tasterank import eval as ev

    assert ev.band([0.1, 0.3, 0.2]) == pytest.approx({"mean": 0.2, "min": 0.1, "max": 0.3, "n": 3})
    up = ev.lift_band({0: 0.25, 1: 0.27, 2: 0.26, 3: 0.24, 4: 0.28}, 0.20)
    assert up["mean"] == pytest.approx(0.06) and up["min"] == pytest.approx(0.04) and up["max"] == pytest.approx(0.08)
    assert up["verdict"] == "lift"
    flat = ev.lift_band({0: 0.19, 1: 0.22, 2: 0.21, 3: 0.20, 4: 0.23}, 0.20)
    assert flat["verdict"] == "no lift"
    down = ev.lift_band({0: 0.10, 1: 0.11, 2: 0.12, 3: 0.13, 4: 0.14}, 0.20)
    assert down["verdict"] == "worse"
    with pytest.raises(ValueError, match="five seeds"):
        ev.lift_band({0: 0.3, 1: 0.3}, 0.2)


def test_results_row_fields(tmp_path, monkeypatch):
    from tasterank import eval as ev

    mf = tmp_path / "machine.txt"
    mf.write_text("test-machine: 1x CPU\n")
    monkeypatch.setenv("FORGE_MACHINE_FILE", str(mf))
    row = ev.make_row(row="POP", split="dev", table="overall", metrics={"ndcg@10": 0.1}, n_customers=3, model_revision=None, quant=None, runtime=None)
    assert set(SECTION6_FIELDS) <= set(row)
    assert row["machine"] == "test-machine: 1x CPU"
    assert len(row["git_sha"]) == 40 and row["run_id"]
    out = tmp_path / "r.jsonl"
    ev.write_rows(out, [row])
    assert json.loads(out.read_text().splitlines()[0])["row"] == "POP"
    for key in sorted(FORBIDDEN_RESULT_KEYS):
        with pytest.raises(ValueError, match="forbidden"):
            ev.write_rows(tmp_path / "bad.jsonl", [{**row, key: ["0100000000"]}])


def test_machine_read_at_run_time(monkeypatch):
    from tasterank import eval as ev

    monkeypatch.delenv("FORGE_MACHINE_FILE", raising=False)
    model = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    assert model in ev.read_machine()


def test_eval_sh_fixture_run(tmp_path):
    out = tmp_path / "results"
    env = {**os.environ, "FORGE_MACHINE_FILE": ""}
    p = subprocess.run(["bash", "scripts/eval.sh", "--fixture", str(out)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    rows = [json.loads(l) for f in sorted(out.glob("*.jsonl")) for l in f.read_text().splitlines()] if out.exists() else []
    assert p.returncode == 0 and rows, f"eval.sh --fixture wrote no results rows (exit {p.returncode}): {p.stdout[-300:]} {p.stderr[-300:]}"
    assert {r["row"] for r in rows} >= {"POP", "E0-tiny"}
    assert {r["split"] for r in rows} == {"dev"}  # fixture runs never touch heldout
    assert any(r["table"] == "overall" for r in rows) and any(r["table"] == "Tops" for r in rows)
    for r in rows:
        assert set(SECTION6_FIELDS) <= set(r) and not (FORBIDDEN_RESULT_KEYS & set(r))
        assert {"ndcg@10", "recall@12", "recall@50", "pool_coverage"} <= set(r["metrics"])
