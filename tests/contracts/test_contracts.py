"""Contract self-checks (P2). Green from P2 on; scripts/test.sh runs them in ci."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_split_hash_matches_eval_md():
    text = (ROOT / "contracts" / "eval.md").read_text()
    want = re.search(r"^split-sha256: *([0-9a-f]{64})", text, re.M).group(1)
    assert hashlib.sha256((ROOT / "contracts" / "split.json").read_bytes()).hexdigest() == want


def test_fixture_rule_only_widens_buckets():
    frozen = json.loads((ROOT / "contracts" / "split.json").read_text())
    fixture = json.loads((ROOT / "fixtures" / "shop_like" / "split.json").read_text())
    for key in ("windows", "train_cutoff", "pool_days_before_window", "pop_days_before_window", "seeds", "eligible"):
        assert fixture[key] == frozen[key], key
    assert fixture["customer_hash"]["salt"] == frozen["customer_hash"]["salt"]
    assert fixture["anchor"]["expected_max_t_dat"] == frozen["anchor"]["expected_max_t_dat"]


def test_fixtures_match_generator():
    p = subprocess.run([sys.executable, str(ROOT / "fixtures" / "make_fixtures.py"), "--check"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


def test_fixture_images_exist():
    arts = (ROOT / "fixtures" / "shop_like" / "articles.csv").read_text().splitlines()[1:]
    for line in arts:
        aid = line.split(",")[0]
        assert (ROOT / "fixtures" / "shop_like" / "images" / aid[:3] / f"{aid}.jpg").is_file(), aid


def test_metric_cases_are_their_formulas():
    cases = json.loads((ROOT / "fixtures" / "metrics_cases.json").read_text())
    env = {"log2": math.log2, "__builtins__": {}}
    for c in cases["cases"]:
        for m in ("ndcg@10", "recall@12", "recall@50"):
            assert math.isclose(eval(c[f"{m}_expr"], env), c[m], abs_tol=1e-12), (c["name"], m)


def test_red_manifest_points_at_real_tests():
    manifest = json.loads((ROOT / "tests" / "red" / "manifest.json").read_text())
    lanes = [p for p in json.loads((ROOT / ".forge" / "state.json").read_text())["phases"] if p.startswith("P3")]
    for t in manifest["tests"]:
        assert t["lane"] in lanes, t["id"]
        m = re.search(r"tests/red/(test_\w+\.py)::(test_\w+)$", t["cmd"])
        assert m, t["cmd"]
        tree = ast.parse((ROOT / "tests" / "red" / m.group(1)).read_text())
        assert m.group(2) in {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}, t["id"]
    for lane in lanes:
        assert any(t["lane"] == lane for t in manifest["tests"]), lane


def test_results_schema_forbids_the_same_keys():
    sys.path.insert(0, str(ROOT / "src"))
    from tasterank.seams import FORBIDDEN_RESULT_KEYS, SECTION6_FIELDS

    schema = json.loads((ROOT / "contracts" / "results.schema.json").read_text())
    assert {k for a in schema["not"]["anyOf"] for k in a["required"]} == set(FORBIDDEN_RESULT_KEYS)
    assert set(SECTION6_FIELDS) <= set(schema["required"])


def test_models_lock_matches_spec():
    lock = json.loads((ROOT / "models.lock").read_text())
    spec = (ROOT / "SPEC.md").read_text()
    for m in lock["models"]:
        assert re.fullmatch(r"[0-9a-f]{40}", m["revision"]) and m["revision"] in spec, m["id"]
    assert lock["seeds"] == json.loads((ROOT / "contracts" / "split.json").read_text())["seeds"]
