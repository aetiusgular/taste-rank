# Review: P3b r2 (PR #7)

Reviewer: forge-review, fresh session. Head reviewed: `6057a40edf479cb5b410f29ab99b9b82f880f3ff`.

Read: BUILD_OUTLINE.md sections 2 to 6 and the P3 checklist, `forge/briefs/TR.md`, `.forge/state.json`, `reviews/P3b-r1.md`, `contracts/seams.md` (P3b table), `contracts/eval.md` ("Customer vector and score", "Training pairs (E1)"), all of `src/tasterank/train/__init__.py`, the r2 diff (`git diff 38c45bd..6057a40`), the README P3b section, and the head's `ci`, `guard`, `forge/guard` and `forge/eval` results (all pass).

## Verdict: FIX

The C1 fix from r1 holds. One D finding about a number (D1), which the verdict rule says must be fixed. One minor C finding (C1). Both are fixed in round 3.

## Checks run

| Check | Command or source | Result |
|---|---|---|
| Lane tests and lint | `bash scripts/setup.sh && bash scripts/test.sh` | `50 passed` |
| Exit gate | `FORGE_BASE=origin/main scripts/forge-gate.sh P3b` | `lane-scope: lane P3b clean since cdd90543caa9 (6 path(s))`, `forge-gate: P3b passed` |
| Tests not weakened, no contract or grader edits | `git diff origin/main -- tests/red tests/contracts contracts src/tasterank/seams.py scripts .github \| wc -l` | `0`. The r2 test diff only adds tests |
| TR-P3b-r1-C1 fix | probe: `train` with `half_life_days` in `inf, None, 7, 30.0, 90, nan, "30", True` on the lane fixture | inf and None train and record `null`. 7, 30, 90 are recorded as floats. NaN gives `ValueError ... half_life_days`, and `"30"` gives `TypeError`, both with nothing written. `True` is recorded as 1.0 (bool is an int; not above minor) |
| Refusal order | `train`: `_check_out_dir`, `_check_pairs`, `_half_life_for_meta`, steps/batch check, then `set_seed`, then `mkdir` | every refusal runs before any write |
| Attribution | `git log origin/main..6057a40 --format=%B` and the PR body, grep co-authored / generated with / claude / anthropic | no hits |

## Findings

### TR-P3b-r2-D1: the README's lane-test count is stale

- Class: D (a number). Severity: minor, but it is a number, so it is fixed.
- Evidence: `README.md` line 15 says "the 9 P3b red tests and 22 lane tests in `tests/train/`". 22 is the count at r1: `pytest --co` on `38c45bd` gives `22 tests collected`. On `6057a40`, `pytest -q --co tests/train` gives `27 tests collected`, because r2 added one test and four parametrized cases.
- Fix: the README now gives the count after this round's added tests: `31 tests collected` (`pytest -q --co tests/train` on the r3 head).

### TR-P3b-r2-C1: `customer_vector` treats a `-inf` half-life as equal weights, but `train` refuses it

- Class: C. Severity: minor (no grid point is `-inf`, and `train` refuses it before `customer_vector` runs).
- Evidence: `src/tasterank/train/__init__.py` line 37: `if half_life_days is None or math.isinf(half_life_days): return torch.ones_like(...)`. `math.isinf` is true for `-inf`. Probe: `train.customer_vector(torch.ones(2,3), [1,2], -math.inf)` is accepted, while `train(..., half_life_days=-inf)` raises (`test_bad_half_life_refused_before_any_write`). P3c calls `customer_vector` directly, and there it would turn a negative-infinite half-life into equal weights without an error.
- Fix: `_weights` returns equal weights only for `None` or `+inf`. `-inf` now falls through to the `not half_life_days > 0` refusal. New lane test: `test_customer_vector_refuses_bad_half_life` (0, -1, NaN, -inf).

## Not findings (checked, left as is)

- `half_life_days=True` is recorded as `1.0`. That is Python's bool-as-int behavior, and the `TrainConfig` type is P4's call (the DECISIONS.md revisit trigger already names it).
- `half_life_days="30"` raises `TypeError` rather than a `ValueError` naming the field. It still raises before any write. Minor.
- The r1 "Not findings" (bucket check, out_dir overwrite, `warn_only` determinism, RFI-P3b-1 and RFI-P3b-2) still stand.
