# Review: P3b r3 (PR #8)

Reviewer: forge-review, fresh session. Head reviewed: `10721bc122e708d5a8856e9a1fdadd2546ab4ce2`.

Read: BUILD_OUTLINE.md sections 2 to 6 and the P3 checklist, `forge/briefs/TR.md`, `SPEC.md`, `.forge/state.json`, `scripts/lane-ownership.json`, `reviews/P3b-r1.md`, `reviews/P3b-r2.md`, all of `src/tasterank/train/__init__.py`, the r3 diff (`git diff 6057a40..10721bc`), the README P3b section, the PR body, and the head's `ci`, `guard`, `report`, `forge/guard` and `forge/eval` results (all success).

## Verdict: PASS

The r2 fixes hold. No S, C or E finding. One D finding about a number, in the PR body, fixed in place (D1).

## Checks run

| Check | Command or source | Result |
|---|---|---|
| Lane tests and lint | `bash scripts/setup.sh && bash scripts/test.sh` | `54 passed` |
| Exit gate | `FORGE_BASE=origin/main scripts/forge-gate.sh P3b` | 9 P3b red tests green, P3c red tests fail for their stated reason, `lane-scope: lane P3b clean since cdd90543caa9 (7 path(s))`, `forge-gate: P3b passed` |
| Tests not weakened, no contract or grader edits | `git diff origin/main -- tests/red tests/contracts contracts src/tasterank/seams.py scripts .github \| wc -l` | `0`. The r3 test diff adds one parametrized test and removes nothing |
| TR-P3b-r2-C1 fix | probe: `customer_vector(torch.ones(2,3), [1,2], hl)` for `hl` in `inf, None, -inf, nan, 0` | `inf` and `None` give equal weights (`[0.5774, 0.5774, 0.5774]`); `-inf`, `nan`, `0` raise `ValueError`. Matches `train`'s `_half_life_for_meta` |
| TR-P3b-r2-D1 fix | `pytest -q --co tests/train` | `31 tests collected`; README line 15 says 31 |
| README section says what ran | `README.md` "Lane P3b: train" | CPU, `tasterank.tiny`, synthetic fixture, no results rows |
| Attribution | `git log origin/main..10721bc --format=%B` and the PR body, grep co-authored / generated with / claude / anthropic | no hits |

## Findings

### TR-P3b-r3-D1: the PR body quotes a lane-scope path count the head does not give

- Class: D (a number). Severity: minor.
- Evidence: PR #8 body, gate checklist: "`FORGE_BASE=origin/main scripts/forge-gate.sh P3b` prints `lane-scope: lane P3b clean since cdd90543caa9 (6 path(s))`". On `10721bc` the same command prints `(7 path(s))`, and so does the `ci` run on that head (run 45455440678, job "Phase gate": `lane-scope: lane P3b clean since cdd90543caa9 (7 path(s))`). The seventh path is `reviews/P3b-r2.md`, added in this round. The repo squash-merges with the PR body as the commit message, so the stale count would land on `main`.
- Fix (applied in place, no code change): the PR body now quotes the count the `ci` run gave on `10721bc` (7) and notes that the review commit adds `reviews/P3b-r3.md` as an eighth path.

## Not findings (checked, left as is)

- `customer_vector` does not check `ages_days` for NaN; a NaN age gives a NaN vector. Ages come from `date` subtraction in `_step_loss` and in P3c, so no NaN can reach it from the pipeline. Not above minor.
- The r1 and r2 "Not findings" still stand (bucket check, out_dir overwrite, `warn_only` determinism, `True` recorded as 1.0, `"30"` raising `TypeError`, RFI-P3b-1 and RFI-P3b-2 for P4 and P5).
