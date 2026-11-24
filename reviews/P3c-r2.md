# Review: P3c r2 (PR #10)

Reviewer: forge-review, fresh session. Head reviewed: `f60b0239a0288e4859de1b3b5d4b7d017237c689`.

Read: BUILD_OUTLINE.md sections 2 to 6 and the P3 checklist, `forge/briefs/TR.md`, `.forge/state.json`, `contracts/eval.md`, `reviews/P3c-r1.md`, all of `src/tasterank/eval/__init__.py`, `src/tasterank/eval/run.py` and `scripts/eval.sh`, the r2 diff (`git diff 0008aee..f60b023`), the README P3c section, DECISIONS.md D-P3c-1 to 8 and the r1 entry, the PR body, and the head's `ci`, `guard`, `report`, `forge/guard` and `forge/eval` results (all success).

## Verdict: PASS

The two round 1 fixes hold, and each has a test that fails on the round 1 code. I found no S, C, E or D findings.

## Checks run

| Check | Command or source | Result |
|---|---|---|
| Lane tests and lint | `bash scripts/setup.sh && bash scripts/test.sh` | `84 passed` |
| Exit gate | `FORGE_BASE=d9af15c bash scripts/forge-gate.sh P3c` | P3c red tests green, `lane-scope: lane P3c clean since d9af15c33049 (8 path(s))`, `forge-gate: P3c passed` |
| Tests not weakened, no contract or grader edits | `git diff --stat d9af15c -- tests/ contracts/ fixtures/`; `git diff d9af15c -- tests/red tests/contracts contracts src/tasterank/seams.py scripts/forge-gate.sh scripts/lane-scope.mjs scripts/lane-ownership.json .github \| wc -l` | only `tests/eval/test_eval.py` (+347, -0); `0` |
| TR-P3c-r1-C1 fix | `run.py` line 87: `missing` is `embed_articles`' count, which iterates `dict.fromkeys(article_ids)`, so each distinct id is counted once across pool and histories | `test_missing_image_in_pool_and_history_counts_once` passes on head and FAILS on `0008aee` (run with `PYTHONPATH=<r1 worktree>/src` so the r1 package is the one imported) |
| TR-P3c-r1-C2 fix | `scripts/eval.sh` lines 20 to 22; probe `cd <scratch>/cwd && bash <repo>/scripts/eval.sh --fixture ./out` | wrote `<scratch>/cwd/out/fixture-*.jsonl` and `summary.md`; repo `git status --short` empty. `test_eval_sh_fixture_dir_is_relative_to_caller` passes on head and FAILS on `0008aee` |
| README test count | `pytest -q --co tests/eval` | `30 tests collected`; README says 30 |
| README says what ran | `README.md` "Lane P3c: eval" | CPU, synthetic fixture, `tasterank.tiny`, no results rows, no real encoder or competition data |
| Attribution | `git log d9af15c..HEAD --format=%B` grep co-authored / generated with / claude / anthropic | no hits; `ci` attribution scan clean |

## Findings

None.

## Not findings (checked, left as is)

- `eval.sh --fixture results`, run from the repo root, still writes into the repo's `results/`. That is the directory the caller named. The file is named `fixture-<run_id>.jsonl` and every row has `dataset: "fixtures/shop_like (synthetic)"`, so it cannot be mistaken for a box row.
- `missing_images` also counts history-only articles that have no image, not only pool articles. D-P3c-6 and the round 1 fix both define it that way, and RFI-P3c-2 sends the rule to P4.
