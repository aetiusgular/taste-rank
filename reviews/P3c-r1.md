# Review: P3c r1 (PR #9)

Reviewer: forge-review, fresh session. Head reviewed: `0008aee10a395646ba577b5c8595edafabf8c0b0`.

Read: BUILD_OUTLINE.md sections 3 to 6 and the P3 checklist, `forge/briefs/TR.md`, `.forge/state.json`, `contracts/eval.md`, `scripts/lane-ownership.json`, all of `src/tasterank/eval/__init__.py` and `src/tasterank/eval/run.py`, `scripts/eval.sh`, `tests/eval/test_eval.py`, the README P3c section, DECISIONS.md D-P3c-1 to 8, and the head's `ci`, `guard` and `forge/eval` results (all pass).

## Verdict: FIX

Two C findings. C1 puts a wrong count into a results row field, and C2 sends fixture output to the wrong directory without saying so. Both are fixed in round 2.

## Checks run

| Check | Command or source | Result |
|---|---|---|
| Lane tests and lint | `bash scripts/setup.sh && bash scripts/test.sh` | `82 passed` |
| Exit gate | `FORGE_BASE=d9af15c bash scripts/forge-gate.sh P3c` | the 10 P3c red tests are green, `lane-scope: lane P3c clean since d9af15c33049 (7 path(s))`, `forge-gate: P3c passed` |
| Tests not weakened, no contract edits | `git diff --stat d9af15c -- tests/ contracts/ fixtures/` | only `tests/eval/test_eval.py` (+326, -0). No red test, contract or fixture changed |
| Metric definitions vs contracts/eval.md | read `ndcg_at_k`, `recall_at_k`, `pool_coverage`, `category_metrics`, `lift_band`; probe `ndcg_at_k(["x","a"], {"a"}, 10)` | `0.6309...`, which equals `1/log2(3)`. The 1-indexed `log2(i+1)` rule holds. The verdict rule matches "touching zero is no lift" |
| Non-finite embeddings | `rank(ones(4), ["a","b"], [[1,0,0,0],[inf,0,0,0]], 2)` | refused: `a score is NaN (non-finite embedding)` |
| Attribution | PR body and `git log d9af15c..0008aee --format=%B`, grep co-authored / generated with / claude / anthropic | no hits (and `ci`'s attribution scan passed) |

## Findings

### TR-P3c-r1-C1: `missing_images` counts an article twice when it is in both the pool and a history

- Class: C. Severity: above minor. The field goes into every E0 and E1 results row, and in P5 the box will publish it as the missing-image count (D-P3c-6, RFI-P3c-2).
- Evidence: `src/tasterank/eval/run.py` lines 89 to 90 and 109. `missing_pool = len(unscored)` counts pool articles with no image. `missing_hist` counts history articles with no image. The row gets `missing_pool + missing_hist`, so an article in both sets is counted twice. In the fixture's dev split, all 24 history articles are also pool articles (pool 39, history 24, overlap 24). Probe: drop the images of 3 overlapping articles, then call `encoder_rankings`. It returned `{'missing_images': 6, ...}` for 3 distinct missing articles. The lane test `test_missing_images_rank_last_and_are_counted` covers only an article in the pool alone, so it does not catch this.
- Fix: use the count `embed_articles` already returns, which counts the distinct ids across the pool and the histories together. New lane test: `test_missing_image_in_pool_and_history_counts_once`. On the r1 code it fails with `missing_images` 2. On the fix it passes.

### TR-P3c-r1-C2: `scripts/eval.sh --fixture <relative dir>` writes under the repo root, not the caller's directory

- Class: C. Severity: above minor. The output lands somewhere other than where the caller asked, and nothing says so. `--fixture results` run from any directory appends synthetic rows to the repo's `results/`, which section 6 item 7 treats as the record of real runs.
- Evidence: `scripts/eval.sh` line 19 runs `cd "$(dirname "$0")/.."` before it passes `$2` to `python -m tasterank.eval.run`. Probe: `cd <scratch>/cwd && bash <repo>/scripts/eval.sh --fixture relout` printed `wrote relout/fixture-...jsonl`. `<scratch>/cwd` stayed empty, and `git status` in the repo showed `?? relout/`.
- Fix: `eval.sh` turns a relative `--fixture` dir into an absolute one against `$PWD` before the `cd`. New lane test: `test_eval_sh_fixture_dir_is_relative_to_caller`, run from a temp directory. On the r1 script it fails because nothing is written to `<tmp>/rel`. On the fix it passes.

## Not findings (checked, left as is)

- `read_machine` raises on an empty `$FORGE_MACHINE_FILE` rather than falling back to /proc (D-P3c-4). The contract's "set and not empty" can be read as being about the variable. Raising is the stricter reading, and it never copies a stale label.
- `check_row` refuses forbidden keys, not values that look like id lists. That is the rule as `contracts/eval.md` states it ("`write_rows` refuses those keys").
- `check_row` does not type-check `n_customers`. Minor. P4 can fold it into the schema with RFI-P3c-1.
- RFI-P3c-1 to 3 are routed correctly to P4 and P5. No contract was edited.
