# Review: P3a r1 (PR #5)

Reviewer: forge-review, fresh session. Head reviewed: `a9cee5be4861109c57cf6bb536bd241a5d9f8ee1`.

Read: BUILD_OUTLINE.md sections 2 to 6 and the P3 checklist, the TR brief, `SPEC.md`, `.forge/state.json`, `contracts/seams.md`, `contracts/eval.md`, `contracts/data.md`, `contracts/split.json`, `scripts/lane-ownership.json`, `src/tasterank/seams.py`, the full PR diff (`git diff main...a9cee5b`), and the `ci`, `guard`, `forge/guard`, `forge/eval` and `report` results on the head (all pass).

## Verdict: PASS

No S, C or E finding. One minor D finding, not about a number.

## Findings

| ID | Class | Severity | Evidence | Fix |
|---|---|---|---|---|
| TR-P3a-r1-D1 | docs | minor | `src/tasterank/data/__init__.py:164`: the `_eligible` docstring says it returns "(visible rows, eligible ids frame, window-only count)", but line 172 returns four values `visible, mine, eligible, cold`. Private helper; every caller unpacks four. | Name the four values in the docstring. Can ride with P4 or the next lane; does not block. |

## Checks run on this head

| Check | Command or source | Result |
|---|---|---|
| Lane tests | `bash scripts/test.sh` | `23 passed` (8 contract checks, 15 lane tests) |
| Exit gate | `bash scripts/forge-gate.sh P3a` | P3a red tests green, the P3b and P3c red tests `fails for its stated reason`, then `forge-gate: P3a passed` |
| Tests not weakened | `git diff main...a9cee5b -- tests/red tests/contracts \| wc -l` | `0`; only `tests/data/test_pipeline.py` is added |
| No contract edits | `git diff --stat main...a9cee5b` | 5 files: `.forge/state.json`, `DECISIONS.md`, `README.md`, `src/tasterank/data/__init__.py`, `tests/data/test_pipeline.py`. All inside P3a's `owns` or `everyLane` in `scripts/lane-ownership.json`; nothing under `contracts/` or `src/tasterank/seams.py` |
| Signatures | `contracts/seams.md` P3a table vs. the module | all nine functions present with the frozen signatures; `scan_transactions` and `iter_training_pairs` are additions, logged as D-P3a-1 and D-P3a-3 |
| Split, pool, POP, relevant sets, cold start | independent brute-force re-implementation of `contracts/eval.md` in plain Python over `fixtures/shop_like` (reviewer's script, not committed) compared to `build_split` on dev and heldout, eager and lazy input | all equal: dev 9 eligible, 3 cold start; heldout 7 eligible, 0 cold start (fixture counts, synthetic data); history row counts equal |
| Training pairs | same brute force, `n_last` 3, train bucket, target before `train_cutoff`, history strictly before the target day, newest first, later row first on a tie | equal to `training_pairs`, 119 pairs on the fixture |
| Canonical id-list hash | brute-force `<split>\t<id>\n` lines, train/dev/heldout, sorted | equal to `split_ids_sha256` |
| Same-day ordering | `last_n_history` on three same-day rows with `row_idx` 5, 9, 7, `n` 2 | returns rows 9 then 7 (later file row first) |
| Ingest | header-only file; leading-zero `article_id`; reordered and extra columns; `2020/09/01` date | schema equals `TRANSACTION_COLUMNS` with zero rows; `0108775015` kept; columns read by name; bad date raises `InvalidOperationError` |
| No pandas | `grep -rn pandas src/tasterank/data tests/data` | only the docstring line saying pandas is not used |
| README says what ran | `README.md` "Lane P3a: data" | names the tests, CPU only, synthetic fixture only, no competition data read |
| Attribution | `git log -1 --format=%B a9cee5b` and the PR body, grep for co-authored, generated, claude, anthropic | no hits |

## Not findings (checked, left as is)

- `iter_training_pairs` and `check_anchor` collect a LazyFrame into memory, and `_customers_in` hashes ids in a Python loop. On the 31M-row file this costs memory and time, not correctness. The PR's RFI-P3a-1 already raises the pair-volume question for P4.
- RFI-P3a-1 asks P4 to add `iter_training_pairs` to `contracts/seams.md`. The lane did not edit the contract, which is the right path.
- `.forge/state.json` sets phase P3a, round 1, `sessions_used.P3a` 1, inside the P3 budget of 2.
