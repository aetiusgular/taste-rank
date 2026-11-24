# Review: P3b r1 (PR #6)

Reviewer: forge-review, fresh session. Head reviewed: `38c45bdaccae6d9929887d60cc3f19adfd98e579`.

Read: BUILD_OUTLINE.md sections 2 to 6 and the P3 checklist, `forge/briefs/TR.md`, `SPEC.md`, `.forge/state.json`, `contracts/seams.md` (P3b table), `contracts/eval.md` ("Customer vector and score", "Training pairs (E1)"), `contracts/split.json`, `src/tasterank/seams.py`, every file in the PR diff, and the `ci`, `guard`, `forge/guard` and `forge/eval` results on the head (all success).

## Verdict: FIX

One C finding above minor (C1). Fixed in round 2.

## Checks run

| Check | Command or source | Result |
|---|---|---|
| Lane tests and lint | `bash scripts/test.sh` | `45 passed` |
| Exit gate | `FORGE_BASE=origin/main scripts/forge-gate.sh P3b` | 9 P3b red tests green, 13 P3a still green, 10 P3c fail for their stated reason, `lane-scope: lane P3b clean since cdd90543caa9 (5 path(s))`, `forge-gate: P3b passed` |
| Tests not weakened, no contract edits | `git diff origin/main -- tests/red tests/contracts contracts src/tasterank/seams.py \| wc -l` | `0` |
| `customer_vector` vs eval.md | read against "Vector": normalize each row, `w = 0.5 ** (age / half_life)`, weighted sum, normalize; ages from target day minus purchase day (`_step_loss`) | matches |
| `info_nce` vs eval.md | `cross_entropy(C·Tᵀ/τ, arange(B))` with normalized inputs | matches |
| Batches vs `arange(B)` labels | `_batches` keeps targets distinct per batch (D-P3b-1); lane test `test_batches_have_distinct_targets_and_are_seeded` | holds |
| History cap | `pr.history[: cfg.n_last]`; `TrainPair.history` is most recent first (`seams.py`) | keeps the last `n_last` |
| Refusals before work | `_check_out_dir` (resolved path, symlinks) and `_check_pairs` run before `set_seed`/`apply_lora`; `test_refusal_writes_nothing` | holds for cutoff and out_dir |
| Checkpoint load | `torch.load(..., weights_only=True)`, strict key match | no pickled code path |
| README section says what ran | `README.md` "Lane P3b: train" | CPU, `tasterank.tiny`, synthetic fixture, no results rows |
| Attribution | `git log origin/main..38c45bd --format=%B` and PR body, grep co-authored / generated with / claude / anthropic | no hits |

## Findings

### TR-P3b-r1-C1: a `null` half-life (a frozen grid point) crashes `train` after training, leaving `lora.pt` without `lora.json`

- Class: C. Severity: medium.
- Evidence: `contracts/split.json` line 24, `"history_grid": { ..., "half_life_days": [7, 30, 90, null] }`, and `contracts/eval.md` line 44, "an infinite (`null`) half-life gives equal weights". The lane accepts `None` in `customer_vector` (docstring "math.inf (or None) gives equal weights"; `test_customer_vector_none_half_life_is_equal_weights`), but `train` writes the meta with `None if math.isinf(cfg.half_life_days) else ...` (`src/tasterank/train/__init__.py`, the `meta` dict in `train`). Run on the head with `TrainConfig(..., half_life_days=None)` and the lane test fixture: every step runs, then `TypeError must be real number, not NoneType`; the out_dir holds `['lora.pt']` only. On the box that is a full 3090 run lost at the last line, with an orphan checkpoint and no config beside it. Also `_weights` checks `half_life_days <= 0`, which lets NaN through, so a NaN half-life trains on NaN losses and writes a checkpoint.
- Fix: validate `cfg.half_life_days` in `train` before any work (`_half_life_for_meta`): `None` and `+inf` record `null`, anything not positive (0, negative, NaN, `-inf`) raises `ValueError` naming `half_life_days` before any file is written; `_weights` uses `not half_life_days > 0` so NaN is refused there too. Tests: `test_null_half_life_trains_and_is_recorded` and `test_bad_half_life_refused_before_any_write` (four values) in `tests/train/test_train.py`.

## Not findings (checked, left as is)

- `train` does not check that pairs come from train-bucket customers. The contract asks only for the cutoff refusal; `training_pairs` (P3a) filters the bucket. Not above minor.
- An existing `out_dir` is overwritten without warning. Box-local and outside the repo; not above minor.
- `set_seed` turns on deterministic algorithms process-wide with `warn_only=True` (D-P3b-4). Logged with a reason; the CPU determinism red test passes.
- RFI-P3b-1 (loader in `contracts/seams.md`) and RFI-P3b-2 (confirm LoRA targets and memory on the real trunk) are correctly left for P4 and P5.
