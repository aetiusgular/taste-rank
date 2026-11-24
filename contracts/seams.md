# Seams

Frozen in P2. The Python surface each lane implements, and what the others may rely on. Types live in `src/tasterank/seams.py` (no lane edits it). Each lane's functions start as stubs in its own package that raise `NotImplementedError("<lane>: <function>")`; the lane replaces the bodies and keeps the signatures. A lane that needs a signature changed writes a Request for integration.

The split, metric and row rules these functions follow are in `contracts/eval.md`; the input files are in `contracts/data.md`.

## P3a data: `tasterank.data` (`src/tasterank/data/`)

| Function | Returns | Rule |
|---|---|---|
| `load_transactions(path)` | polars DataFrame, columns and dtypes `TRANSACTION_COLUMNS` | ids as strings (leading zeros kept), `t_dat` a Date, `row_idx` = 0-based file row. Header-only file: zero rows, same schema. polars or DuckDB, never pandas. Lazy or streaming reads are fine for the 31M-row file. |
| `load_articles(path)` | polars DataFrame with at least `ARTICLE_COLUMNS` | `article_id` a string |
| `customer_bucket(customer_id, rule)` | int 0–99 | eval.md "Customer bucket" |
| `assign_split(customer_id, rule)` | `"train"`, `"dev"` or `"heldout"` | bucket ranges |
| `check_anchor(transactions, rule)` | None | `ValueError` naming "min t_dat" or "max t_dat" on a mismatch |
| `build_split(transactions, rule, which)` | `EvalSplit` | windows, eligibility, cold start, pool, POP counts, relevant sets. Does not call `check_anchor` (the pipeline does). |
| `last_n_history(history, n)` | dict customer_id → list of `Purchase`, most recent first | ordering in eval.md |
| `training_pairs(transactions, rule, n_last)` | list of `TrainPair` | eval.md "Training pairs" |
| `split_ids_sha256(transactions, rule)` | 64-hex string | eval.md "Canonical id-list hash" |

## P3b train: `tasterank.train` (`src/tasterank/train/`)

| Function | Returns | Rule |
|---|---|---|
| `set_seed(seed)` | None | seeds Python, numpy and torch; deterministic algorithms on CPU |
| `customer_vector(embeddings [n, d], ages_days [n], half_life_days)` | tensor [d], unit norm | eval.md "Customer vector"; `math.inf` = equal weights |
| `info_nce(customer_vecs [B, d], target_vecs [B, d], temperature)` | scalar tensor | inputs normalized inside; eval.md "Training pairs" |
| `apply_lora(encoder, cfg)` | module | LoRA (peft) on modules whose names end in `cfg.target_modules`; every other parameter frozen |
| `train(encoder, pairs, images, cfg)` | `TrainResult` | refuses a pair with `target_date >= cfg.train_cutoff` (`ValueError` naming train_cutoff) and an `out_dir` inside the repo (`ValueError` "inside the repo"); same seed, same result on CPU; writes the LoRA weights under `out_dir` only |

`TrainConfig`, `TrainResult`, `TrainPair`, `Purchase`, `ImageSource` and `Encoder` are in `seams.py`. An `Encoder` maps pixels `[B, 3, H, W]` in [0, 1] to embeddings `[B, D]`; the real encoders' preprocessing (resize, normalization) is wrapped inside the encoder in P5, so lanes never see it.

### Tiny encoder

`tasterank.tiny.tiny_encoder(seed)` is a 16x16 TinyViT with timm-style names (`blocks.N.attn.qkv`, `blocks.N.attn.proj`). It is the CPU stand-in for the base model in every lane test and in `eval.sh --fixture`. The headline base model's trunk is timm `vit_base_patch16_siglip_224` (its `open_clip_config.json` at the pinned commit), which uses the same attention names; P5 confirms the LoRA targets on the real model.

## P3c eval: `tasterank.eval` (`src/tasterank/eval/`) and `scripts/eval.sh`

| Function | Returns | Rule |
|---|---|---|
| `rank(customer_vec, candidate_ids, candidate_embs, k)` | top-k ids | cosine, ties by id ascending |
| `ndcg_at_k(ranked, relevant, k)`, `recall_at_k(ranked, relevant, k)` | float | eval.md "Metrics" |
| `pool_coverage(relevant, pool)` | float | eval.md "Metrics" |
| `pop_ranking(pop_counts, k)` | top-k ids | count descending, ties by id ascending |
| `category_metrics(rankings, relevant, article_group)` | `{group: {"ndcg@10", "recall@12", "recall@50", "n_customers"}}` | eval.md "Per-category tables"; `rankings` holds each customer's full pool ranking |
| `band(values)` | `{"mean", "min", "max", "n"}` | |
| `lift_band(e1_by_seed, e0)` | band plus `"verdict"`: `lift`, `no lift` or `worse` | five seeds or `ValueError` |
| `read_machine()` | str | eval.md "Results rows" |
| `make_row(**fields)` | dict | fills `run_id`, `git_sha` (`git rev-parse HEAD`), `machine`, `date` |
| `write_rows(path, rows)` | None | JSON lines; `ValueError` "forbidden" on any `FORBIDDEN_RESULT_KEYS` key |

`scripts/eval.sh` (owned by P3c) keeps its box contract (header comment). With `--fixture <dir>` it runs POP and `E0-tiny` on `fixtures/shop_like` (rule `fixtures/shop_like/split.json`), dev split only, and writes rows to `<dir>`. With no arguments on the box it runs the P5 pipeline; until P5 wires the real encoders it prints "not an eval phase" and exits 0.
