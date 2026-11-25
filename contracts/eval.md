# Eval contract

Frozen in P2 (2026-11-24). Lanes build against it and do not edit it; a change goes under "Requests for integration" and P4 decides. The split, metrics and bands below are fixed before any data is read and before any tuning (BUILD_OUTLINE.md section 6, items 1 and 2).

split-sha256: e54a85448bd1f760e159affa00c7ce6218879822abd67dff6c69ced8c91143a2
split-path: contracts/split.json

## What is frozen, and why the hash covers the rule

The competition data may not be redistributed (SPEC.md, "Data"), and nothing in this repo has read it yet: the Kaggle token is a P5 item. So the repo freezes the **rule** that produces the split, and `split-sha256` above is the sha256 of `contracts/split.json`. `scripts/forge-gate.sh` recomputes it on every phase after P2, so any edit to the dates, the salt, the bucket ranges or the pool and POP windows fails `ci`.

The id lists are a deterministic function of that rule and `transactions_train.csv`. They stay on the box. Every results row carries `split_ids_sha256`, the hash of those lists in the canonical format below. The first P5 run's value becomes the expected value; P5 writes it into this file under "Box values", and any later row with a different value is invalid.

## Split (contracts/split.json)

| Item | Value | Source |
|---|---|---|
| Table | `transactions_train.csv`, column `t_dat` | competition data description |
| Anchor | min `t_dat` 2018-09-20, max `t_dat` 2020-09-22. `check_anchor` refuses a file whose min or max differs (a different file version). | secondary sources, read 2026-11-24: Pisa, master's thesis, Politecnico di Milano, `https://www.politesi.polimi.it/bitstream/10589/204133/1/Master_Thesis_With_Summary_Arcangelo-Pisa.pdf` ("from 2018-09-20 to 2020-09-22"); Outerbounds recsys tutorial, `https://outerbounds.com/docs/recsys-tutorial-S2E1/` (first and last dates the same). Not yet confirmed on the file itself: the box confirms it in P5 through `check_anchor`. |
| Heldout window | 2020-09-16 to 2020-09-22, both days included (the last 7 days of the file, the competition's 7-day horizon) | rule |
| Dev window | 2020-09-09 to 2020-09-15 | rule |
| `train_cutoff` (T_dev) | 2020-09-09. Training reads purchases with `t_dat` < 2020-09-09 only. | rule |
| Customer bucket | `int(sha256("taste-rank/split/v1:" + customer_id)[:16 hex], 16) mod 100` | rule |
| Buckets | train 0–79, dev 80–89, heldout 90–99 | rule |
| Eligible (dev, heldout) | bucket matches, at least one purchase before the window start, at least one in the window | rule |
| Cold start | bucket matches, a window purchase, no history. Counted (`cold_start`), not evaluated. | rule |
| Train customers | train bucket, at least one purchase before `train_cutoff` | rule |

Heldout customers are never read by training or tuning, in either window. Dev customers are read only by tuning. The heldout split runs once per release.

### Canonical id-list hash

`split_ids_sha256` is the sha256 of the UTF-8 text made of one line per customer, `<split>\t<customer_id>\n`, with the splits in the order `train`, `dev`, `heldout` and the ids sorted inside each. `train` lists the train customers; `dev` and `heldout` list the eligible customers.

## Candidate pool and POP

- **Pool:** distinct articles with at least one purchase (any customer, any bucket) with `t_dat` in [window start − 84 days, window start − 1 day]. Known before the window, so no leakage. Sorted by article id.
- **POP counts:** purchases of pool articles with `t_dat` in [window start − 7 days, window start − 1 day], any customer. Articles with no purchase in that week are absent from the counts.
- **POP ranking:** count descending, ties by article id ascending. Same list for every customer.

## Customer vector and score

- **History:** the customer's purchases before the window start (for training pairs: before the target day), most recent first. Same-day purchases order by file row, later row first. The last `n_last` are kept; repeats count once per purchase.
- **Vector:** each history embedding is L2-normalized; weight `w = 0.5 ** (age_days / half_life_days)`, `age_days` = window start (or target day) minus purchase day; an infinite (`null`) half-life gives equal weights. The vector is the weighted sum, L2-normalized.
- **Score:** cosine similarity between the customer vector and each pool article's normalized image embedding. Ties by article id ascending. History articles are not removed (customers rebuy).
- **`n_last` and `half_life_days`:** no data has been read, so P2 fixes the default blind (`n_last` 20, half-life 30 days; `history_default`) and the grid (`history_grid`). P5 may choose one grid point on **dev with E0 only**, before any E1 training, logs the choice in a results row and DECISIONS.md, and uses it for every row (POP excepted) on dev and heldout. Otherwise the default stands.

### Missing images (P4, RFI-P3c-2)

The contract above assumes every article has an image. The competition's image folder may lack some. The rule, adopted from D-P3c-6, never favours an encoder row over POP:

- An article id that is not `[0-9A-Za-z_-]+` is refused (`ValueError`), since ids name image paths.
- A pool article with no image gets no score and ranks after every scored article, by id ascending. It stays in the pool, so it still counts for `pool_coverage` and as a miss.
- A history purchase with no image is skipped. `n_last` counts purchases, so a skipped one is not replaced by an older purchase.
- A customer with no history image left gets a zero vector: every score ties and the pool ranks by id. The customer stays evaluated (not dropped), and the count is reported.
- `missing_images` in each encoder row = distinct articles, pool and evaluated customers' histories together, with no image, each counted once. POP rows carry 0.
- Training pairs need every image they use: P5 drops a pair whose target or any kept history item has no image before `train`, and logs the count.

## Training pairs (E1)

For each train customer and each purchase with `t_dat` < `train_cutoff`: target = that article, history = the customer's purchases strictly before the target day (ordering above), last `n_last`. Pairs with empty history are dropped. The loss is in-batch InfoNCE: `cross_entropy(C · Tᵀ / temperature, arange(B))`, with C the normalized customer vectors (through the LoRA'd encoder) and T the normalized target embeddings. The train lane refuses any pair dated on or after `train_cutoff`.

## Metrics

R = distinct articles the customer bought in the window, in the pool or not. The ranking covers pool articles only, so out-of-pool purchases are misses.

- `ndcg@10` = DCG/IDCG; DCG = Σ over ranks i ≤ 10 of [item i in R] / log2(i + 1); IDCG = Σ over i ≤ min(|R|, 10) of 1 / log2(i + 1).
- `recall@k` = |top-k ∩ R| / |R|, for k = 12 and 50.
- `pool_coverage` = Σ |R ∩ pool| / Σ |R| over evaluated customers.
- Every metric is a mean over evaluated customers (each customer weighs the same).
- Hand-derived cases: `fixtures/metrics_cases.json`.

**Per-category tables.** For each `product_group_name` g: the customers with at least one purchase of a g article in the window; their ranking filtered to pool articles in g; R filtered to g articles (pool or not). Category comes from `articles.csv`. The category row's `pool_coverage` is `pool_coverage` over that table's customers, with R filtered to g and the pool filtered to g; `eval.sh` computes it with `pool_coverage`, and `category_metrics` keeps the return keys in `contracts/seams.md`. Scores per table, never a composite (section 6, item 6).

## Rows and bands

| Row | Encoder | Runs |
|---|---|---|
| POP | none | 1 |
| E0-siglip | base, zero-shot (models.lock) | 1, plus one repeat run that must match to 1e-6 |
| E0-fclip | second base, zero-shot | 1 |
| E1-siglip | base + LoRA | five seeds: 0, 1, 2, 3, 4 |

- **Band:** mean, min and max over the five seeds.
- **Lift (headline):** per seed s, `lift_s = E1_s − E0` on heldout `ndcg@10`, overall table. Verdict: `lift` if min > 0, `worse` if max < 0, else `no lift`. A band crossing or touching zero is reported as no lift. Fewer than five seeds is refused.
- Seeds and model revisions are pinned in `models.lock`.

## Results rows

One JSON object per line in `results/*.jsonl`; schema in `contracts/results.schema.json`. Required: the section 6 fields (`run_id`, `git_sha`, `machine`, `model_revision`, `quant`, `runtime`, `date`) and `row`, `split`, `table`, `metrics`, `n_customers`. Box rows also carry `seed`, `split_rule_sha256`, `split_ids_sha256`, `gpu` and `gpu_power_limit_w`.

- `machine` is read at run time: the contents of `$FORGE_MACHINE_FILE` when it is set and not empty, otherwise the CPU model from `/proc/cpuinfo`, total memory from `/proc/meminfo` and, when present, `nvidia-smi --query-gpu=name,power.limit`.
- A row never holds a customer id, an article id or a list of either (`FORBIDDEN_RESULT_KEYS` in `src/tasterank/seams.py`). `write_rows` refuses those keys. Aggregate counts and category names are allowed.
- Fixture runs (`scripts/eval.sh --fixture <dir>`) evaluate the fixture's dev split only, with `tasterank.tiny` as the encoder (row `E0-tiny`), and never a heldout split.

## Box values

Filled by P5 from the first box run: customer counts per split, cold-start counts, pool sizes, pool coverage, and `split_ids_sha256`. Until then: none. No count in this repo is an estimate of them.
