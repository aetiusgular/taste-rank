# SPEC: taste-rank

Written in P1 (2026-11-23). Research, sources and the scored pick are in `research/P1.md`. Later phases do not reopen the pick (BUILD_OUTLINE.md section 2); a review finding with evidence can ask, and reopening is Tony's call.

## What it is

A personalized ranker for a fashion catalogue. A customer's past purchases, embedded by an image-text encoder, rank the catalogue for that customer. The encoder is LoRA-fine-tuned so that history embeddings land near what the customer buys next. The evaluation follows the derm method: zero-shot E0 first, a frozen held-out split, five-seed bands, per-category tables, published misses.

Why this build: Krea is hiring an ML engineer to build personalization "from the ground up", including "curated feeds" and "ranking"; its public product has no taste learned from behaviour and no per-viewer feed (`research/P1.md` sections 2 and 3).

## Headline number

**nDCG@10 lift, E1 minus E0, on held-out customers in the held-out window, same base model, mean and min–max band over five seeds.**

- Ground truth: the customers' recorded purchases in the held-out window. Recorded by the retailer; the build never produces or edits them.
- E0 and E1 run on the same customers, candidate pool and history window. Only the encoder weights differ.
- A lift whose band crosses zero is reported as no lift. A negative result ships as a negative result.

## Data

- H&M Personalized Fashion Recommendations (Kaggle): `articles.csv`, `customers.csv`, `transactions_train.csv`, article images. Downloaded on the box by the host with Tony's Kaggle token (P5 needs-tony item). Never in the repo.
- Terms: non-commercial use, academic research and education; no redistribution (`research/P1.md` section 5). So: no H&M row, image, id list or derived table in the repo, a results row, a PR or any public artifact. Trained weights are not published.
- Tables are read with polars or DuckDB, not pandas.
- Grailed data is never used.

## Split (rule here; exact dates, ids and hash frozen in P2)

- **Time:** a cutoff date T. History window: purchases before T. Held-out window: the 7 days from T (the competition's own horizon). A dev cutoff T_dev = T − 7 days gives the dev split, same shape.
- **Customers:** customers with at least one purchase in the history window and at least one in the held-out window are split by a fixed hash of `customer_id` into train (fine-tuning only), dev and heldout. Heldout customers are never seen in training, in either window.
- **Training data** comes only from train customers and only from before T_dev. Tuning reads dev only. Heldout runs once per release (BUILD_OUTLINE.md section 6).
- P2 wrote `contracts/eval.md` and `contracts/split.json`: T, T_dev, the hash rule and the pool and POP windows. No data has been read yet, so `split-sha256` covers the rule file, and the id-list hash and counts come from the first box run (P5), carried in every results row as `split_ids_sha256`. The id lists stay on the box, since they are derived from the competition data.

## Ranking task

- **Candidate pool:** articles with at least one sale in the 12 weeks before the window starts. Known before the window, so no leakage. Purchases outside the pool count as misses, and the pool coverage of held-out purchases is reported.
- **Customer vector:** recency-weighted mean of the encoder's image embeddings of the customer's last N history purchases. P2 had no data, so N and the decay have a blind default and a fixed grid in `contracts/split.json`; P5 may pick one grid point on dev with E0 only, before any E1 run (`contracts/eval.md`).
- **Score:** cosine similarity between the customer vector and each candidate's image embedding. Top 12 and top 50 kept.
- **Relevant set:** distinct articles the customer bought in the window.

## Models

| Role | Model | Licence | Revision |
|---|---|---|---|
| Base (headline) | Marqo/marqo-fashionSigLIP | Apache-2.0 | `c56244cc94f92419e8369fa71efdaf403b124ce8` |
| Second E0 row | patrickjohncyh/fashion-clip | MIT | `7e3ba62ce16b379a1ab479346b66f192e76f51b7` |

P2 writes `models.lock` with both commits. The SigLIP model's remote code is read and pinned before it runs (P4 security review).

**E1 training:** LoRA on the image tower's attention projections. Contrastive (in-batch InfoNCE) objective: a train customer's history vector (built as above, through the LoRA'd encoder) against the image embedding of an article that customer bought next. Rank, learning rate and steps tuned on dev only. Five seeds. Runs on the box's 3090; power limit and card recorded in every results row.

## Evaluation rows (scores per table, never a composite)

| Row | What |
|---|---|
| POP | Most-bought pool articles in the last week of history, same for everyone |
| E0-siglip | Zero-shot base model, headline base |
| E0-fclip | Zero-shot second base model |
| E1-siglip | LoRA, five seeds |

Metrics: nDCG@10, recall@12, recall@50, pool coverage. Tables: overall, and per `product_group_name` (the article table's product group). Misses: at least one written-up failure case (customers or categories where E1 is worse than E0 or POP), described in aggregate without H&M rows.

Every number is a `results/*.jsonl` row with run_id, git_sha, machine (read at run time), model_revision, quant, runtime, date (`results/README.md`).

## Lanes (P3, one after another)

| Lane | Owns (paths set in P2) | Done when |
|---|---|---|
| P3a data | Ingest (polars/DuckDB), split builder, pool, customer histories; CPU-tested on synthetic fixtures | its red tests green |
| P3b train | LoRA training loop, customer-vector builder, seeds, checkpointing to a box-local path | its red tests green on a tiny CPU model |
| P3c eval | Ranking, metrics (nDCG, recall, coverage), per-category tables, results rows, `scripts/eval.sh` wiring | its red tests green; metrics match hand-computed fixture values |

Fixtures are synthetic: made-up articles, customers and images generated in P2, so the public repo holds no competition data.

## Out of scope

- No UI (no P2w).
- No serving or latency target. recs-engine's 5 ms retrieval SLO is not claimed here.
- No generated-image data. The transfer to a generated-image feed is a written argument in P6, labelled as such.
- No published weights, no published H&M data.

## Open items for Tony

- Kaggle token on the box, with the competition rules accepted (needed by P5). Key id, host and scope go in the P5 PR.
- Whether a public portfolio repo plus an outreach note fits the rules' "non-commercial ... academic research and education" (decide by P6).
