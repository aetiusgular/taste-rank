# Data contract

## The competition files (box only)

H&M Personalized Fashion Recommendations, Kaggle. Downloaded on the box by the host with Tony's Kaggle token after the rules are accepted (P5, needs-tony). Path given to the pipeline as `$TASTE_RANK_DATA` (a box-local folder, never inside the repo).

| File | Columns the build reads |
|---|---|
| `transactions_train.csv` | `t_dat` (YYYY-MM-DD), `customer_id` (64 hex), `article_id` (10 digits, leading zero), `price`, `sales_channel_id` |
| `articles.csv` | `article_id`, `product_group_name` (others ignored) |
| `customers.csv` | `customer_id` (not used for features; the split hash needs only the id) |
| `images/<first 3 digits>/<article_id>.jpg` | article photo; some articles have none |

An article with no image is dropped from the pool and the history, and the count dropped is a results-row field (`missing_images`). Its purchases still count in R. This is a P5 pipeline rule: every fixture article has an image, so no P3 red test covers it.

Terms (research/P1.md section 5): non-commercial, academic research and education; no redistribution. So no row, image, id, id list or derived table from these files enters the repo, a results row, a PR, a log posted to GitHub, or any public artifact. Trained weights stay on the box.

## Fixtures (in the repo)

`fixtures/shop_like/` has the same file names, columns and image layout, all synthetic (`fixtures/make_fixtures.py`, `fixtures/README.md`). Its own rule is `fixtures/shop_like/split.json`: the frozen rule's dates and windows, with wider dev and heldout buckets so 48 customers fill every split.
