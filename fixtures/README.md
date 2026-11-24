# fixtures/

Written in P2. Small, committed inputs the red tests and lanes run against. Everything here is synthetic: made up by `make_fixtures.py` (seeded, deterministic), licence the repo's. Nothing is copied or derived from the competition data, which may not be redistributed (`contracts/data.md`). No customer data, no scraped marketplace data, no keys.

| Path | What |
|---|---|
| `shop_like/articles.csv` | 40 made-up articles: 4 product groups (Tops, Bottoms, Accessories, Footwear) x 5 colours x 2. Columns follow the competition's `articles.csv` (a subset). |
| `shop_like/customers.csv` | 48 made-up customers; `customer_id` is the sha256 of `fixture-customer-<i>`. |
| `shop_like/transactions_train.csv` | 449 purchases, 2020-05-01 to 2020-09-22, sorted by day like the competition file. |
| `shop_like/images/<3>/<id>.jpg` | 16x16 JPEGs: the colour group's colour plus a stripe whose position marks the product group, with seeded noise. |
| `shop_like/split.json` | The fixture's split rule: the frozen rule's dates and windows, buckets widened to train 0–49, dev 50–74, heldout 75–99 so every split has customers. |
| `metrics_cases.json` | Hand-derived nDCG@10, recall@12 and recall@50 cases with their formulas. |

Corner cases planted in `shop_like` on purpose:

- every sixth customer (i mod 6 = 4) buys only inside the windows: cold start;
- every sixth customer (i mod 6 = 5) buys only before the windows: never evaluated;
- the last article is sold only inside the heldout window, so it is never in a pool but is a miss for the customer who buys it;
- the second-to-last article is sold once, 85 days before the heldout window: outside the heldout pool, inside the dev pool.

Regenerate with `.venv/bin/python fixtures/make_fixtures.py`; `tests/contracts` fails if the committed CSVs or the rule differ from the generator.
