"""Builds fixtures/shop_like/: a synthetic catalogue shaped like the competition files (contracts/data.md).

Every article, customer, purchase and image here is made up by this script
(seeded, deterministic). Nothing is copied or derived from the competition
data, which may not be redistributed (SPEC.md, "Data"). Column names and the
image folder layout follow the competition's public data description so the
same loaders read both.

    .venv/bin/python fixtures/make_fixtures.py          # rewrite fixtures/shop_like/
    .venv/bin/python fixtures/make_fixtures.py --check  # exit 1 if the CSVs differ

Corner cases planted on purpose are listed in fixtures/README.md.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import random
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "shop_like"

GROUPS = ["Tops", "Bottoms", "Accessories", "Footwear"]
COLOURS = {"Red": (200, 40, 40), "Blue": (40, 60, 200), "Green": (40, 160, 70), "Black": (25, 25, 25), "White": (235, 235, 235)}
START = dt.date(2020, 5, 1)
END = dt.date(2020, 9, 22)
HELDOUT_START = dt.date(2020, 9, 16)
DEV_START = dt.date(2020, 9, 9)

RULE = {
    "version": 1,
    "dataset": "fixtures/shop_like (synthetic)",
    "anchor": {"date_column": "t_dat", "expected_min_t_dat": START.isoformat(), "expected_max_t_dat": END.isoformat()},
    "windows": {"heldout": {"start": "2020-09-16", "end": "2020-09-22"}, "dev": {"start": "2020-09-09", "end": "2020-09-15"}},
    "train_cutoff": "2020-09-09",
    "pool_days_before_window": 84,
    "pop_days_before_window": 7,
    "customer_hash": {
        "algorithm": "sha256",
        "message": "<salt>:<customer_id>, UTF-8",
        "salt": "taste-rank/split/v1",
        "bucket": "int(first 16 hex digits of the digest, base 16) mod 100",
        "buckets": {"train": [0, 49], "dev": [50, 74], "heldout": [75, 99]},
    },
    "eligible": "bucket matches the split, at least one purchase before window start, at least one purchase in [start, end]",
    "history_grid": {"n_last": [2, 5], "half_life_days": [7, None]},
    "history_default": {"n_last": 5, "half_life_days": 30},
    "seeds": [0, 1, 2, 3, 4],
}


def bucket(customer_id: str, salt: str = "taste-rank/split/v1") -> int:
    return int(hashlib.sha256(f"{salt}:{customer_id}".encode()).hexdigest()[:16], 16) % 100


def split_of(customer_id: str) -> str:
    b = bucket(customer_id)
    for name, (lo, hi) in RULE["customer_hash"]["buckets"].items():
        if lo <= b <= hi:
            return name
    raise AssertionError(b)


def articles() -> list[dict[str, str]]:
    rows = []
    n = 0
    for gi, group in enumerate(GROUPS):
        for ci, colour in enumerate(COLOURS):
            for v in range(2):
                n += 1
                aid = f"0{100000000 + gi * 1000 + ci * 10 + v:09d}"[-10:]
                rows.append(
                    {
                        "article_id": aid,
                        "product_code": aid[:7],
                        "prod_name": f"Fixture {group[:-1] if group.endswith('s') else group} {colour} {v}",
                        "product_type_name": f"{group} type {v}",
                        "product_group_name": group,
                        "colour_group_name": colour,
                        "detail_desc": f"A made-up {colour.lower()} item in {group.lower()}.",
                    }
                )
    return rows


def image_bytes(article: dict[str, str]) -> bytes:
    from PIL import Image

    rng = random.Random(article["article_id"])
    base = COLOURS[article["colour_group_name"]]
    stripe = GROUPS.index(article["product_group_name"])
    img = Image.new("RGB", (16, 16))
    px = img.load()
    for y in range(16):
        for x in range(16):
            on = (x // 4) == stripe
            c = tuple(max(0, min(255, (255 - v if on else v) + rng.randint(-12, 12))) for v in base)
            px[x, y] = c
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def build() -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(7)
    arts = articles()
    by_taste: dict[tuple[str, str], list[str]] = {}
    for a in arts:
        by_taste.setdefault((a["product_group_name"], a["colour_group_name"]), []).append(a["article_id"])
    # The last article of the catalogue is sold only inside the heldout window
    # (never in any pool); the one before it only on 2020-06-23, 85 days before
    # the heldout window (outside the 84-day pool) and 78 days before dev (inside).
    window_only = arts[-1]["article_id"]
    stale = arts[-2]["article_id"]
    regular = [a["article_id"] for a in arts[:-2]]

    customers = []
    tx: list[tuple[dt.date, str, str, float, int]] = []
    days = (END - START).days
    for i in range(48):
        cid = hashlib.sha256(f"fixture-customer-{i}".encode()).hexdigest()
        customers.append({"customer_id": cid, "age": str(18 + (i * 7) % 50)})
        taste = (GROUPS[i % 4], list(COLOURS)[(i // 4) % 5])
        liked = [a for a in by_taste[taste] if a in regular] or regular[:2]
        kind = i % 6  # 0-3 regular, 4 cold start (window only), 5 history only
        if kind != 4:
            for _ in range(rng.randint(4, 12)):
                d = START + dt.timedelta(days=rng.randint(0, (DEV_START - START).days - 1))
                aid = rng.choice(liked) if rng.random() < 0.7 else rng.choice(regular)
                tx.append((d, cid, aid, round(rng.uniform(0.005, 0.08), 6), rng.choice([1, 2])))
        if kind != 5:
            for start in (DEV_START, HELDOUT_START):
                for _ in range(rng.randint(1, 3)):
                    d = start + dt.timedelta(days=rng.randint(0, 6))
                    aid = rng.choice(liked) if rng.random() < 0.6 else rng.choice(regular)
                    tx.append((d, cid, aid, round(rng.uniform(0.005, 0.08), 6), rng.choice([1, 2])))
    first = customers[0]["customer_id"]
    tx.append((HELDOUT_START + dt.timedelta(days=2), first, window_only, 0.05, 2))
    tx.append((HELDOUT_START - dt.timedelta(days=85), customers[1]["customer_id"], stale, 0.03, 1))
    # The file is sorted by date like the competition file; within a day, by generation order.
    order = sorted(range(len(tx)), key=lambda k: (tx[k][0], k))
    tx_rows = [
        {"t_dat": tx[k][0].isoformat(), "customer_id": tx[k][1], "article_id": tx[k][2], "price": f"{tx[k][3]:.6f}", "sales_channel_id": str(tx[k][4])}
        for k in order
    ]
    return arts, customers, tx_rows


def csv_text(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def outputs() -> dict[str, str]:
    arts, customers, tx = build()
    # Pin the anchor: make sure the last day of the file is END.
    if tx[-1]["t_dat"] != END.isoformat():
        tx.append({"t_dat": END.isoformat(), "customer_id": customers[0]["customer_id"], "article_id": arts[0]["article_id"], "price": "0.010000", "sales_channel_id": "2"})
    return {
        "articles.csv": csv_text(arts),
        "customers.csv": csv_text(customers),
        "transactions_train.csv": csv_text(tx),
        "split.json": json.dumps(RULE, indent=2) + "\n",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    outs = outputs()
    if args.check:
        bad = [n for n, t in outs.items() if (OUT / n).read_text() != t]
        if bad:
            print("fixtures differ from the generator:", ", ".join(bad))
            return 1
        print("fixtures match the generator")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    for n, t in outs.items():
        (OUT / n).write_text(t)
    for a in articles():
        p = OUT / "images" / a["article_id"][:3] / f"{a['article_id']}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(image_bytes(a))
    tx = list(csv.DictReader(io.StringIO(outs["transactions_train.csv"])))
    counts: dict[str, int] = {}
    for c in {r["customer_id"] for r in tx}:
        counts[split_of(c)] = counts.get(split_of(c), 0) + 1
    print(f"wrote {OUT}: {len(articles())} articles, {len(tx)} transactions, customers by bucket {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
