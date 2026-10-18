"""A small synthetic shop in the raw Online Retail II layout, for tests and a local dry run.

Customers differ in how often they order and what they buy, and some stop ordering
part-way through, so the pipeline has real signal to find. It is not a model of the
real data; it only exists so the code can run without downloading anything.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

PRODUCTS = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
    ("22423", "REGENCY CAKESTAND 3 TIER", 12.75),
    ("85099B", "JUMBO BAG RED RETROSPOT", 1.95),
    ("20725", "LUNCH BAG RED RETROSPOT", 1.65),
    ("21212", "PACK OF 72 RETROSPOT CAKE CASES", 0.55),
    ("22086", "PAPER CHAIN KIT 50'S CHRISTMAS", 2.95),
    ("84879", "ASSORTED COLOUR BIRD ORNAMENT", 1.69),
    ("22197", "POPCORN HOLDER", 0.85),
    ("21034", "REX CASH+CARRY JUMBO SHOPPER", 0.95),
    ("22720", "SET OF 3 CAKE TINS PANTRY DESIGN", 4.95),
]


def raw_transactions(customers: int = 300, start: str = "2009-12-01", end: str = "2011-12-09",
                     seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0, t1 = datetime.fromisoformat(start), datetime.fromisoformat(end)
    span = (t1 - t0).days
    rows, invoice = [], 489434
    for c in range(customers):
        cid = 12346 + c
        rate = rng.gamma(1.2, 1.0) / 30  # orders per day
        first = rng.integers(0, span // 2)
        last = span if rng.random() < 0.6 else rng.integers(first + 30, span)  # some customers stop
        likes = rng.dirichlet(np.ones(len(PRODUCTS)) * 0.5)
        country = "United Kingdom" if rng.random() < 0.85 else str(rng.choice(["Germany", "France", "EIRE"]))
        day = float(first)
        while True:
            day += rng.exponential(1 / rate)
            if day >= last:
                break
            ts = t0 + timedelta(days=day, hours=int(rng.integers(8, 18)))
            invoice += 1
            for i in rng.choice(len(PRODUCTS), size=rng.integers(1, 5), replace=False, p=likes):
                code, desc, price = PRODUCTS[i]
                qty = int(rng.integers(1, 24))
                rows.append((str(invoice), code, desc, qty, ts, price, float(cid), country))
                if rng.random() < 0.03:  # occasional return
                    rows.append((f"C{invoice}", code, desc, -qty, ts + timedelta(days=3), price, float(cid), country))
    # noise the cleaning step must remove
    rows.append(("999999", "POST", "POSTAGE", 1, t0 + timedelta(days=10), 18.0, 12346.0, "Germany"))
    rows.append(("999998", "22423", "REGENCY CAKESTAND 3 TIER", 1, t0 + timedelta(days=11), 12.75, np.nan, "United Kingdom"))
    rows.append(("999997", "22423", "REGENCY CAKESTAND 3 TIER", 1, t0 + timedelta(days=12), 0.0, 12347.0, "United Kingdom"))
    return pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price",
                                       "Customer ID", "Country"])
