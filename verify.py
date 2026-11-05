#!/usr/bin/env python3
"""Recompute the headline metrics straight from the raw CSVs (not from
reports/metrics.json) and print them, so the README numbers can be
checked independently of the pipeline.
"""
import csv
import json
import os
from datetime import date

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")


def read_daily_totals():
    rows = []
    with open(os.path.join(RAW, "daily_totals.csv")) as f:
        for r in csv.DictReader(f):
            rows.append((date.fromisoformat(r["day"][:10]), int(r["contacts"])))
    return sorted(rows)


def main():
    totals = read_daily_totals()
    by_day = dict(totals)
    total = sum(v for _, v in totals)
    print(f"days: {len(totals)}  range: {totals[0][0]}..{totals[-1][0]}  total contacts: {total:,}")
    # seasonal naive on the test window 2024-10-01..2024-12-31
    test = [(d, v) for d, v in totals if date(2024, 10, 1) <= d <= date(2024, 12, 31)]
    errs, abs_errs = [], []
    for d, v in test:
        from datetime import timedelta
        naive = by_day.get(d - timedelta(days=7))
        if naive is None:
            continue
        errs.append(abs(naive - v) / v)
        abs_errs.append(abs(naive - v))
    print(f"test days: {len(test)}  seasonal-naive MAPE: {sum(errs)/len(errs):.4f}  "
          f"WAPE: {sum(abs_errs)/sum(v for _, v in test):.4f}")
    # top drivers from the raw driver file
    from collections import defaultdict
    drv = defaultdict(int)
    with open(os.path.join(RAW, "daily_by_driver.csv")) as f:
        for r in csv.DictReader(f):
            drv[r["complaint_type"]] += int(r["contacts"])
    top3 = sorted(drv.items(), key=lambda kv: -kv[1])[:3]
    print("top 3 drivers:", [(k, f"{v:,}") for k, v in top3])
    # channel shares
    ch = defaultdict(int)
    with open(os.path.join(RAW, "daily_by_channel.csv")) as f:
        for r in csv.DictReader(f):
            ch[r["open_data_channel_type"]] += int(r["contacts"])
    ch_total = sum(ch.values())
    print("channel shares:", {k: round(v / ch_total, 4) for k, v in sorted(ch.items(), key=lambda kv: -kv[1])})
    # sample rows
    with open(os.path.join(RAW, "contacts_sample.csv")) as f:
        sample = list(csv.DictReader(f))
    with_text = sum(1 for r in sample if r.get("descriptor"))
    print(f"sample contacts: {len(sample)}  with descriptor text: {with_text} ({with_text/len(sample):.1%})")
    # cross-check against reports/metrics.json
    with open(os.path.join(ROOT, "reports", "metrics.json")) as f:
        m = json.load(f)
    assert m["dataset"]["total_contacts"] == total, "metrics.json total mismatch"
    assert abs(m["forecast"]["total"]["seasonal_naive"]["mape"] - sum(errs) / len(errs)) < 1e-9, "naive MAPE mismatch"
    print("cross-check vs reports/metrics.json: OK")


if __name__ == "__main__":
    main()
