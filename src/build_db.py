#!/usr/bin/env python3
"""Ingest the downloaded NYC 311 CSVs into SQLite (data/contacts.db)."""
import os
import sqlite3

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
DB = os.path.join(ROOT, "data", "contacts.db")


def main():
    if os.path.exists(DB):
        os.remove(DB)
    con = sqlite3.connect(DB)
    totals = pd.read_csv(os.path.join(RAW, "daily_totals.csv"), parse_dates=["day"])
    totals.to_sql("daily_totals", con, index=False)
    drivers = pd.read_csv(os.path.join(RAW, "daily_by_driver.csv"), parse_dates=["day"])
    drivers.to_sql("daily_by_driver", con, index=False)
    channels = pd.read_csv(os.path.join(RAW, "daily_by_channel.csv"), parse_dates=["day"])
    channels.to_sql("daily_by_channel", con, index=False)
    sample = pd.read_csv(os.path.join(RAW, "contacts_sample.csv"))
    sample["created_date"] = pd.to_datetime(sample["created_date"])
    sample.to_sql("contacts_sample", con, index=False)
    con.execute("CREATE INDEX idx_totals_day ON daily_totals(day)")
    con.execute("CREATE INDEX idx_driver_day ON daily_by_driver(day, complaint_type)")
    con.execute("CREATE INDEX idx_channel_day ON daily_by_channel(day, open_data_channel_type)")
    con.commit()
    for table in ("daily_totals", "daily_by_driver", "daily_by_channel", "contacts_sample"):
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}: {n} rows")
    con.close()
    print(f"wrote {DB}")


if __name__ == "__main__":
    main()
