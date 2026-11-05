#!/usr/bin/env python3
"""Download real NYC 311 contact-center data from the NYC Open Data
Socrata API (no key required) into data/raw/.

Dataset: NYC 311 Service Requests from 2010 to Present, id erm2-nwe9,
https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9
NYC 311 is a real customer contact center (phone, online, mobile app):
each row is one customer contact with a timestamp, a contact driver
(complaint_type), a free-text descriptor, a channel
(open_data_channel_type) and a borough (market segment).

Pulls (all 2023-01-01..2024-12-31 unless noted):
  daily_totals.csv   - daily contact totals (731 rows)
  daily_by_driver.csv - daily contacts for the top 15 drivers
  daily_by_channel.csv - daily contacts by channel
  contacts_sample.csv - every individual contact 2024-01-08..2024-01-10
                        (complete days, descriptor text for NLP)
"""
import csv
import json
import time
import os
import sys
import urllib.parse
import urllib.request

BASE = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
os.makedirs(RAW, exist_ok=True)

TOP_DRIVERS = [
    "Illegal Parking", "Noise - Residential", "HEAT/HOT WATER",
    "Noise - Street/Sidewalk", "Blocked Driveway", "UNSANITARY CONDITION",
    "Request Large Bulky Item Collection", "Street Condition",
    "Water System", "Noise - Vehicle", "PLUMBING", "Noise - Commercial",
    "Abandoned Vehicle", "PAINT/PLASTER", "Noise",
]
WINDOW = "created_date between '2023-01-01T00:00:00' and '2024-12-31T23:59:59'"


def fetch(params, tries=3):
    qs = urllib.parse.urlencode(params)
    url = BASE + "?" + qs
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "contact-center-forecasting-insights/1.0"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.load(resp)
        except Exception as e:
            last = e
            print(f"  attempt {attempt + 1} failed: {e}; retrying...")
            time.sleep(5)
    raise last


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path}: {len(rows)} rows")


def main():
    # 1. daily totals
    rows = fetch({
        "$select": "date_trunc_ymd(created_date) as day, count(*) as contacts",
        "$where": WINDOW,
        "$group": "day",
        "$order": "day",
        "$limit": "50000",
    })
    write_csv(os.path.join(RAW, "daily_totals.csv"), rows, ["day", "contacts"])

    # 2. daily by top-15 drivers, split per year (Socrata times out on the
    #    full 2-year grouped query without an app token)
    in_list = "(" + ",".join("'" + d.replace("'", "''") + "'" for d in TOP_DRIVERS) + ")"
    all_rows = []
    for year in (2023, 2024):
        w = f"created_date between '{year}-01-01T00:00:00' and '{year}-12-31T23:59:59'"
        rows = fetch({
            "$select": "date_trunc_ymd(created_date) as day, complaint_type, count(*) as contacts",
            "$where": f"{w} AND complaint_type in {in_list}",
            "$group": "day, complaint_type",
            "$order": "day, complaint_type",
            "$limit": "50000",
        })
        print(f"  {year}: {len(rows)} driver-day rows")
        all_rows.extend(rows)
    write_csv(os.path.join(RAW, "daily_by_driver.csv"), all_rows, ["day", "complaint_type", "contacts"])

    # 3. daily by channel, split per year
    all_rows = []
    for year in (2023, 2024):
        w = f"created_date between '{year}-01-01T00:00:00' and '{year}-12-31T23:59:59'"
        rows = fetch({
            "$select": "date_trunc_ymd(created_date) as day, open_data_channel_type, count(*) as contacts",
            "$where": w,
            "$group": "day, open_data_channel_type",
            "$order": "day, open_data_channel_type",
            "$limit": "50000",
        })
        print(f"  {year}: {len(rows)} channel-day rows")
        all_rows.extend(rows)
    write_csv(os.path.join(RAW, "daily_by_channel.csv"), all_rows, ["day", "open_data_channel_type", "contacts"])

    # 4. raw individual contacts, complete days 2024-01-08..2024-01-10
    rows = fetch({
        "$select": "unique_key,created_date,agency,complaint_type,descriptor,borough,open_data_channel_type,status,resolution_description",
        "$where": "created_date between '2024-01-08T00:00:00' and '2024-01-10T23:59:59'",
        "$order": "created_date",
        "$limit": "50000",
    })
    write_csv(os.path.join(RAW, "contacts_sample.csv"), rows, [
        "unique_key", "created_date", "agency", "complaint_type", "descriptor",
        "borough", "open_data_channel_type", "status", "resolution_description",
    ])
    print("done")


if __name__ == "__main__":
    main()
