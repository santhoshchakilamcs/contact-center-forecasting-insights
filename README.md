# Contact-Center Demand Forecasting & Insights

A personal project that does the two halves of the Netflix **Customer Service
Analytics & Insights Analyst** role on real public data: (1) demand-forecast
judgment — interpret a forecast, apply event overlays, explain variances and
root causes; (2) CS insights — contact-driver analysis, emerging friction
points, and NLP over real contact text.

**Data is real, not synthetic.** The dataset is NYC 311 service requests —
a real customer contact center (phone, online, mobile app) run by the City
of New York — pulled from the NYC Open Data Socrata API
(`data.cityofnewyork.us`, dataset `erm2-nwe9`, "311 Service Requests from
2010 to Present"). No contact, volume, or text in this project is generated.

## Verified results

All numbers below were produced by `src/pipeline.py` and independently
recomputed by `verify.py` (which re-reads the raw CSVs and cross-checks
`reports/metrics.json`).

| Fact | Value |
|---|---|
| Date range | 2023-01-01 .. 2024-12-31 (731 days, no gaps) |
| Total contacts | 6,681,493 (mean 9,140/day) |
| Individual contacts with text | 30,470 (2024-01-08..10), 30,224 with descriptor text |
| Drivers in daily file | 14 (top 15 queried; "Request Large Bulky Item Collection" returned no rows in the window) |
| Top 3 drivers | Illegal Parking 982,537; Noise - Residential 677,749; HEAT/HOT WATER 494,259 |
| Channel mix | ONLINE 41.4%, PHONE 30.5%, MOBILE 20.2%, UNKNOWN 7.8% |
| Train / test | Train 2023-01-01..2024-09-30; test 2024-10-01..2024-12-31 (92 days) |
| **Total demand, calendar-ridge model, test MAPE** | **10.9%** (WAPE 11.1%, bias -2.5%) |
| Total demand, seasonal-naive baseline, test MAPE | 15.4% (WAPE 15.4%, bias -0.2%) |
| Best drivers (ridge test MAPE) | Blocked Driveway 7.3%, Illegal Parking 9.5% |
| Worst drivers (ridge test MAPE) | HEAT/HOT WATER 87.6% (WAPE 50.2%), Noise - Residential 40.3% |

### Event overlays (measured uplift vs seasonal naive, train period)

| Event | Dates | Mean uplift |
|---|---|---|
| Jan 2024 coastal storm (heavy rain, flooding) | 2024-01-09..10 | +8.7% (2 days) |
| Remnants of Hurricane Debby | 2024-08-09 | +14.2% (1 day) |
| Feb 2024 snowstorm | 2024-02-13 | -22.2% (1 day — contacts fell) |
| Jul 2024 severe thunderstorms | 2024-07-16 | -3.1% (1 day) |

Single-day uplifts are noisy; scenario bands for one-day events use a flat
±5pp band and say so. Scenario example: a severe winter storm on
2025-01-09..10 re-using the Jan 2024 storm uplift forecasts 20,177 contacts
(range 18,881–21,473) against a same-weekday-last-year base of 18,564.

### Insights

- **Emerging drivers (last 90 days vs prior 90):** HEAT/HOT WATER share
  rose 18.9pp — but the recent window is Oct–Dec (heating season) and the
  prior window Jul–Sep, so this is mostly **seasonality, not a new trend**;
  compare same-quarter last year before acting on it. Noise - Street/Sidewalk
  fell 7.1pp for the same seasonal reason.
- **Channel shift:** ONLINE 39.5% → 40.0%, PHONE 30.9% → 29.9%,
  MOBILE 22.1% → 23.3% — small, no dramatic channel migration.
- **Largest forecast misses (test):** under-forecasts cluster in Nov–Dec
  (e.g. 2024-12-16: actual 14,536 vs forecast 9,821, -32%); Thanksgiving
  Day (Nov 28) was over-forecast by 50% — the model has no holiday feature.
  Those are the misses an analyst would investigate first.
- **Keywords (TF-IDF over real descriptor text):** Illegal Parking →
  blocked, parking, hydrant, posted, sign; Noise - Residential → banging,
  pounding, loud, party, music; HEAT/HOT WATER → entire, building,
  apartment; Rodent → sighting, rat, rodents.

## How it works

- `src/download_data.py` — downloads the four CSVs from the Socrata API
  (no key; run it to refresh, or use the files already in `data/raw/`).
  The 2-year grouped queries are split per year because Socrata times out
  on the full grouped query without an app token.
- `src/build_db.py` — ingests the CSVs into SQLite `data/contacts.db`.
- `src/forecast.py` — seasonal-naive baseline vs calendar-ridge regression
  (day-of-week, annual sin/cos, trend, lags 7/14/28, 7-day rolling mean;
  numpy least squares with a small ridge penalty). Also fits per-driver
  (top 5) and per-channel models, and measures event uplifts.
- `src/insights.py` — driver mix, emerging drivers, channel shift, and
  pure-Python TF-IDF keywords per driver over real descriptor text.
- `src/pipeline.py` — runs everything, applies measured event uplifts to
  the future scenarios in `data/events.json`, writes `reports/metrics.json`
  and `reports/dashboard.html`.
- `app.py` — stdlib-only HTTP server: `/` serves the dashboard,
  `/metrics` serves `reports/metrics.json`.
- `verify.py` — recomputes headline metrics from the raw CSVs and
  cross-checks `reports/metrics.json`.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/download_data.py   # refresh data (optional; files included)
.venv/bin/python src/build_db.py
.venv/bin/python src/pipeline.py        # builds reports/metrics.json + reports/dashboard.html
.venv/bin/python verify.py
.venv/bin/python -m pytest tests/ -q   # 7 tests
PORT=8000 .venv/bin/python app.py       # http://127.0.0.1:8000/
```

## Honest limits

- **This is a personal project on public NYC 311 data. It is not Netflix
  data, not a production forecast, and not contact-center job experience.**
- NYC 311 is a city-service contact center, not a subscription streaming
  service: its drivers (parking, noise, heat) and volumes do not transfer
  to Netflix. Nothing here claims otherwise.
- The ridge model works for total demand and stable drivers but **fails on
  weather-driven drivers** (HEAT/HOT WATER test MAPE 87.6%) — the model has
  no weather inputs, and that gap is stated, not hidden.
- **No handle-time (AHT), workload, staffing, or multi-vendor data exists
  in this dataset**, so none of it is modeled or claimed.
- Event uplifts rest on 1–2 days each and are noisy; the Nov 2024 wildfire
  smoke event falls inside the test window and is reported only, never used
  to fit an overlay.
- The individual-contact sample covers 3 days (2024-01-08..10) only, so
  keyword findings describe that window, not the full two years.
- Kaggle's "Customer Subscription Data" (a subscription-product call-center
  set) would have been a closer match, but it requires a Kaggle account and
  API key to download, which was not available here — so NYC 311, which is
  downloadable without authentication, was used instead, and the choice is
  recorded here rather than hidden.

## Repo

https://github.com/santhoshchakilamcs/contact-center-forecasting-insights
