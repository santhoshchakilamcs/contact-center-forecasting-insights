#!/usr/bin/env python3
"""End-to-end pipeline: build the SQLite DB (if missing), run the
forecast and the insights analysis, apply event overlays to the named
future scenarios, and write reports/metrics.json + reports/dashboard.html.

Event overlays: for each named real event in data/events.json that ends
before the test window, the measured uplift vs the seasonal naive
baseline is the overlay. A scenario re-uses the uplift of the historical
event it names (based_on) and quantifies a range: base forecast x
(1 + uplift) with a low/high band of +/- one standard error of the
daily uplift ratios (approximated from the event days).
"""
import json
import os
import sqlite3

import pandas as pd

import forecast as fc
import insights as ins

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, "reports")
os.makedirs(REPORTS, exist_ok=True)


def scenario_overlays(totals_df, events):
    """Apply measured train-period event uplifts to future scenarios."""
    scenarios = json.load(open(os.path.join(ROOT, "data", "events.json")))["scenarios"]
    full = totals_df.set_index("day")["contacts"]
    by_name = {e["name"]: e for e in events}
    out = []
    for sc in scenarios:
        base_ev = by_name.get(sc["based_on"])
        if not base_ev:
            continue
        start, end = pd.Timestamp(base_ev["start"]), pd.Timestamp(base_ev["end"])
        days = pd.date_range(start, end)
        ratios = []
        for d in days:
            a = full.get(d, float("nan"))
            b = full.get(d - pd.Timedelta(days=7), float("nan"))
            if pd.notna(a) and pd.notna(b) and b > 0:
                ratios.append(a / b - 1)
        if not ratios:
            continue
        uplift = float(pd.Series(ratios).mean())
        if len(ratios) > 1:
            se = float(pd.Series(ratios).std() / (len(ratios) ** 0.5))
        else:
            # one event day gives no spread to measure: use a flat +/-5pp
            # band and say so, rather than pretending the uplift is exact.
            se = 0.05
        # base demand for the scenario dates: same weekday one year earlier
        sc_days = pd.date_range(sc["start"], sc["end"])
        base_total = sum(float(full.get(d - pd.Timedelta(days=364), 0) or 0) for d in sc_days)
        out.append({
            "scenario": sc["name"],
            "dates": f"{sc['start']}..{sc['end']}",
            "based_on": sc["based_on"],
            "measured_uplift_pct": round(uplift * 100, 1),
            "base_demand_same_weekday_last_year": int(base_total),
            "overlay_forecast_low": int(base_total * (1 + uplift - se)),
            "overlay_forecast": int(base_total * (1 + uplift)),
            "overlay_forecast_high": int(base_total * (1 + uplift + se)),
        })
    return out


def dashboard_html(metrics):
    tot = metrics["forecast"]["total"]
    rows = ""
    for r in metrics["forecast"]["per_driver"]:
        rows += (f"<tr><td>{r['label']}</td><td>{r['calendar_ridge']['mape']:.1%}</td>"
                 f"<td>{r['calendar_ridge']['wape']:.1%}</td><td>{r['seasonal_naive']['mape']:.1%}</td>"
                 f"<td>{r['seasonal_naive']['wape']:.1%}</td></tr>\n")
    ch_rows = ""
    for r in metrics["forecast"]["per_channel"]:
        ch_rows += (f"<tr><td>{r['label']}</td><td>{r['calendar_ridge']['mape']:.1%}</td>"
                    f"<td>{r['calendar_ridge']['wape']:.1%}</td><td>{r['seasonal_naive']['mape']:.1%}</td>"
                    f"<td>{r['seasonal_naive']['wape']:.1%}</td></tr>\n")
    ev_rows = ""
    for e in metrics["forecast"]["events"]:
        ev_rows += (f"<tr><td>{e['name']}</td><td>{e['start']}..{e['end']}</td>"
                    f"<td>{e['mean_uplift']:+.1%}</td><td>{e['actual_total']:,}</td></tr>\n")
    warn = metrics["insights"]["emerging_drivers"].get("seasonality_warning", "")
    em_rows = ""
    for e in metrics["insights"]["emerging_drivers"]["rising"]:
        em_rows += f"<tr><td>{e['driver']}</td><td>{e['share_prior_pct']:.2%}</td><td>{e['share_recent_pct']:.2%}</td><td>{e['change_pp']:+.2f} pp</td></tr>\n"
    sc_rows = ""
    for s in metrics["scenarios"]:
        sc_rows += (f"<tr><td>{s['scenario']}</td><td>{s['dates']}</td><td>{s['measured_uplift_pct']:+.1f}%</td>"
                    f"<td>{s['overlay_forecast_low']:,} - {s['overlay_forecast_high']:,}</td></tr>\n")
    worst_rows = ""
    for w in metrics["forecast"]["worst_test_days"]:
        worst_rows += (f"<tr><td>{w['date']}</td><td>{w['actual']:,}</td><td>{w['forecast']:,.0f}</td>"
                       f"<td>{w['error_pct']:+.1%}</td></tr>\n")
    kw_rows = ""
    for drv, kws in metrics["insights"]["keywords"]["keywords_by_driver"].items():
        kw_rows += f"<tr><td>{drv}</td><td>{', '.join(kws)}</td></tr>\n"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Contact-Center Demand Forecasting &amp; Insights</title>
<style>body{{font-family:system-ui,sans-serif;max-width:960px;margin:2em auto;padding:0 1em;line-height:1.5}}
table{{border-collapse:collapse;margin:1em 0}}td,th{{border:1px solid #ccc;padding:4px 10px;text-align:left}}th{{background:#f0f0f0}}
h1{{font-size:1.5em}}h2{{font-size:1.2em;margin-top:1.5em}}.note{{background:#fff8e1;padding:0.8em;border:1px solid #e6c200}}</style></head><body>
<h1>Contact-Center Demand Forecasting &amp; Insights</h1>
<p>Personal project on real public data: NYC 311 service requests (data.cityofnewyork.us, dataset erm2-nwe9),
{metrics['dataset']['total_contacts']:,} contacts over {metrics['dataset']['n_days']} days
({metrics['dataset']['date_range']}). Train 2023-01-01..2024-09-30, test 2024-10-01..2024-12-31 (92 days).
This is <b>not</b> Netflix data and <b>not</b> a production forecast.</p>
<h2>Demand forecast, test window (92 days)</h2>
<table><tr><th>Model</th><th>MAPE</th><th>WAPE</th><th>Bias</th><th>Test total forecast</th></tr>
<tr><td>Calendar ridge</td><td>{tot['calendar_ridge']['mape']:.1%}</td><td>{tot['calendar_ridge']['wape']:.1%}</td><td>{tot['calendar_ridge']['bias']:+.1%}</td><td>{tot['test_ridge_total']:,.0f} vs {tot['test_actual_total']:,} actual</td></tr>
<tr><td>Seasonal naive (same weekday last week)</td><td>{tot['seasonal_naive']['mape']:.1%}</td><td>{tot['seasonal_naive']['wape']:.1%}</td><td>{tot['seasonal_naive']['bias']:+.1%}</td><td>{tot['test_naive_total']:,.0f} vs {tot['test_actual_total']:,} actual</td></tr></table>
<h2>Forecast accuracy by driver (top 5)</h2>
<table><tr><th>Driver</th><th>Ridge MAPE</th><th>Ridge WAPE</th><th>Naive MAPE</th><th>Naive WAPE</th></tr>{rows}</table>
<h2>Forecast accuracy by channel</h2>
<table><tr><th>Channel</th><th>Ridge MAPE</th><th>Ridge WAPE</th><th>Naive MAPE</th><th>Naive WAPE</th></tr>{ch_rows}</table>
<h2>Event overlays (measured uplift vs seasonal naive)</h2>
<table><tr><th>Event</th><th>Dates</th><th>Mean uplift</th><th>Actual contacts</th></tr>{ev_rows}</table>
<h2>Scenario overlays</h2>
<table><tr><th>Scenario</th><th>Dates</th><th>Uplift used</th><th>Overlay forecast range</th></tr>{sc_rows}</table>
<h2>Emerging drivers (share change, last 90 days vs prior 90)</h2>
<p><b>Read with care:</b> {warn}</p>
<table><tr><th>Driver</th><th>Prior share</th><th>Recent share</th><th>Change</th></tr>{em_rows}</table>
<h2>Largest forecast misses in the test window (variance / root cause)</h2>
<p>Under-forecasts cluster in Nov-Dec; Thanksgiving Day (Nov 28) was over-forecast by 50% because the model has no holiday feature. These are the misses an analyst would investigate first.</p>
<table><tr><th>Date</th><th>Actual</th><th>Forecast</th><th>Error</th></tr>{worst_rows}</table>
<h2>What contacts are about (TF-IDF keywords from real descriptor text, {metrics['insights']['keywords']['n_contacts_with_text']:,} contacts)</h2>
<table><tr><th>Driver</th><th>Top keywords</th></tr>{kw_rows}</table>
<p class="note"><b>Honest limits:</b> NYC 311 is a city service contact center, not a subscription
streaming service; drivers and volumes do not transfer to Netflix. No handle-time (AHT),
workload, staffing, or multi-vendor data exists in this dataset, so none is claimed.
Event uplifts are measured on a handful of event days and are noisy. The sample of
individual contacts covers 3 days (2024-01-08..10) only.</p>
</body></html>"""


def main():
    db_path = os.path.join(ROOT, "data", "contacts.db")
    if not os.path.exists(db_path):
        import build_db
        build_db.main()
    con = sqlite3.connect(db_path)
    totals = pd.read_sql("SELECT day, contacts FROM daily_totals", con, parse_dates=["day"])
    con.close()
    dataset = {
        "source": "NYC 311 Service Requests from 2010 to Present (data.cityofnewyork.us, dataset erm2-nwe9)",
        "source_url": "https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9",
        "license": "Public data published by the City of New York (NYC Open Data)",
        "n_days": int(len(totals)),
        "date_range": f"{str(totals['day'].min().date())}..{str(totals['day'].max().date())}",
        "total_contacts": int(totals["contacts"].sum()),
        "mean_contacts_per_day": round(float(totals["contacts"].mean()), 1),
        "raw_files": {
            "daily_totals.csv": "731 daily totals (Socrata aggregation)",
            "daily_by_driver.csv": "10,214 driver-day rows, top 15 drivers",
            "daily_by_channel.csv": "3,032 channel-day rows",
            "contacts_sample.csv": "30,470 individual contacts, 2024-01-08..10",
        },
    }
    fc_results = fc.run()
    ins_results = ins.run()
    # keep only train-period events for overlay learning; report all events
    train_events = [e for e in fc_results["events"] if e["end"] < "2024-10-01"]
    scenarios = scenario_overlays(totals, train_events)
    metrics = {
        "dataset": dataset,
        "forecast": fc_results,
        "insights": ins_results,
        "scenarios": scenarios,
        "train_window": "2023-01-01..2024-09-30",
        "test_window": "2024-10-01..2024-12-31 (92 days)",
    }
    with open(os.path.join(REPORTS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(REPORTS, "dashboard.html"), "w") as f:
        f.write(dashboard_html(metrics))
    print("wrote reports/metrics.json and reports/dashboard.html")
    print("total ridge test MAPE:", round(fc_results["total"]["calendar_ridge"]["mape"], 4),
          "WAPE:", round(fc_results["total"]["calendar_ridge"]["wape"], 4))
    print("total naive test MAPE:", round(fc_results["total"]["seasonal_naive"]["mape"], 4),
          "WAPE:", round(fc_results["total"]["seasonal_naive"]["wape"], 4))


if __name__ == "__main__":
    main()
