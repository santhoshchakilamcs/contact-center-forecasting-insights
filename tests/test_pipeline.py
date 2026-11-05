"""Tests for contact-center-forecasting-insights. Run: .venv/bin/python -m pytest tests/ -q"""
import math
import os
import sqlite3
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import forecast as fc
import insights as ins

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "contacts.db")


def test_db_tables_populated():
    con = sqlite3.connect(DB)
    counts = {
        t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("daily_totals", "daily_by_driver", "daily_by_channel", "contacts_sample")
    }
    con.close()
    assert counts["daily_totals"] == 731
    assert counts["daily_by_driver"] > 10000
    assert counts["daily_by_channel"] > 3000
    assert counts["contacts_sample"] > 30000


def test_daily_totals_cover_full_window_no_gaps():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT day FROM daily_totals ORDER BY day", con, parse_dates=["day"])
    con.close()
    assert (df["day"].diff().dropna() == pd.Timedelta(days=1)).all()
    assert str(df["day"].iloc[0].date()) == "2023-01-01"
    assert str(df["day"].iloc[-1].date()) == "2024-12-31"


def test_forecast_beats_seasonal_naive_on_wape():
    totals = fc.load_series("daily_totals")
    result, _ = fc.forecast_series(totals, "all contacts")
    assert result["calendar_ridge"]["wape"] < result["seasonal_naive"]["wape"]
    assert 0 < result["calendar_ridge"]["mape"] < 0.5
    assert result["test_actual_total"] > 0


def test_metrics_math():
    m = fc.metrics([100, 200, 300], [110, 190, 300])
    assert abs(m["mape"] - (0.10 + 0.05 + 0.0) / 3) < 1e-9
    assert abs(m["wape"] - 20 / 600) < 1e-9
    assert m["n_days"] == 3


def test_event_uplift_is_measured_not_invented():
    totals = fc.load_series("daily_totals")
    events = fc.event_uplifts(totals)
    names = [e["name"] for e in events]
    assert any("coastal storm" in n for n in names)
    storm = [e for e in events if "coastal storm" in e["name"]][0]
    assert storm["mean_uplift"] > 0  # the Jan 2024 storm really did lift contacts
    assert storm["days"] == 2


def test_keywords_come_from_real_text():
    kw = ins.keywords_by_driver()
    assert kw["n_contacts_with_text"] > 25000
    by_driver = kw["keywords_by_driver"]
    assert "Illegal Parking" in by_driver
    assert len(by_driver["Illegal Parking"]) == 8
    # every keyword must actually appear in some descriptor for that driver
    con = sqlite3.connect(DB)
    df = pd.read_sql(
        "SELECT descriptor FROM contacts_sample WHERE complaint_type='Illegal Parking' AND descriptor IS NOT NULL",
        con,
    )
    con.close()
    blob = " ".join(df["descriptor"].astype(str)).lower()
    for w in by_driver["Illegal Parking"]:
        assert w in blob


def test_emerging_drivers_shares_sum_sensibly():
    em = ins.emerging_drivers()
    assert em["rising"] and em["falling"]
    for row in em["rising"] + em["falling"]:
        assert 0 <= row["share_recent_pct"] <= 1
        assert 0 <= row["share_prior_pct"] <= 1
