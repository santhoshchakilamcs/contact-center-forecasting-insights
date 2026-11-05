#!/usr/bin/env python3
"""Demand forecasting for NYC 311 daily contacts.

Models (both fitted on data before the test window only - no lookahead):
  * seasonal naive: forecast = contacts 7 days earlier (same weekday).
  * calendar ridge: ridge regression (numpy lstsq, small penalty) on
    day-of-week one-hots, annual sin/cos, a linear trend, and lags
    (lag 7, lag 14, lag 28, and the 7-day mean ending 8 days ago).

Test window: 2024-10-01 .. 2024-12-31 (92 days). Train: everything before.
Metrics: MAPE, WAPE, mean bias (forecast - actual) / actual.
Also fits the same models per top driver and per channel, and measures
the uplift of named real events (data/events.json) against the seasonal
naive baseline, so an analyst overlay can be applied to future scenarios.
"""
import json
import os
import sqlite3

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "contacts.db")
EVENTS = os.path.join(ROOT, "data", "events.json")

TEST_START = pd.Timestamp("2024-10-01")
TEST_END = pd.Timestamp("2024-12-31")
TOP_DRIVERS = 5
RIDGE_LAMBDA = 10.0


def load_series(table, value_col="contacts", group_col=None, group_value=None):
    con = sqlite3.connect(DB)
    q = f"SELECT day, {value_col} FROM {table}"
    if group_col is not None:
        q += f" WHERE {group_col} = ?"
        df = pd.read_sql(q, con, params=(group_value,), parse_dates=["day"])
    else:
        df = pd.read_sql(q, con, parse_dates=["day"])
    con.close()
    return df.sort_values("day").reset_index(drop=True)


def _design(df):
    """Calendar + lag features for a daily series with a 'contacts' column."""
    d = df["day"]
    dow = d.dt.dayofweek.values
    doy = d.dt.day_of_year.values
    n = len(df)
    X = np.zeros((n, 13))
    X[:, 0] = 1.0  # intercept
    for k in range(6):
        X[:, 1 + k] = (dow == k + 1).astype(float)  # Mon=0 is the reference
    X[:, 7] = np.sin(2 * np.pi * doy / 365.25)
    X[:, 8] = np.cos(2 * np.pi * doy / 365.25)
    X[:, 9] = np.arange(n) / 365.25  # trend in years
    y = df["contacts"].values.astype(float)
    lag7 = np.full(n, np.nan)
    lag14 = np.full(n, np.nan)
    lag28 = np.full(n, np.nan)
    roll = np.full(n, np.nan)
    lag7[7:] = y[:-7]
    lag14[14:] = y[:-14]
    lag28[28:] = y[:-28]
    for i in range(14, n):
        roll[i] = y[i - 14:i - 7].mean()
    X[:, 10] = lag7
    X[:, 11] = lag14
    X[:, 12] = roll
    return X, y, lag28


def fit_ridge(X, y):
    ok = ~np.isnan(X).any(axis=1)
    Xtr, ytr = X[ok], y[ok]
    # standardise non-intercept columns, then ridge (no penalty on intercept)
    mu = Xtr[:, 1:].mean(axis=0)
    sd = Xtr[:, 1:].std(axis=0)
    sd[sd == 0] = 1.0
    Z = (Xtr[:, 1:] - mu) / sd
    Z = np.column_stack([np.ones(len(Z)), Z])
    pen = np.eye(Z.shape[1]) * RIDGE_LAMBDA
    pen[0, 0] = 0.0
    beta = np.linalg.solve(Z.T @ Z + pen, Z.T @ ytr)
    return {"beta": beta, "mu": mu, "sd": sd}


def predict_ridge(model, X):
    ok = ~np.isnan(X).any(axis=1)
    out = np.full(len(X), np.nan)
    Z = (X[ok][:, 1:] - model["mu"]) / model["sd"]
    Z = np.column_stack([np.ones(len(Z)), Z])
    out[ok] = Z @ model["beta"]
    return out


def metrics(actual, forecast):
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    ok = ~np.isnan(f) & (a > 0)
    a, f = a[ok], f[ok]
    err = f - a
    return {
        "n_days": int(len(a)),
        "mape": float(np.mean(np.abs(err) / a)),
        "wape": float(np.sum(np.abs(err)) / np.sum(a)),
        "bias": float(np.mean(err) / np.mean(a)),
        "mae": float(np.mean(np.abs(err))),
    }


def forecast_series(df, label):
    """Fit on train, forecast the test window. Returns dict of metrics."""
    train = df[df["day"] < TEST_START].copy()
    test = df[(df["day"] >= TEST_START) & (df["day"] <= TEST_END)].copy()
    # seasonal naive needs the 7 days before the test window: use full series
    full = df.set_index("day")["contacts"]
    naive = np.array([full.get(d - pd.Timedelta(days=7), np.nan) for d in test["day"]])
    Xtr, ytr, _ = _design(train)
    model = fit_ridge(Xtr, ytr)
    Xte, yte, _ = _design(test)
    # lag features inside the test window must come from the full history,
    # so rebuild design on the concatenated series and take the test rows.
    Xfull, yfull, _ = _design(df)
    mask = (df["day"] >= TEST_START) & (df["day"] <= TEST_END)
    ridge = predict_ridge(model, Xfull[mask.values])
    actual = test["contacts"].values.astype(float)
    out = {
        "label": label,
        "seasonal_naive": metrics(actual, naive),
        "calendar_ridge": metrics(actual, ridge),
        "test_actual_total": int(actual.sum()),
        "test_naive_total": float(np.nansum(naive)),
        "test_ridge_total": float(np.nansum(ridge)),
    }
    return out, test.assign(naive=naive, ridge=ridge)


def event_uplifts(df):
    """Measured uplift of each named real event vs the seasonal naive baseline.
    Uplift = mean(actual / naive - 1) over the event days (train period only).
    """
    with open(EVENTS) as f:
        events = json.load(f)["events"]
    full = df.set_index("day")["contacts"]
    out = []
    for ev in events:
        start, end = pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"])
        days = pd.date_range(start, end)
        rows = []
        for d in days:
            a = full.get(d, np.nan)
            b = full.get(d - pd.Timedelta(days=7), np.nan)
            if pd.notna(a) and pd.notna(b) and b > 0:
                rows.append(a / b - 1)
        if rows:
            out.append({
                "name": ev["name"],
                "start": ev["start"],
                "end": ev["end"],
                "days": len(rows),
                "mean_uplift": float(np.mean(rows)),
                "actual_total": int(sum(full.get(d, 0) for d in days)),
            })
    return out


def run():
    totals = load_series("daily_totals")
    result, test_df = forecast_series(totals, "all contacts")
    results = {"total": result}
    # per-driver models (top 5 drivers by volume)
    drivers = pd.read_sql(
        "SELECT complaint_type, SUM(contacts) AS n FROM daily_by_driver GROUP BY 1 ORDER BY 2 DESC LIMIT ?",
        sqlite3.connect(DB), params=(TOP_DRIVERS,),
    )
    per_driver = []
    for drv in drivers["complaint_type"]:
        df = load_series("daily_by_driver", group_col="complaint_type", group_value=drv)
        r, _ = forecast_series(df, drv)
        per_driver.append(r)
    results["per_driver"] = per_driver
    # per-channel models
    channels = pd.read_sql(
        "SELECT open_data_channel_type, SUM(contacts) AS n FROM daily_by_channel GROUP BY 1 ORDER BY 2 DESC",
        sqlite3.connect(DB),
    )
    per_channel = []
    for ch in channels["open_data_channel_type"]:
        if ch in (None, "OTHER"):
            continue
        df = load_series("daily_by_channel", group_col="open_data_channel_type", group_value=ch)
        r, _ = forecast_series(df, ch)
        per_channel.append(r)
    results["per_channel"] = per_channel
    results["events"] = event_uplifts(totals)
    # variance / root-cause breakdown on the test window: error by weekday
    test_df = test_df.copy()
    test_df["dow"] = test_df["day"].dt.day_name()
    test_df["ridge_err"] = test_df["ridge"] - test_df["contacts"]
    by_dow = (
        test_df.groupby("dow")["ridge_err"].agg(["mean", "count"])
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    )
    results["test_error_by_weekday"] = {
        str(k): {"mean_error": float(v["mean"]), "n_days": int(v["count"])}
        for k, v in by_dow.iterrows()
    }
    # worst test days (largest absolute ridge error)
    test_df["abs_err"] = test_df["ridge_err"].abs()
    worst = test_df.sort_values("abs_err", ascending=False).head(5)
    results["worst_test_days"] = [
        {"date": str(r.day.date()), "actual": int(r.contacts), "forecast": round(float(r.ridge), 1),
         "error_pct": round(float((r.ridge - r.contacts) / r.contacts), 3)}
        for r in worst.itertuples()
    ]
    return results


if __name__ == "__main__":
    import pprint
    pprint.pp(run())
