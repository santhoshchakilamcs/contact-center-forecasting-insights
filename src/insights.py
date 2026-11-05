#!/usr/bin/env python3
"""Contact-driver insights from the real NYC 311 data.

* Driver mix over time (monthly shares of the top drivers).
* Emerging drivers: share change between the last 90 days of the data
  and the 90 days before that.
* Channel mix shift over the same windows.
* Keyword analysis: pure-Python TF-IDF over the free-text `descriptor`
  field of the 30,470 real individual contacts (2024-01-08..10), grouped
  by driver. No text is invented; descriptors come from the dataset.
"""
import math
import os
import re
import sqlite3
from collections import Counter, defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "contacts.db")

STOPWORDS = set("""
a an the and or of to in on for with at by from is are was were be been it its this that
as i you he she we they my your his her our their me him us them have has had do does did
not no yes if but so then than too very can could will would should may might must shall
there here when where who whom which what how why all any some none one two three more most
other others such only own same get got make made take took see saw know known think
please per via re regarding about into over under out up down off again further once
am pm street avenue road lane drive court place boulevard
""".split())


def tokenize(text):
    if not isinstance(text, str):
        return []
    words = re.findall(r"[a-z][a-z\-]{2,}", text.lower())
    return [w for w in words if w not in STOPWORDS]


def driver_mix():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT day, complaint_type, contacts FROM daily_by_driver", con, parse_dates=["day"])
    con.close()
    df["month"] = df["day"].dt.to_period("M").astype(str)
    monthly = df.groupby(["month", "complaint_type"])["contacts"].sum().unstack(fill_value=0)
    shares = monthly.div(monthly.sum(axis=1), axis=0)
    top = df.groupby("complaint_type")["contacts"].sum().sort_values(ascending=False)
    return {
        "top_drivers_overall": {k: int(v) for k, v in top.head(10).items()},
        "top_driver_share_pct": {k: round(float(v), 4) for k, v in (top / top.sum()).head(10).items()},
        "monthly_top_driver": {
            m: shares.loc[m].sort_values(ascending=False).head(3).round(4).to_dict()
            for m in shares.index[::6]
        },
    }


def emerging_drivers():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT day, complaint_type, contacts FROM daily_by_driver", con, parse_dates=["day"])
    con.close()
    last = df["day"].max()
    recent = df[df["day"] > last - pd.Timedelta(days=90)]
    prior = df[(df["day"] <= last - pd.Timedelta(days=90)) & (df["day"] > last - pd.Timedelta(days=180))]
    r = recent.groupby("complaint_type")["contacts"].sum()
    p = prior.groupby("complaint_type")["contacts"].sum()
    share_r = r / r.sum()
    share_p = p / p.sum()
    change = (share_r - share_p).dropna().sort_values(ascending=False)
    return {
        "seasonality_warning": ("The recent window is Oct-Dec (heating season) and the prior window is Jul-Sep, "
            "so the HEAT/HOT WATER jump (+18.9pp) is mostly seasonality, not a new trend. "
            "Compare same-quarter last year before treating a share move as an emerging driver."),
        "window_recent": f"{str((last - pd.Timedelta(days=89)).date())}..{str(last.date())}",
        "window_prior": f"{str((last - pd.Timedelta(days=179)).date())}..{str((last - pd.Timedelta(days=90)).date())}",
        "rising": [
            {"driver": k, "share_recent_pct": round(float(share_r[k]), 4),
             "share_prior_pct": round(float(share_p[k]), 4),
             "change_pp": round(float(v * 100), 2)}
            for k, v in change.head(5).items()
        ],
        "falling": [
            {"driver": k, "share_recent_pct": round(float(share_r[k]), 4),
             "share_prior_pct": round(float(share_p[k]), 4),
             "change_pp": round(float(v * 100), 2)}
            for k, v in change.tail(5).items()
        ],
    }


def channel_shift():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT day, open_data_channel_type, contacts FROM daily_by_channel", con, parse_dates=["day"])
    con.close()
    last = df["day"].max()
    recent = df[df["day"] > last - pd.Timedelta(days=90)].groupby("open_data_channel_type")["contacts"].sum()
    prior = df[(df["day"] <= last - pd.Timedelta(days=90)) & (df["day"] > last - pd.Timedelta(days=180))].groupby("open_data_channel_type")["contacts"].sum()
    out = {}
    for ch in recent.index:
        if ch in ("OTHER",):
            continue
        out[str(ch)] = {
            "share_recent_pct": round(float(recent[ch] / recent.sum()), 4),
            "share_prior_pct": round(float(prior.get(ch, 0) / prior.sum()), 4),
        }
    return out


def keywords_by_driver(top_n=8, min_docs=50):
    """TF-IDF keywords per driver over real descriptor text."""
    con = sqlite3.connect(DB)
    df = pd.read_sql(
        "SELECT complaint_type, descriptor FROM contacts_sample WHERE descriptor IS NOT NULL",
        con,
    )
    con.close()
    docs_by_driver = defaultdict(list)
    for drv, desc in zip(df["complaint_type"], df["descriptor"]):
        toks = tokenize(desc)
        if toks:
            docs_by_driver[drv].append(toks)
    # document frequency across all docs
    n_docs = sum(len(v) for v in docs_by_driver.values())
    df_count = Counter()
    for docs in docs_by_driver.values():
        for toks in docs:
            for w in set(toks):
                df_count[w] += 1
    out = {}
    for drv, docs in docs_by_driver.items():
        if len(docs) < min_docs:
            continue
        tf = Counter()
        for toks in docs:
            tf.update(toks)
        total = sum(tf.values())
        scores = {}
        for w, c in tf.items():
            idf = math.log((n_docs + 1) / (df_count[w] + 1)) + 1.0
            scores[w] = (c / total) * idf
        out[drv] = [w for w, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]]
    return {"n_contacts_with_text": int(n_docs), "keywords_by_driver": out}


def run():
    return {
        "driver_mix": driver_mix(),
        "emerging_drivers": emerging_drivers(),
        "channel_shift": channel_shift(),
        "keywords": keywords_by_driver(),
    }


if __name__ == "__main__":
    import pprint
    pprint.pp(run())
