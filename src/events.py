"""
Event loader and event study utilities — "We vs Them" NLP project.

Two event sources:
  1. Curated CSV  — data/events/curated_events.csv
  2. ACLED API    — https://acleddata.com/api  (OAuth, email + password)

Public API
----------
load_curated(path)                        → pd.DataFrame of curated events
get_acled_token(email, password)          → Bearer token string
fetch_acled(email, password, **filters)   → pd.DataFrame of ACLED events
load_all_events(curated_path, email, password, **acled_filters) → combined df

event_study(df_posts, events, window_days, metric_cols)
    → pd.DataFrame with daily deviations from global mean per event
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ACLED_TOKEN_URL    = "https://acleddata.com/oauth/token"
ACLED_ENDPOINT     = "https://acleddata.com/api/acled/read"
ACLED_CACHE_PATH   = Path("data/events/acled_cache.json")

# ---------------------------------------------------------------------------
# Curated events
# ---------------------------------------------------------------------------

def load_curated(path: str | Path = "data/events/curated_events.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["source_type"] = "curated"
    return df.dropna(subset=["date"])


# ---------------------------------------------------------------------------
# ACLED
# ---------------------------------------------------------------------------

def get_acled_token(email: str, password: str) -> str:
    """
    Obtain a Bearer token from the ACLED OAuth endpoint.
    Token is valid for 24 hours.
    """
    resp = requests.post(
        ACLED_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "username":   email,
            "password":   password,
            "grant_type": "password",
            "client_id":  "acled",
            "scope":      "authenticated",
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def fetch_acled(
    email: str,
    password: str,
    date_range_start: str = "2020-01-01",
    date_range_end:   str | None = None,
    countries: list[str] | None = None,
    event_types: list[str] | None = None,
    limit: int = 5000,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch individual events from the ACLED API (OAuth).

    Parameters
    ----------
    email            : myACLED email
    password         : myACLED password
    date_range_start : ISO date "YYYY-MM-DD"
    date_range_end   : ISO date or None (= today)
    countries        : e.g. ["United Kingdom", "France"]
    event_types      : e.g. ["Riots", "Protests"] (None = all)
    limit            : max rows (ACLED cap = 10 000)
    use_cache        : load from local JSON cache when available

    Returns
    -------
    pd.DataFrame with columns: date, title, category, country, region, fatalities, source_type
    """
    cache_key = f"{date_range_start}_{date_range_end}_{countries}_{event_types}"
    if use_cache and ACLED_CACHE_PATH.exists():
        try:
            with open(ACLED_CACHE_PATH) as f:
                cache = json.load(f)
            if cache_key in cache:
                return pd.DataFrame(cache[cache_key])
        except Exception:
            pass

    token = get_acled_token(email, password)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    params: dict = {
        "_format":          "json",
        "event_date":       date_range_start,
        "event_date_where": "BETWEEN",
        "fields":           "event_date|event_type|sub_event_type|country|region|notes|fatalities",
        "limit":            limit,
    }
    if date_range_end:
        params["event_date2"] = date_range_end
    if countries:
        params["country"] = ":OR:country=".join(countries)
    if event_types:
        params["event_type"] = ":OR:event_type=".join(event_types)

    resp = requests.get(ACLED_ENDPOINT, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("data", [])

    rows = []
    for item in data:
        rows.append({
            "date":        pd.to_datetime(item.get("event_date"), errors="coerce"),
            "title":       (item.get("notes") or item.get("sub_event_type", ""))[:120],
            "category":    item.get("event_type", "").lower().replace(" ", "_").replace("/", "_"),
            "country":     item.get("country", ""),
            "region":      item.get("region", ""),
            "fatalities":  item.get("fatalities", 0),
            "source_type": "acled",
        })

    df = pd.DataFrame(rows).dropna(subset=["date"])

    if use_cache and not df.empty:
        ACLED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing = json.load(open(ACLED_CACHE_PATH)) if ACLED_CACHE_PATH.exists() else {}
        except Exception:
            existing = {}
        existing[cache_key] = df.assign(date=df["date"].astype(str)).to_dict("records")
        with open(ACLED_CACHE_PATH, "w") as f:
            json.dump(existing, f)

    return df


# ---------------------------------------------------------------------------
# Combined loader
# ---------------------------------------------------------------------------

def load_all_events(
    curated_path: str | Path = "data/events/curated_events.csv",
    email:    str | None = None,
    password: str | None = None,
    **acled_filters,
) -> pd.DataFrame:
    frames = [load_curated(curated_path)]
    if email and password:
        try:
            acled = fetch_acled(email, password, **acled_filters)
            frames.append(acled)
        except Exception as exc:
            print(f"[events] ACLED fetch failed: {exc}")
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    return combined.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Event study
# ---------------------------------------------------------------------------

def event_study(
    df_posts: pd.DataFrame,
    events: pd.DataFrame,
    window_days: int = 14,
    metric_cols: list[str] | None = None,
    date_col: str = "post_date",
) -> pd.DataFrame:
    """
    For each event, compute daily metric deviations from the global mean
    in a window of [-window_days, +window_days].

    Parameters
    ----------
    df_posts    : posts dataframe with `date_col` and metric columns
    events      : events dataframe with `date` column
    window_days : half-window size in days
    metric_cols : columns to analyse (default: othering_predicted, toxicity)
    date_col    : name of the date column in df_posts

    Returns
    -------
    pd.DataFrame with columns:
        day_offset, event_title, event_date, event_category,
        <metric>_mean, <metric>_deviation, n_posts
    """
    if metric_cols is None:
        metric_cols = [c for c in ["othering_predicted", "toxicity"] if c in df_posts.columns]

    df_posts = df_posts.copy()
    df_posts[date_col] = pd.to_datetime(df_posts[date_col], errors="coerce")
    df_posts = df_posts.dropna(subset=[date_col])
    df_posts["_day"] = df_posts[date_col].dt.normalize()

    # Global baseline (mean per metric over full dataset)
    global_means = {col: df_posts[col].mean() for col in metric_cols if col in df_posts.columns}
    if not global_means:
        return pd.DataFrame()

    daily = df_posts.groupby("_day").agg(
        n_posts=("_day", "count"),
        **{col: (col, "mean") for col in global_means}
    ).reset_index()

    rows = []
    for _, evt in events.iterrows():
        evt_date = pd.Timestamp(evt["date"]).normalize()
        window_start = evt_date - pd.Timedelta(days=window_days)
        window_end   = evt_date + pd.Timedelta(days=window_days)

        sub = daily[(daily["_day"] >= window_start) & (daily["_day"] <= window_end)].copy()
        if sub.empty:
            continue
        if sub["n_posts"].sum() < 5:
            continue

        sub["day_offset"] = (sub["_day"] - evt_date).dt.days
        sub["event_title"]    = evt["title"]
        sub["event_date"]     = evt_date
        sub["event_category"] = evt.get("category", "")
        sub["event_country"]  = evt.get("country", "")

        for col in global_means:
            sub[f"{col}_deviation"] = sub[col] - global_means[col]

        rows.append(sub)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)
