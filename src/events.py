"""
Event study utilities — "We vs Them" NLP project.

Public API
----------
event_study(df_posts, events, window_days, metric_cols)
    → pd.DataFrame with daily deviations from global mean per event
"""

import pandas as pd


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

    metric_cols = [c for c in metric_cols if c in df_posts.columns]
    if not metric_cols:
        return pd.DataFrame()

    daily = df_posts.groupby("_day").agg(
        n_posts=("_day", "count"),
        **{col: (col, "mean") for col in metric_cols}
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

        sub["day_offset"]     = (sub["_day"] - evt_date).dt.days
        sub["event_title"]    = evt["title"]
        sub["event_date"]     = evt_date
        sub["event_category"] = evt.get("category", "")
        sub["event_country"]  = evt.get("country", "")

        # Pre-event window as baseline (standard event-study methodology)
        pre = sub[sub["day_offset"] < 0]
        for col in metric_cols:
            baseline = pre[col].mean() if not pre.empty and pre[col].notna().any() else sub[col].mean()
            sub[f"{col}_deviation"] = sub[col] - baseline

        rows.append(sub)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)
