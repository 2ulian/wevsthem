"""
Fetch Reddit posts from Arctic Shift API — sampled around curated events.
For each event in curated_events.csv, fetch posts from ±30 days window.
Output: data/datasets/imported/reddit_immigration.csv
"""
import time
import datetime
from pathlib import Path

import pandas as pd
import requests

API_BASE   = "https://arctic-shift.photon-reddit.com/api/posts/search"
OUT_PATH   = Path("data/datasets/imported/reddit_immigration.csv")
EVENTS_CSV = Path("data/events/curated_events.csv")

SUBREDDITS    = ["immigration", "europe", "worldnews", "UKpolitics", "politics"]
WINDOW_DAYS   = 30
POSTS_PER_WINDOW = 50   # per subreddit per event window


def iso(ts: int) -> str:
    return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")


def fetch_window(session, subreddit: str, after_dt: str, before_dt: str, n: int) -> list[dict]:
    rows = []
    cursor = after_dt
    while len(rows) < n:
        params = {
            "subreddit": subreddit,
            "after":     cursor,
            "before":    before_dt,
            "limit":     min(100, n - len(rows)),
            "sort":      "asc",
        }
        try:
            r = session.get(API_BASE, params=params, timeout=15)
            if r.status_code == 422:
                break
            r.raise_for_status()
            data = r.json().get("data", [])
        except Exception as e:
            print(f"    Error ({subreddit}): {e}")
            time.sleep(3)
            break

        if not data:
            break

        for post in data:
            title    = str(post.get("title", "") or "")
            selftext = str(post.get("selftext", "") or "")
            text     = (title + " " + selftext).strip()
            utc      = post.get("created_utc")
            if not text or not utc:
                continue
            rows.append({
                "text":        text[:800],
                "subreddit":   post.get("subreddit", subreddit),
                "created_utc": int(utc),
                "score":       post.get("score", 0),
                "source":      "reddit",
            })

        last_utc = data[-1].get("created_utc")
        if not last_utc:
            break
        cursor = iso(int(last_utc) + 1)
        time.sleep(0.2)

    return rows[:n]


events = pd.read_csv(EVENTS_CSV, parse_dates=["date"])
events = events[(events["date"].dt.year >= 2020) & (events["date"].dt.year <= 2023)]
print(f"Processing {len(events)} events (2020-2023)…")

session = requests.Session()
session.headers.update({"User-Agent": "WevsThem/1.0 research project"})

seen = set()
all_rows = []

for _, evt in events.iterrows():
    evt_date = evt["date"]
    after_dt  = (evt_date - pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%dT%H:%M:%S")
    before_dt = (evt_date + pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%dT%H:%M:%S")
    print(f"\n[{evt_date.date()}] {evt['title'][:60]}")

    for sub in SUBREDDITS:
        rows = fetch_window(session, sub, after_dt, before_dt, POSTS_PER_WINDOW)
        new = [r for r in rows if (r["text"][:60], r["created_utc"]) not in seen]
        for r in new:
            seen.add((r["text"][:60], r["created_utc"]))
        all_rows.extend(new)
        print(f"  r/{sub}: +{len(new)} posts (total {len(all_rows)})")

df = pd.DataFrame(all_rows)
df["post_date"] = pd.to_datetime(df["created_utc"], unit="s", utc=True).dt.tz_localize(None)
df = df.drop(columns=["created_utc"])
df = df.sort_values("post_date").reset_index(drop=True)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(df):,} posts → {OUT_PATH}")
print(f"Date range: {df['post_date'].min().date()} → {df['post_date'].max().date()}")
print(f"By subreddit:\n{df['subreddit'].value_counts().to_string()}")
