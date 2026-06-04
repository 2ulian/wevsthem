"""
Patch dataset_final.csv and dataset_classified.csv with post_date
from the Pushshift Reddit source (fddemarco/pushshift-reddit).

Matches by cleaned text prefix (first 80 chars) against the 1 289
Reddit rows in the processed datasets.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import datetime
import pandas as pd
from datasets import load_dataset

FINAL_CSV      = Path("data/processed/dataset_final.csv")
CLASSIFIED_CSV = Path("data/processed/dataset_classified.csv")
TARGET_SUBS    = {"politics", "immigration", "europe", "worldnews", "conspiracy"}
KEEP_COLS      = ["title", "selftext", "subreddit", "created_utc"]
STREAM_LIMIT   = 50_000


def build_text(row):
    t = (str(row.get("title", "") or "") + " " + str(row.get("selftext", "") or "")).strip()
    return t[:80].lower()


print("Loading Pushshift Reddit (streaming 50 000 rows)…")
reddit_raw = load_dataset("fddemarco/pushshift-reddit", split="train", streaming=True)

rows = []
for i, row in enumerate(reddit_raw):
    if i >= STREAM_LIMIT:
        break
    if row.get("subreddit") not in TARGET_SUBS:
        continue
    rows.append({c: row.get(c, "") for c in KEEP_COLS})

push_df = pd.DataFrame(rows)
push_df["_key"] = (
    (push_df["title"].fillna("") + " " + push_df["selftext"].fillna(""))
    .str.strip().str[:80].str.lower()
)
push_df["post_date"] = pd.to_datetime(
    pd.to_numeric(push_df["created_utc"], errors="coerce"), unit="s", utc=True
).dt.tz_localize(None)

key_to_date = push_df.dropna(subset=["post_date"]).set_index("_key")["post_date"].to_dict()
print(f"  {len(push_df):,} Reddit rows fetched, {len(key_to_date):,} with valid dates")


def patch(csv_path: Path):
    df = pd.read_csv(csv_path, low_memory=False)
    if "post_date" in df.columns:
        print(f"{csv_path.name}: post_date already exists, overwriting Reddit rows only")
    else:
        df["post_date"] = pd.NaT

    mask = df["source"] == "reddit"
    df.loc[mask, "_key"] = df.loc[mask, "text"].str[:80].str.lower()
    df.loc[mask, "post_date"] = df.loc[mask, "_key"].map(key_to_date)
    df = df.drop(columns=["_key"], errors="ignore")

    matched = df.loc[mask, "post_date"].notna().sum()
    total   = mask.sum()
    print(f"{csv_path.name}: matched {matched}/{total} Reddit rows with dates")
    df.to_csv(csv_path, index=False)
    print(f"  Saved → {csv_path}")


patch(FINAL_CSV)
if CLASSIFIED_CSV.exists():
    patch(CLASSIFIED_CSV)

print("\nDone.")
