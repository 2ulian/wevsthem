"""
Step 0 — Collect and explore raw data for "We vs Them" NLP project.
"""

import random
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datasets import load_dataset

# Task 1 HuggingFace hate speech dataset

print("=" * 60)
print("TASK 1 — Loading ucberkeley-dlab/measuring-hate-speech")
print("=" * 60)

hs_dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")
split = "train" if "train" in hs_dataset else list(hs_dataset.keys())[0]
hs_df = hs_dataset[split].to_pandas()

hs_df.to_csv("data/raw/hate_speech.csv", index=False)
print(f"Saved {len(hs_df):,} rows to data/raw/hate_speech.csv")


# Task 2 Pushshift Reddit dataset

print("\n" + "=" * 60)
print("TASK 2 — Loading fddemarco/pushshift-reddit (first 50 000 rows)")
print("=" * 60)

TARGET_SUBREDDITS = ["politics", "immigration", "europe", "worldnews", "conspiracy"]
KEEP_COLS = ["title", "selftext", "subreddit", "created_utc", "score"]

reddit_raw = load_dataset(
    "fddemarco/pushshift-reddit",
    split="train",
    streaming=True,
)

rows = []
for i, row in enumerate(reddit_raw):
    if i >= 50_000:
        break
    rows.append({c: row.get(c, "") for c in KEEP_COLS})

reddit_df = pd.DataFrame(rows)

# Filter to target subreddits
reddit_df = reddit_df[reddit_df["subreddit"].isin(TARGET_SUBREDDITS)].copy()

# Merge title + selftext into a single "text" column
reddit_df["title"] = reddit_df["title"].fillna("").astype(str)
reddit_df["selftext"] = reddit_df["selftext"].fillna("").astype(str)
reddit_df["text"] = (reddit_df["title"] + " " + reddit_df["selftext"]).str.strip()

# Drop original title / selftext columns
reddit_df = reddit_df.drop(columns=["title", "selftext"])

reddit_df.to_csv("data/raw/reddit_filtered.csv", index=False)
print(f"Saved {len(reddit_df):,} rows to data/raw/reddit_filtered.csv")


# Task 3 Explore both datasets

def explore(df, name, text_col="text", subreddit_col=None):
    print(f"\n{'─'*60}")
    print(f"  Dataset: {name}")
    print(f"{'─'*60}")
    print(f"Rows × Columns : {df.shape[0]:,} × {df.shape[1]}")
    print(f"\nColumn types:")
    print(df.dtypes.to_string())

    if subreddit_col and subreddit_col in df.columns:
        print(f"\nValue counts by {subreddit_col}:")
        print(df[subreddit_col].value_counts().to_string())

    lengths = df[text_col].fillna("").astype(str).str.len()
    print(f"\nAverage text length : {lengths.mean():.1f} chars")

    short = (lengths < 20).sum()
    print(f"Posts < 20 chars    : {short:,}")

    print(f"\n5 random text examples:")
    samples = df[text_col].dropna().astype(str)
    samples = samples[samples.str.len() > 0]
    for i, txt in enumerate(samples.sample(min(5, len(samples)), random_state=42), 1):
        preview = txt[:200].replace("\n", " ")
        print(f"  [{i}] {preview}")


# For hate speech, look for a usable text column
hs_text_col = None
for candidate in ["text", "comment", "post"]:
    if candidate in hs_df.columns:
        hs_text_col = candidate
        break
if hs_text_col is None:
    # Fall back to first string column
    hs_text_col = hs_df.select_dtypes(include="object").columns[0]

print("\n" + "=" * 60)
print("TASK 3 — Exploration")
print("=" * 60)

explore(hs_df, "HuggingFace — measuring-hate-speech", text_col=hs_text_col)
explore(reddit_df, "Pushshift Reddit (filtered)", text_col="text", subreddit_col="subreddit")


# Task 4 Clean and save final dataset

print("\n" + "=" * 60)
print("TASK 4 — Cleaning & merging")
print("=" * 60)

#Hate speech
hs_clean = hs_df.copy()
hs_clean["text"] = hs_clean[hs_text_col].fillna("").astype(str)
hs_clean = hs_clean[hs_clean["text"].str.len() >= 20].copy()
hs_clean["source"] = "hate_speech"
hs_clean["subreddit"] = np.nan

#reddit
reddit_clean = reddit_df.copy()
reddit_clean = reddit_clean[reddit_clean["text"].str.len() >= 20].copy()
reddit_clean["source"] = "reddit"

# keep only common useful columns
shared_cols = ["text", "source", "subreddit"]
final_df = pd.concat(
    [hs_clean[shared_cols], reddit_clean[shared_cols]],
    ignore_index=True,
)

# Final null drop
final_df = final_df.dropna(subset=["text"])
final_df = final_df[final_df["text"].str.strip().str.len() >= 20]

final_df.to_csv("data/raw/dataset_clean.csv", index=False)
print(f"Saved {len(final_df):,} rows to data/raw/dataset_clean.csv")


# Summary

print("\n" + "=" * 60)
print("SUMMARY — Ready for Step 1")
print("=" * 60)
print(f"Total posts : {len(final_df):,}")
print(f"\nBreakdown by source:")
print(final_df["source"].value_counts().to_string())
print(f"\nBreakdown by subreddit (reddit posts only):")
reddit_part = final_df[final_df["source"] == "reddit"]
print(reddit_part["subreddit"].value_counts().to_string())
print("\ndata/raw/dataset_clean.csv is ready for Step 1.")
