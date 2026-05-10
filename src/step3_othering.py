"""
Step 3 — Othering Detector for "We vs Them" NLP project.

Pipeline:
  - Task 1: Load data/processed/dataset_enriched.csv
  - Task 2: Apply detect_othering() to all posts
  - Task 3: Validation on 100 random examples (rule-based proxy accuracy)
  - Task 4: Save data/processed/dataset_othering.csv
  - Task 5: Summary — % posts with othering, top matched patterns

Usage:
  python src/step3_othering.py              # full dataset
  python src/step3_othering.py --max-rows 5000  # quick test on 5k rows
"""

import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from collections import Counter

sys.path.insert(0, "src")
from othering import detect_othering, apply_othering, ALL_FAMILIES

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--max-rows", type=int, default=None,
                    help="Limit number of rows to process (for quick tests)")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Task 1 — Load
# ---------------------------------------------------------------------------
print("=" * 60)
print("TASK 1 — Loading dataset_enriched.csv")
print("=" * 60)

df = pd.read_csv("data/processed/dataset_enriched.csv")
print(f"  Loaded {len(df):,} rows")

if args.max_rows:
    df = df.sample(n=min(args.max_rows, len(df)), random_state=42).reset_index(drop=True)
    print(f"  Subsampled to {len(df):,} rows (--max-rows {args.max_rows})")

# Ensure clean_text column exists
if "clean_text" not in df.columns:
    print("  Warning: 'clean_text' not found — using 'text' column instead")
    df["clean_text"] = df["text"].fillna("")


# ---------------------------------------------------------------------------
# Task 2 — Detect othering (full dataset)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 2 — Detecting othering patterns")
print("=" * 60)

df = apply_othering(df, text_col="clean_text")

n_othering = df["has_othering"].sum()
print(f"  Posts with othering : {n_othering:,} / {len(df):,} ({n_othering/len(df)*100:.1f}%)")
print(f"\n  Score distribution (0 = none, 4 = all four families):")
print(df["othering_score"].value_counts().sort_index().to_string())

# Top matched patterns (flatten the lists)
all_matched = [p for sublist in df["matched_patterns"] for p in sublist]
pattern_counts = Counter(all_matched)
print(f"\n  Top 15 matched patterns:")
for pattern, count in pattern_counts.most_common(15):
    print(f"    {pattern:<25} {count:>6,}")


# ---------------------------------------------------------------------------
# Task 3 — Validation on 100 random examples
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 3 — Validation on 100 random examples")
print("=" * 60)

sample_100 = df.sample(n=min(100, len(df)), random_state=7)

# Proxy ground truth: use toxicity score + pronoun type as heuristic
# A post is "likely othering" if toxicity > 0.5 AND has_them = True
# This is a rough proxy since we have no manual labels
has_proxy = (
    (sample_100.get("toxicity", pd.Series(0, index=sample_100.index)).fillna(0) > 0.5) &
    (sample_100.get("has_them", pd.Series(False, index=sample_100.index)).fillna(False))
)

# Precision proxy: among predicted othering, how many also satisfy the proxy?
predicted_othering = sample_100["has_othering"]
tp = (predicted_othering & has_proxy).sum()
fp = (predicted_othering & ~has_proxy).sum()
fn = (~predicted_othering & has_proxy).sum()
tn = (~predicted_othering & ~has_proxy).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"  (Proxy ground truth: toxicity > 0.5 AND has_them = True)")
print(f"  Precision : {precision:.2%}")
print(f"  Recall    : {recall:.2%}")
print(f"  F1        : {f1:.2%}")
print(f"\n  Confusion matrix (proxy):")
print(f"               Predicted+ Predicted-")
print(f"  Actual +      {tp:>5}      {fn:>5}")
print(f"  Actual -      {fp:>5}      {tn:>5}")

# Print 10 true positive examples
print(f"\n  --- 10 true-positive examples (othering detected + proxy agrees) ---")
tp_examples = sample_100[predicted_othering & has_proxy].head(10)
for _, row in tp_examples.iterrows():
    patterns = ", ".join(row["matched_patterns"][:3])
    preview  = str(row["clean_text"])[:120].replace("\n", " ")
    print(f"  [{patterns}]")
    print(f"    {preview}")

# Print 5 false positive examples (our detector flagged, proxy disagrees)
print(f"\n  --- 5 false-positive examples (othering detected, proxy disagrees) ---")
fp_examples = sample_100[predicted_othering & ~has_proxy].head(5)
for _, row in fp_examples.iterrows():
    patterns = ", ".join(row["matched_patterns"][:3])
    preview  = str(row["clean_text"])[:120].replace("\n", " ")
    print(f"  [{patterns}]")
    print(f"    {preview}")


# ---------------------------------------------------------------------------
# Task 4 — Save
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 4 — Saving dataset_othering.csv")
print("=" * 60)

out_path = "data/processed/dataset_othering.csv"
df.to_csv(out_path, index=False)
print(f"  Saved {len(df):,} rows to {out_path}")
print(f"  Columns: {list(df.columns)}")


# ---------------------------------------------------------------------------
# Task 5 — Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 5 — Summary")
print("=" * 60)

pct_othering = df["has_othering"].mean() * 100
print(f"  % posts with othering      : {pct_othering:.1f}%")

if "source" in df.columns:
    print(f"\n  Othering rate by source:")
    src_group = df.groupby("source")["has_othering"].mean() * 100
    for src, pct in src_group.items():
        print(f"    {src:<20} {pct:.1f}%")

if "subreddit" in df.columns:
    print(f"\n  Othering rate by subreddit (reddit only):")
    reddit_df = df[df["source"] == "reddit"] if "source" in df.columns else df
    if len(reddit_df) > 0:
        sub_group = reddit_df.groupby("subreddit")["has_othering"].mean() * 100
        for sub, pct in sub_group.sort_values(ascending=False).items():
            print(f"    {sub:<20} {pct:.1f}%")

print(f"\n  Top 10 matched patterns (full dataset):")
for pattern, count in pattern_counts.most_common(10):
    print(f"    {pattern:<25} {count:>6,}  ({count/len(df)*100:.2f}% of posts)")

if "emotion" in df.columns:
    print(f"\n  Top emotions in othering posts:")
    oth_emotions = df[df["has_othering"]]["emotion"].dropna().value_counts().head(5)
    print(oth_emotions.to_string())

    print(f"\n  Top emotions in non-othering posts:")
    non_emotions = df[~df["has_othering"]]["emotion"].dropna().value_counts().head(5)
    print(non_emotions.to_string())

if "toxicity" in df.columns:
    avg_tox_oth     = df[df["has_othering"]]["toxicity"].mean()
    avg_tox_non_oth = df[~df["has_othering"]]["toxicity"].mean()
    print(f"\n  Avg toxicity — othering posts    : {avg_tox_oth:.4f}")
    print(f"  Avg toxicity — non-othering posts: {avg_tox_non_oth:.4f}")

print("\nReady for Step 4 (ML Classifier).")
