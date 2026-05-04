"""
Steps 1 & 2 — Clean, tag pronouns, score toxicity (unitary/toxic-bert)
and emotion (GoEmotions) on a stratified sample.

Full pipeline:
  - Task 1: clean_text applied to ALL 134k rows
  - Task 2: pronoun tagging applied to ALL rows
  - Task 3: Detoxify (unitary/toxic-bert) on a stratified 10k sample
  - Task 4: GoEmotions on the same 10k sample
  - Task 5: save full dataset (NaN for unscored rows) + scored-only slice

Input : data/raw/dataset_clean.csv
Output: data/processed/dataset_enriched.csv
        data/processed/dataset_scored_sample.csv  (10k rows, all scores filled)
"""

import warnings
warnings.filterwarnings("ignore")

import re
import sys
import pandas as pd
import numpy as np
import torch
from transformers import pipeline

sys.path.insert(0, "src")
from cleaning import clean_text

SAMPLE_SIZE  = 10_000
DETOX_BATCH  = 64
EMOTION_BATCH = 32
MAX_CHARS    = 512

device = 0 if torch.cuda.is_available() else -1
print(f"Device: {'GPU' if device == 0 else 'CPU'}")

# Load data
print("\nLoading dataset…")
df = pd.read_csv("data/raw/dataset_clean.csv")
print(f"  {len(df):,} rows loaded")


# Task 1 — Clean texts (full dataset)
print("\n" + "=" * 60)
print("TASK 1 — Cleaning texts (full dataset)")
print("=" * 60)

df["clean_text"] = df["text"].apply(clean_text)

before = len(df)
df = df[df["clean_text"].str.len() >= 20].reset_index(drop=True)
print(f"  Rows after cleaning: {len(df):,}  (dropped {before - len(df):,})")


# Task 2 — Tag We / Them pronouns (full dataset)
print("\n" + "=" * 60)
print("TASK 2 — Tagging We/Them pronouns (full dataset)")
print("=" * 60)

WE_PATTERN = re.compile(
    r"\b(we|us|our|ours|ourselves)\b", re.IGNORECASE
)
THEM_PATTERN = re.compile(
    r"\b(they|them|their|theirs|those\s+people|these\s+people|people\s+like\s+them)\b",
    re.IGNORECASE,
)


def tag_pronouns(text: str) -> dict:
    we_matches   = WE_PATTERN.findall(text)
    them_matches = THEM_PATTERN.findall(text)
    has_we   = len(we_matches) > 0
    has_them = len(them_matches) > 0

    if has_we and has_them:
        ptype = "both"
    elif has_we:
        ptype = "we_only"
    elif has_them:
        ptype = "them_only"
    else:
        ptype = "none"

    return {
        "has_we":       has_we,
        "has_them":     has_them,
        "we_count":     len(we_matches),
        "them_count":   len(them_matches),
        "pronoun_type": ptype,
    }


pronoun_df = pd.DataFrame(df["clean_text"].apply(tag_pronouns).tolist())
df = pd.concat([df, pronoun_df], axis=1)

print(f"  Pronoun type distribution:")
print(df["pronoun_type"].value_counts().to_string())


# Build stratified 10k sample
print(f"\nBuilding stratified sample of {SAMPLE_SIZE:,} rows…")

# Stratify by source; within reddit also by subreddit
strat_col = df["subreddit"].fillna(df["source"])
df["_strat"] = strat_col

sample_idx = (
    df.groupby("_strat", group_keys=False)
    .apply(lambda g: g.sample(
        min(len(g), max(1, int(SAMPLE_SIZE * len(g) / len(df)))),
        random_state=42
    ))
    .index
)
# Top up to exactly SAMPLE_SIZE if rounding left us short
if len(sample_idx) < SAMPLE_SIZE:
    remaining = df.index.difference(sample_idx)
    extra = df.loc[remaining].sample(SAMPLE_SIZE - len(sample_idx), random_state=0).index
    sample_idx = sample_idx.union(extra)

sample_idx = sample_idx[:SAMPLE_SIZE]
df_sample = df.loc[sample_idx].copy()
print(f"  Sample size: {len(df_sample):,}")
print(f"  Sample breakdown by source/subreddit:")
print(df_sample["_strat"].value_counts().to_string())

df = df.drop(columns=["_strat"])
df_sample = df_sample.drop(columns=["_strat"])


# Task 3 — Toxicity scoring (sample only)
print("\n" + "=" * 60)
print(f"TASK 3 — Toxicity scoring (unitary/toxic-bert, {len(df_sample):,} rows)")
print("=" * 60)

DETOX_COLS = ["toxicity", "severe_toxicity", "identity_attack", "insult", "threat"]
LABEL_MAP  = {
    "toxic":         "toxicity",
    "severe_toxic":  "severe_toxicity",
    "identity_hate": "identity_attack",
    "insult":        "insult",
    "threat":        "threat",
}

detox_pipe = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    top_k=None,
    device=device,
    truncation=True,
    max_length=512,
)

def scores_from_result(result):
    lookup = {r["label"]: r["score"] for r in result}
    return {dst: lookup.get(src, 0.0) for src, dst in LABEL_MAP.items()}

# Smoke-test on 10 rows
print("  Smoke-test on 10 rows…")
smoke = detox_pipe(df_sample["clean_text"].iloc[:10].tolist(), batch_size=DETOX_BATCH)
avg  = np.mean([scores_from_result(r)["toxicity"] for r in smoke])
print(f"  OK — avg toxicity on smoke set: {avg:.4f}")

# Full sample run
texts_s = df_sample["clean_text"].tolist()
n_s     = len(texts_s)
all_scores = {c: [] for c in DETOX_COLS}

for start in range(0, n_s, DETOX_BATCH):
    batch   = texts_s[start : start + DETOX_BATCH]
    results = detox_pipe(batch, batch_size=DETOX_BATCH)
    for r in results:
        s = scores_from_result(r)
        for col in DETOX_COLS:
            all_scores[col].append(s[col])
    if start > 0 and start % 500 == 0:
        print(f"  Toxicity progress: {start:,}/{n_s:,}")

print(f"  Toxicity progress: {n_s:,}/{n_s:,} — done")

for col in DETOX_COLS:
    df_sample[col] = all_scores[col]

# Propagate scores back to full df (NaN for unscored rows)
for col in DETOX_COLS:
    df[col] = np.nan  # float64 — compatible with float scores
df.loc[df_sample.index, DETOX_COLS] = df_sample[DETOX_COLS].values


# Task 4 — GoEmotions (sample only)
print("\n" + "=" * 60)
print(f"TASK 4 — GoEmotions (monologg/bert-base-cased-goemotions-original, {len(df_sample):,} rows)")
print("=" * 60)

emotion_pipe = pipeline(
    "text-classification",
    model="monologg/bert-base-cased-goemotions-original",
    top_k=1,
    device=device,
    truncation=True,
    max_length=512,
)

# Smoke-test on 10 rows
print("  Smoke-test on 10 rows…")
smoke_e = emotion_pipe(
    [t[:MAX_CHARS] for t in df_sample["clean_text"].iloc[:10].tolist()],
    batch_size=EMOTION_BATCH,
)
print(f"  OK — first result: {smoke_e[0]}")

texts_e  = [t[:MAX_CHARS] for t in df_sample["clean_text"].tolist()]
emotions = []
e_scores = []

for start in range(0, n_s, EMOTION_BATCH):
    batch   = texts_e[start : start + EMOTION_BATCH]
    results = emotion_pipe(batch, batch_size=EMOTION_BATCH)
    for res in results:
        top = res[0] if isinstance(res, list) else res
        emotions.append(top["label"])
        e_scores.append(round(top["score"], 4))
    if start > 0 and start % 500 == 0:
        print(f"  GoEmotions progress: {start:,}/{n_s:,}")

print(f"  GoEmotions progress: {n_s:,}/{n_s:,} — done")

df_sample["emotion"]       = emotions
df_sample["emotion_score"] = e_scores

# Propagate back (object dtype for strings, float for scores)
df["emotion"]       = None       # object dtype → accepts strings
df["emotion_score"] = np.nan     # float64
df.loc[df_sample.index, "emotion"]       = df_sample["emotion"].values
df.loc[df_sample.index, "emotion_score"] = df_sample["emotion_score"].values


# Task 5 — Save
print("\n" + "=" * 60)
print("TASK 5 — Saving enriched datasets")
print("=" * 60)

df.to_csv("data/processed/dataset_enriched.csv", index=False)
print(f"  Full dataset  : data/processed/dataset_enriched.csv  ({len(df):,} rows)")

df_sample.to_csv("data/processed/dataset_scored_sample.csv", index=False)
print(f"  Scored sample : data/processed/dataset_scored_sample.csv  ({len(df_sample):,} rows)")


# Summary
pct_we   = df["has_we"].mean()   * 100
pct_them = df["has_them"].mean() * 100
pct_both = (df["pronoun_type"] == "both").mean() * 100

print("\n" + "=" * 60)
print("SUMMARY — Steps 1 & 2 complete")
print("=" * 60)
print(f"Total rows processed       : {len(df):,}")
print(f"Rows scored (sample)       : {len(df_sample):,}")
print(f"\nTop 5 emotions (sample):")
print(df_sample["emotion"].value_counts().head(5).to_string())
print(f"\nAverage toxicity (sample)  : {df_sample['toxicity'].mean():.4f}")
print(f"% posts with We markers    : {pct_we:.1f}%")
print(f"% posts with Them markers  : {pct_them:.1f}%")
print(f"% posts with both          : {pct_both:.1f}%")
print("\nReady for Step 3.")
