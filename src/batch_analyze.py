"""
Batch analysis script — runs the full pipeline on all raw datasets.

For each CSV in data/Datasets/:
  1. Detect text column
  2. Clean text + pronoun tagging + othering detection
  3. Detoxify  (toxicity, severe_toxicity, identity_attack, insult, threat)
  4. GoEmotions (emotion, emotion_score)
  5. Save results in-place

Usage:
    python src/batch_analyze.py [--skip-toxicity] [--skip-emotions] [--dry-run]
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from cleaning import clean_text
from othering import apply_othering

TEXT_ALIASES = [
    "text", "Text", "TEXT", "caption", "Caption",
    "tweet", "Tweet", "tweet_text", "comment", "Comment", "comment_text",
    "body", "Body", "content", "Content", "message", "Message", "description",
    "transcription", "video_transcription_text",
    "parentText", "childCommentText", "post", "post_text", "selftext", "title",
]

WE_WORDS     = ["we", "us", "our", "ours", "ourselves"]
THEM_WORDS   = ["they", "them", "their", "theirs"]
THEM_PHRASES = ["those people", "these people", "people like them"]


def detect_text_column(df):
    cols = df.columns.tolist()
    for alias in TEXT_ALIASES:
        if alias in cols:
            return alias
    lower_map = {c.lower(): c for c in cols}
    for alias in TEXT_ALIASES:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    str_cols = df.select_dtypes(include="object").columns.tolist()
    avg_lens = {c: df[c].dropna().astype(str).str.len().mean() for c in str_cols}
    long_cols = sorted([c for c, l in avg_lens.items() if l and l > 30],
                       key=lambda c: -avg_lens[c])
    return long_cols[0] if long_cols else None


def tag_pronouns(text):
    if not isinstance(text, str):
        return {"has_we": False, "has_them": False, "we_count": 0,
                "them_count": 0, "pronoun_type": "none"}
    t = text.lower()
    wc = sum(len(re.findall(r"\b" + w + r"\b", t)) for w in WE_WORDS)
    tc = sum(len(re.findall(r"\b" + w + r"\b", t)) for w in THEM_WORDS)
    tc += sum(t.count(p) for p in THEM_PHRASES)
    has_we, has_them = wc > 0, tc > 0
    ptype = ("both" if has_we and has_them
             else "we_only" if has_we
             else "them_only" if has_them
             else "none")
    return {"has_we": has_we, "has_them": has_them,
            "we_count": wc, "them_count": tc, "pronoun_type": ptype}


def run_base(df):
    """Cleaning + pronouns + othering (fast, no GPU)."""
    if "clean_text" not in df.columns:
        print("    cleaning text...")
        df["clean_text"] = df["text"].fillna("").apply(clean_text)

    if "pronoun_type" not in df.columns:
        print("    tagging pronouns...")
        pron = pd.DataFrame(df["clean_text"].apply(tag_pronouns).tolist(),
                            index=df.index)
        df = pd.concat([df, pron], axis=1)

    if "has_othering" not in df.columns:
        print("    detecting othering...")
        df = apply_othering(df, text_col="clean_text")

    if "othering_predicted" not in df.columns:
        df["othering_predicted"] = df["has_othering"].astype(int)
    if "othering_proba" not in df.columns:
        df["othering_proba"] = df["othering_score"] / 4.0

    return df


def run_toxicity(df):
    """Detoxify — fills toxicity + sub-scores."""
    if df.get("toxicity", pd.Series(dtype=float)).notna().any():
        print("    toxicity already present, skipping.")
        return df

    print("    running Detoxify...")
    from detoxify import Detoxify
    model  = Detoxify("original")
    texts  = df["clean_text"].fillna("").tolist()
    tox_cols = ["toxicity", "severe_toxicity", "identity_attack", "insult", "threat"]
    results  = {c: [] for c in tox_cols}

    batch = 64
    for i in range(0, len(texts), batch):
        res = model.predict(texts[i:i + batch])
        for c in tox_cols:
            results[c].extend(res[c])
        pct = min(i + batch, len(texts)) / len(texts) * 100
        print(f"\r    Detoxify {pct:.0f}%", end="", flush=True)
    print()

    for c in tox_cols:
        df[c] = results[c]
    return df


def run_emotions(df):
    """GoEmotions — fills emotion + emotion_score."""
    if "emotion" in df.columns and df["emotion"].notna().any():
        print("    emotions already present, skipping.")
        return df

    print("    running GoEmotions...")
    from transformers import pipeline as hf_pipeline
    pipe  = hf_pipeline(
        "text-classification",
        model="monologg/bert-base-cased-goemotions-original",
        top_k=1,
    )
    texts   = df["clean_text"].fillna("").str[:512].tolist()
    emotions, scores = [], []

    batch = 16
    for i in range(0, len(texts), batch):
        res = pipe(texts[i:i + batch])
        emotions.extend(r[0]["label"] for r in res)
        scores.extend(r[0]["score"]   for r in res)
        pct = min(i + batch, len(texts)) / len(texts) * 100
        print(f"\r    GoEmotions {pct:.0f}%", end="", flush=True)
    print()

    df["emotion"]       = emotions
    df["emotion_score"] = scores
    return df


def process_file(path: Path, skip_tox: bool, skip_emo: bool, dry_run: bool):
    print(f"\n{'='*60}")
    print(f"  {path.relative_to(BASE_DIR)}")

    df = pd.read_csv(path, low_memory=False)
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"  {len(df):,} rows, {len(df.columns)} cols")

    col = detect_text_column(df)
    if col is None:
        print("  !! no text column found, skipping.")
        return
    if col != "text":
        print(f"  renaming '{col}' → 'text'")
        df = df.rename(columns={col: "text"})

    df = run_base(df)
    if not skip_tox:
        df = run_toxicity(df)
    if not skip_emo:
        df = run_emotions(df)

    if dry_run:
        print("  [dry-run] not saving.")
        return

    df.to_csv(path, index=False)
    print(f"  saved → {path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-toxicity", action="store_true")
    parser.add_argument("--skip-emotions", action="store_true")
    parser.add_argument("--dry-run",       action="store_true")
    args = parser.parse_args()

    datasets_dir = BASE_DIR / "data" / "Datasets"
    files = sorted(datasets_dir.rglob("*.csv"))
    print(f"Found {len(files)} CSV files.")

    for path in files:
        try:
            process_file(path, args.skip_toxicity, args.skip_emotions, args.dry_run)
        except Exception as e:
            print(f"  !! ERROR on {path.name}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
