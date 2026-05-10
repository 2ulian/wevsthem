"""
Step 4 — ML Classifier for othering detection ("We vs Them" NLP project).

Pipeline:
  - Task 1: Load data/processed/dataset_othering.csv
  - Task 2: Build TF-IDF features → train LR + SVM → evaluate
  - Task 3: Build sentence-transformer embeddings → train LR + SVM → evaluate
  - Task 4: Compare all models, pick best
  - Task 5: Save best model to src/othering_classifier.pkl
  - Task 6: Print full classification report + error analysis

Usage:
  python src/step4_classifier.py                   # full dataset
  python src/step4_classifier.py --max-rows 5000   # quick test
  python src/step4_classifier.py --skip-embeddings # TF-IDF only (faster)
"""

import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, "src")
from classifier import (
    build_tfidf_features,
    build_embedding_features,
    train_and_evaluate,
    save_model,
)

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--max-rows",        type=int,  default=None)
parser.add_argument("--skip-embeddings", action="store_true",
                    help="Skip sentence-transformer step (faster)")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Task 1 — Load
# ---------------------------------------------------------------------------
print("=" * 60)
print("TASK 1 — Loading dataset_othering.csv")
print("=" * 60)

df = pd.read_csv("data/processed/dataset_othering.csv")
print(f"  Loaded {len(df):,} rows")

if args.max_rows:
    df = df.sample(n=min(args.max_rows, len(df)), random_state=42).reset_index(drop=True)
    print(f"  Subsampled to {len(df):,} rows (--max-rows {args.max_rows})")

# Use clean_text; fall back to text
text_col = "clean_text" if "clean_text" in df.columns else "text"
texts  = df[text_col].fillna("").tolist()
labels = df["has_othering"].astype(int).tolist()

pos = sum(labels)
print(f"  Label distribution: {pos:,} othering ({pos/len(labels)*100:.1f}%) / "
      f"{len(labels)-pos:,} non-othering ({(len(labels)-pos)/len(labels)*100:.1f}%)")

# 80/20 stratified split
X_text_train, X_text_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"  Train: {len(y_train):,}  Test: {len(y_test):,}")

y_train_arr = np.array(y_train, dtype=bool)
y_test_arr  = np.array(y_test,  dtype=bool)

all_results = []
best_overall = {"f1": -1.0, "model": None, "name": None, "vectorizer": None}


# ---------------------------------------------------------------------------
# Task 2 — TF-IDF features
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 2 — TF-IDF features (LR + SVM)")
print("=" * 60)

X_tfidf_all, vectorizer = build_tfidf_features(texts)
X_tfidf_train = X_tfidf_all[:len(X_text_train)]
X_tfidf_test  = X_tfidf_all[len(X_text_train):]

# Re-split to keep alignment
from sklearn.model_selection import train_test_split as tts
import scipy.sparse as sp

X_tfidf_full, _ = build_tfidf_features(
    X_text_train + X_text_test,
    max_features=50_000,
)
# Fit on train only to avoid leakage
X_tfidf_tr, vect = build_tfidf_features(X_text_train)
X_tfidf_te       = vect.transform(X_text_test)

tfidf_res = train_and_evaluate(
    X_tfidf_tr, X_tfidf_te,
    y_train_arr, y_test_arr,
    feature_name="TF-IDF",
    test_texts=X_text_test,
)
all_results.extend(tfidf_res["results"])

for r in tfidf_res["results"]:
    if r["f1"] > best_overall["f1"]:
        best_overall.update({
            "f1": r["f1"], "model": r["model"],
            "name": r["model_name"] + "_TFIDF", "vectorizer": vect,
        })


# ---------------------------------------------------------------------------
# Task 3 — Sentence-transformer embeddings
# ---------------------------------------------------------------------------
if not args.skip_embeddings:
    print("\n" + "=" * 60)
    print("TASK 3 — Sentence-transformer embeddings (all-MiniLM-L6-v2)")
    print("=" * 60)

    print("  Encoding train set…")
    X_emb_train = build_embedding_features(X_text_train, show_progress=True)
    print("  Encoding test set…")
    X_emb_test  = build_embedding_features(X_text_test,  show_progress=True)
    print(f"  Embedding shape: {X_emb_train.shape}")

    emb_res = train_and_evaluate(
        X_emb_train, X_emb_test,
        y_train_arr, y_test_arr,
        feature_name="Embeddings",
        test_texts=X_text_test,
    )
    all_results.extend(emb_res["results"])

    for r in emb_res["results"]:
        if r["f1"] > best_overall["f1"]:
            best_overall.update({
                "f1": r["f1"], "model": r["model"],
                "name": r["model_name"] + "_Embeddings", "vectorizer": None,
            })
else:
    print("\nTask 3 skipped (--skip-embeddings).")


# ---------------------------------------------------------------------------
# Task 4 — Compare all models
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 4 — Model comparison")
print("=" * 60)

summary = pd.DataFrame([
    {
        "model":     r["model_name"],
        "features":  r["feature"],
        "precision": round(r["precision"], 4),
        "recall":    round(r["recall"],    4),
        "f1":        round(r["f1"],        4),
    }
    for r in all_results
]).sort_values("f1", ascending=False)

print(summary.to_string(index=False))
print(f"\n  Best model: {best_overall['name']}  (F1 = {best_overall['f1']:.4f})")


# ---------------------------------------------------------------------------
# Task 5 — Save best model
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 5 — Saving best model")
print("=" * 60)

payload = {
    "model":      best_overall["model"],
    "vectorizer": best_overall["vectorizer"],   # None for embeddings model
    "model_name": best_overall["name"],
}
save_model(payload, "src/othering_classifier.pkl")


# ---------------------------------------------------------------------------
# Task 6 — Full report on best model
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 6 — Full classification report (best model)")
print("=" * 60)

best_entry = next(r for r in all_results if r["f1"] == best_overall["f1"])
print(best_entry["report"])
print(f"  Confusion matrix:")
print(best_entry["cm"])

print("\nReady for Step 5 (BERTopic).")
