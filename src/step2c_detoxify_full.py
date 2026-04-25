# Enrichit les posts sans toxicité avec Detoxify → met à jour tous les fichiers processed

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from detoxify import Detoxify

BATCH_SIZE = 32
FILES = [
    "data/processed/dataset_final.csv",
    "data/processed/dataset_classified.csv",
    "data/processed/dataset_othering.csv",
    "data/processed/dataset_enriched.csv",
]
TOX_COLS = ["toxicity", "severe_toxicity", "identity_attack", "insult", "threat"]

print("Chargement dataset_final.csv…")
df = pd.read_csv("data/processed/dataset_final.csv", low_memory=False)
print(f"  {len(df):,} lignes total")

missing_mask = df["toxicity"].isna()
n_missing = missing_mask.sum()
print(f"  {n_missing:,} lignes sans toxicité ({n_missing/len(df)*100:.1f}%)")

if n_missing == 0:
    print("Rien à faire.")
    raise SystemExit(0)

text_col = "clean_text" if "clean_text" in df.columns else "text"
texts_to_score = df.loc[missing_mask, text_col].fillna("").tolist()

print("\nChargement Detoxify (original)…")
model = Detoxify("original")
print("  Modèle chargé.")

print(f"\nInférence sur {n_missing:,} posts (batch={BATCH_SIZE})…")
results = {col: [] for col in TOX_COLS}

for start in range(0, len(texts_to_score), BATCH_SIZE):
    batch = texts_to_score[start:start + BATCH_SIZE]
    try:
        preds = model.predict(batch)
        for col in TOX_COLS:
            results[col].extend([round(float(v), 4) for v in preds[col]])
    except Exception:
        for col in TOX_COLS:
            results[col].extend([None] * len(batch))

    if (start // BATCH_SIZE) % 200 == 0:
        done = min(start + BATCH_SIZE, n_missing)
        print(f"  {done:,}/{n_missing:,} ({done/n_missing*100:.1f}%)", flush=True)

print(f"  {n_missing:,}/{n_missing:,} — done", flush=True)

missing_indices = df.index[missing_mask]
for col in TOX_COLS:
    df.loc[missing_indices, col] = results[col]

print(f"\nToxicité moyenne (nouveaux posts): {pd.Series(results['toxicity']).mean():.4f}")

tox_update = df[TOX_COLS].copy()

for fpath in FILES:
    try:
        target = pd.read_csv(fpath, low_memory=False)
        if len(target) != len(df):
            print(f"  SKIP {fpath} — taille différente")
            continue
        for col in TOX_COLS:
            target[col] = tox_update[col].values
        target.to_csv(fpath, index=False)
        print(f"  Mis à jour → {fpath}", flush=True)
    except FileNotFoundError:
        print(f"  SKIP {fpath} — introuvable")

print("\nDetoxify full run terminé.", flush=True)
