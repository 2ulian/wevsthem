# Enrichit les posts sans émotion avec GoEmotions → met à jour tous les fichiers processed

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from transformers import pipeline

EMOTION_BATCH = 16
FILES = [
    "data/processed/dataset_final.csv",
    "data/processed/dataset_classified.csv",
    "data/processed/dataset_othering.csv",
    "data/processed/dataset_enriched.csv",
]

# ---------------------------------------------------------------------------
# Charger dataset_final et identifier les lignes sans émotion
# ---------------------------------------------------------------------------
print("Chargement dataset_final.csv…")
df = pd.read_csv("data/processed/dataset_final.csv", low_memory=False)
print(f"  {len(df):,} lignes total")

missing_mask = df["emotion"].isna()
n_missing = missing_mask.sum()
print(f"  {n_missing:,} lignes sans émotion ({n_missing/len(df)*100:.1f}%)")

if n_missing == 0:
    print("Rien à faire — toutes les émotions sont déjà renseignées.")
    raise SystemExit(0)

text_col = "clean_text" if "clean_text" in df.columns else "text"
texts_to_score = df.loc[missing_mask, text_col].fillna("").tolist()

# ---------------------------------------------------------------------------
# Charger le modèle GoEmotions
# ---------------------------------------------------------------------------
print("\nChargement GoEmotions (monologg/bert-base-cased-goemotions-original)…")
emotion_pipe = pipeline(
    "text-classification",
    model="monologg/bert-base-cased-goemotions-original",
    top_k=1,
    truncation=True,
    max_length=128,
)

# Smoke test
_ = emotion_pipe(["test"])
print("  Modèle chargé.")

# ---------------------------------------------------------------------------
# Inférence par batches
# ---------------------------------------------------------------------------
print(f"\nInférence sur {n_missing:,} posts (batch={EMOTION_BATCH})…")
emotions = []
e_scores = []

for start in range(0, len(texts_to_score), EMOTION_BATCH):
    batch = texts_to_score[start:start + EMOTION_BATCH]
    try:
        results = emotion_pipe(batch, batch_size=EMOTION_BATCH)
        for res in results:
            top = res[0] if isinstance(res, list) else res
            emotions.append(top["label"])
            e_scores.append(round(top["score"], 4))
    except Exception:
        emotions.extend(["neutral"] * len(batch))
        e_scores.extend([0.0] * len(batch))

    if (start // EMOTION_BATCH) % 100 == 0:
        done = min(start + EMOTION_BATCH, n_missing)
        print(f"  {done:,}/{n_missing:,} ({done/n_missing*100:.1f}%)")

print(f"  {n_missing:,}/{n_missing:,} — done")

# ---------------------------------------------------------------------------
# Mettre à jour dataset_final en mémoire
# ---------------------------------------------------------------------------
missing_indices = df.index[missing_mask]
df.loc[missing_indices, "emotion"]       = emotions
df.loc[missing_indices, "emotion_score"] = e_scores

print(f"\nTop 5 émotions (nouveaux posts) :")
print(pd.Series(emotions).value_counts().head(5).to_string())

# ---------------------------------------------------------------------------
# Sauvegarder tous les fichiers processed
# ---------------------------------------------------------------------------
emotion_update = df[["emotion", "emotion_score"]].copy()

for fpath in FILES:
    try:
        target = pd.read_csv(fpath, low_memory=False)
        if len(target) != len(df):
            print(f"  SKIP {fpath} — taille différente ({len(target)} vs {len(df)})")
            continue
        target["emotion"]       = emotion_update["emotion"].values
        target["emotion_score"] = emotion_update["emotion_score"].values
        target.to_csv(fpath, index=False)
        print(f"  Mis à jour → {fpath}")
    except FileNotFoundError:
        print(f"  SKIP {fpath} — fichier introuvable")

print("\nGoEmotions full run terminé.")
