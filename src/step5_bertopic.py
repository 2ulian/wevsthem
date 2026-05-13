# Step 5 — BERTopic topic modeling on classified dataset → dataset_final.csv

import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--max-rows", type=int, default=None)
args = parser.parse_args()

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Task 1 — Load
# ---------------------------------------------------------------------------
print("=" * 60)
print("TASK 1 — Loading dataset_classified.csv")
print("=" * 60)

df = pd.read_csv("data/processed/dataset_classified.csv")
print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")

if args.max_rows:
    df = df.sample(n=min(args.max_rows, len(df)), random_state=42).reset_index(drop=True)
    print(f"  Subsampled to {len(df):,} rows (--max-rows {args.max_rows})")

text_col = "clean_text" if "clean_text" in df.columns else "text"
docs = df[text_col].fillna("").tolist()
print(f"  Text column: '{text_col}'")

cache_suffix = f"_{args.max_rows}" if args.max_rows else "_full"


# ---------------------------------------------------------------------------
# Task 2 — Fit BERTopic
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 2 — Fitting BERTopic model")
print("=" * 60)

min_topic_size = max(5, 20 if not args.max_rows else max(5, args.max_rows // 150))
print(f"  min_topic_size={min_topic_size}, nr_topics='auto'")

# --- Embeddings (cached) ---
emb_cache = CACHE_DIR / f"embeddings{cache_suffix}.npy"
if emb_cache.exists():
    print(f"  Loading cached embeddings from {emb_cache}")
    embeddings = np.load(str(emb_cache))
else:
    print("  Computing sentence embeddings (all-MiniLM-L6-v2)…")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = st_model.encode(docs, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    np.save(str(emb_cache), embeddings)
    print(f"  Embeddings saved to {emb_cache}")

# --- UMAP (cached) ---
umap_cache = CACHE_DIR / f"umap{cache_suffix}.npy"
if umap_cache.exists():
    print(f"  Loading cached UMAP from {umap_cache}")
    umap_embeddings = np.load(str(umap_cache))
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
else:
    print("  Fitting UMAP (this is the slow step)…")
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    umap_embeddings = umap_model.fit_transform(embeddings)
    np.save(str(umap_cache), umap_embeddings)
    print(f"  UMAP saved to {umap_cache}")

# --- HDBSCAN with n_jobs=1 (fix for Python 3.13 BrokenPipeError) ---
hdbscan_model = HDBSCAN(
    min_cluster_size=min_topic_size,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
    core_dist_n_jobs=1,
)

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    nr_topics="auto",
    verbose=True,
    calculate_probabilities=False,
)

topics, probs = topic_model.fit_transform(docs, embeddings=umap_embeddings)

n_outliers_before = sum(t == -1 for t in topics)
print(f"\n  Outliers before reduce_outliers: {n_outliers_before:,} ({n_outliers_before/len(docs)*100:.1f}%)")
print("  Running reduce_outliers (strategy=c-tf-idf)…")
topics = topic_model.reduce_outliers(docs, topics, strategy="c-tf-idf")
topic_model.update_topics(docs, topics=topics)

df["topic"] = topics

topic_info = topic_model.get_topic_info()
n_topics = len(topic_info[topic_info["Topic"] != -1])
n_outliers = sum(t == -1 for t in topics)
print(f"  Topics found       : {n_topics}")
print(f"  Outlier docs (-1)  : {n_outliers:,} ({n_outliers/len(docs)*100:.1f}%)")
print("\n  Top 10 topics by size:")
print(topic_info[topic_info["Topic"] != -1].head(10)[["Topic", "Count", "Name"]].to_string(index=False))


# ---------------------------------------------------------------------------
# Task 3 — Topic naming
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 3 — Topic naming")
print("=" * 60)

# Auto-name from top 3 words; fill in TOPIC_NAMES below for manual overrides
TOPIC_NAMES: dict[int, str] = {
    # -1: "outlier",   # uncomment and add IDs after reviewing topics
}

def make_topic_name(topic_id: int) -> str:
    if topic_id == -1:
        return "outlier"
    if topic_id in TOPIC_NAMES:
        return TOPIC_NAMES[topic_id]
    words = topic_model.get_topic(topic_id)
    if words:
        return "_".join(w for w, _ in words[:3])
    return f"topic_{topic_id}"

df["topic_name"] = df["topic"].apply(make_topic_name)

print("  Topic name mapping (first 15):")
for _, row in topic_info[topic_info["Topic"] != -1].head(15).iterrows():
    name = make_topic_name(int(row["Topic"]))
    print(f"    topic {int(row['Topic']):>3}  ({int(row['Count']):>6} docs)  → {name}")


# ---------------------------------------------------------------------------
# Task 4 — Save dataset_final.csv
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 4 — Saving dataset_final.csv")
print("=" * 60)

out_path = "data/processed/dataset_final.csv"
df.to_csv(out_path, index=False)
print(f"  Saved {len(df):,} rows, {df.shape[1]} columns → {out_path}")
print(f"  New columns added: topic, topic_name")


# ---------------------------------------------------------------------------
# Task 5 — Cross-analysis by topic
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 5 — Cross-analysis (toxicity / emotion / othering by topic)")
print("=" * 60)

df_topics = df[df["topic"] != -1].copy()

agg = df_topics.groupby("topic_name").agg(
    count=("topic", "count"),
    toxicity_mean=("toxicity", "mean"),
    othering_pct=("othering_predicted", "mean"),
    othering_score_mean=("othering_score", "mean"),
).reset_index().sort_values("count", ascending=False)

agg["toxicity_mean"]     = agg["toxicity_mean"].round(4)
agg["othering_pct"]      = (agg["othering_pct"] * 100).round(2)
agg["othering_score_mean"] = agg["othering_score_mean"].round(4)

print(agg.to_string(index=False))

# Dominant emotion per topic
if "emotion" in df.columns:
    top_emotion = (
        df_topics.groupby(["topic_name", "emotion"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .drop_duplicates("topic_name")
        .rename(columns={"emotion": "top_emotion", "n": "emotion_count"})
    )
    agg = agg.merge(top_emotion[["topic_name", "top_emotion"]], on="topic_name", how="left")
    print("\n  Dominant emotion added to cross-analysis.")

cross_path = "reports/platform_comparison.csv"
agg.to_csv(cross_path, index=False)
print(f"\n  Cross-analysis saved → {cross_path}")


# ---------------------------------------------------------------------------
# Task 6 — Figures
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 6 — Generating figures")
print("=" * 60)

# Figure 1: topic sizes (bar chart, top 20)
top20 = agg.head(20)
fig1 = px.bar(
    top20.sort_values("count"),
    x="count", y="topic_name",
    orientation="h",
    title="Topic sizes (top 20)",
    labels={"count": "Number of documents", "topic_name": "Topic"},
)
fig1.update_layout(height=600)
fig1.write_html(str(FIGURES_DIR / "topic_sizes.html"))
print(f"  Saved → reports/figures/topic_sizes.html")

# Figure 2: mean toxicity per topic (top 20)
fig2 = px.bar(
    top20.sort_values("toxicity_mean"),
    x="toxicity_mean", y="topic_name",
    orientation="h",
    title="Mean toxicity by topic (top 20 topics by size)",
    labels={"toxicity_mean": "Mean toxicity score", "topic_name": "Topic"},
    color="toxicity_mean",
    color_continuous_scale="Reds",
)
fig2.update_layout(height=600)
fig2.write_html(str(FIGURES_DIR / "topic_toxicity.html"))
print(f"  Saved → reports/figures/topic_toxicity.html")

# Figure 3: othering % per topic (top 20)
fig3 = px.bar(
    top20.sort_values("othering_pct"),
    x="othering_pct", y="topic_name",
    orientation="h",
    title="Othering % by topic (top 20 topics by size)",
    labels={"othering_pct": "% othering posts", "topic_name": "Topic"},
    color="othering_pct",
    color_continuous_scale="Oranges",
)
fig3.update_layout(height=600)
fig3.write_html(str(FIGURES_DIR / "topic_othering.html"))
print(f"  Saved → reports/figures/topic_othering.html")

print("\nStep 5 complete. Ready for Step 6 (temporal analysis).")
