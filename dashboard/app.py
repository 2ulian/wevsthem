import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(layout="wide", page_title="We vs Them — Dashboard", page_icon="🔍")

BASE_DIR = Path(__file__).parent.parent
FINAL_CSV = BASE_DIR / "data" / "processed" / "dataset_final.csv"
CLASSIFIED_CSV = BASE_DIR / "data" / "processed" / "dataset_classified.csv"


@st.cache_data
def load_data():
    if FINAL_CSV.exists():
        df = pd.read_csv(FINAL_CSV, low_memory=False)
        source_file = "dataset_final.csv"
    else:
        df = pd.read_csv(CLASSIFIED_CSV, low_memory=False)
        source_file = "dataset_classified.csv"
    df["othering_predicted"] = df["othering_predicted"].fillna(0).astype(int)
    df["toxicity"] = pd.to_numeric(df["toxicity"], errors="coerce")
    df["othering_proba"] = pd.to_numeric(df["othering_proba"], errors="coerce")
    df["othering_score"] = pd.to_numeric(df["othering_score"], errors="coerce")
    df["subreddit"] = df["subreddit"].fillna("unknown")
    df["source"] = df["source"].fillna("unknown")
    df["pronoun_type"] = df["pronoun_type"].fillna("unknown")
    return df, source_file


df_full, loaded_file = load_data()
has_topics = "topic" in df_full.columns and "topic_name" in df_full.columns

st.sidebar.title("We vs Them")
st.sidebar.caption(f"Loaded file: **{loaded_file}**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Sentiment & Toxicity", "Emotions", "Othering & Topics"],
)

st.sidebar.markdown("---")
keyword = st.sidebar.text_input("Keyword search", placeholder="e.g. immigrant, they...")

known_subreddits = sorted(s for s in df_full["subreddit"].unique() if s != "unknown")
selected_subreddits = st.sidebar.multiselect(
    "Filter by subreddit",
    options=known_subreddits,
    default=[],
    placeholder="All subreddits",
)

df = df_full.copy()
if keyword.strip():
    mask = df["text"].fillna("").str.contains(keyword.strip(), case=False, na=False)
    df = df[mask]
if selected_subreddits:
    df = df[df["subreddit"].isin(selected_subreddits)]

if len(df) == 0:
    st.warning("No results for these filters.")
    st.stop()


# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Overview")

    col1, col2, col3 = st.columns(3)
    total_posts = len(df)
    pct_othering = df["othering_predicted"].mean() * 100
    pct_high_tox = (df["toxicity"].dropna() > 0.5).mean() * 100

    col1.metric("Total posts", f"{total_posts:,}")
    col2.metric("% Othering detected", f"{pct_othering:.1f}%")
    col3.metric("Highly toxic posts (>0.5)", f"{pct_high_tox:.1f}%")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Distribution by source")
        source_counts = df["source"].value_counts().reset_index()
        source_counts.columns = ["source", "count"]
        source_counts["pct"] = (source_counts["count"] / source_counts["count"].sum() * 100).round(1)
        fig_src = px.bar(
            source_counts,
            x="source",
            y="count",
            text=source_counts.apply(lambda r: f"{r['count']:,} ({r['pct']}%)", axis=1),
            labels={"source": "Source", "count": "Number of posts"},
        )
        fig_src.update_traces(textposition="outside")
        fig_src.update_layout(showlegend=False)
        st.plotly_chart(fig_src, use_container_width=True)

    with col_right:
        reddit_df = df[df["subreddit"] != "unknown"]
        n_subs = reddit_df["subreddit"].nunique()
        st.subheader(f"Top subreddits ({n_subs} total)")
        top_sub = reddit_df["subreddit"].value_counts().head(10).reset_index()
        top_sub.columns = ["subreddit", "count"]
        if top_sub.empty:
            st.info("No Reddit posts in the current selection.")
        else:
            fig_sub = px.bar(
                top_sub,
                x="count",
                y="subreddit",
                orientation="h",
                text="count",
            )
            fig_sub.update_layout(yaxis={"categoryorder": "total ascending"})
            fig_sub.update_traces(textposition="outside")
            st.plotly_chart(fig_sub, use_container_width=True)
        st.caption(f"{df[df['subreddit'] == 'unknown'].shape[0]:,} hate_speech posts without subreddit excluded.")

    st.subheader("Distribution of pronoun types")
    pronoun_counts = (
        df[df["pronoun_type"] != "unknown"]["pronoun_type"]
        .value_counts()
        .reset_index()
    )
    pronoun_counts.columns = ["pronoun_type", "count"]
    n_unknown_pron = (df["pronoun_type"] == "unknown").sum()
    fig_pron = px.bar(pronoun_counts, x="pronoun_type", y="count", text="count")
    fig_pron.update_traces(textposition="outside")
    st.plotly_chart(fig_pron, use_container_width=True)
    if n_unknown_pron > 0:
        st.caption(f"{n_unknown_pron:,} posts with unknown pronoun_type excluded.")


# ---------------------------------------------------------------------------
elif page == "Sentiment & Toxicity":
    st.title("Sentiment & Toxicity")

    st.subheader("Toxicity distribution")
    fig_hist = px.histogram(
        df.dropna(subset=["toxicity"]),
        x="toxicity",
        nbins=60,
        labels={"toxicity": "Toxicity score", "count": "Number of posts"},
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Toxicity by othering detection")
    box_src = df.dropna(subset=["toxicity", "othering_predicted"]).copy()
    box_src["Othering detected"] = box_src["othering_predicted"].map({0: "No (0)", 1: "Yes (1)"})
    fig_box = px.box(
        box_src,
        x="Othering detected",
        y="toxicity",
        color="Othering detected",
        color_discrete_map={"No (0)": "#636EFA", "Yes (1)": "#EF553B"},
        labels={"toxicity": "Toxicity score"},
        points=False,
    )
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Average toxicity by source")
    tox_source = (
        df.dropna(subset=["toxicity"])
        .groupby("source")["toxicity"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"mean": "Mean", "median": "Median", "count": "Nb posts"})
    )
    tox_source["Mean"] = tox_source["Mean"].round(4)
    tox_source["Median"] = tox_source["Median"].round(4)
    st.dataframe(tox_source, use_container_width=True, hide_index=True)

    st.subheader("Top 10 most toxic posts")
    top_toxic = (
        df.dropna(subset=["toxicity"])
        .nlargest(10, "toxicity")[["text", "source", "subreddit", "toxicity", "othering_predicted"]]
        .reset_index(drop=True)
    )
    top_toxic.index += 1
    st.dataframe(top_toxic, use_container_width=True)


# ---------------------------------------------------------------------------
elif page == "Emotions":
    st.title("Emotions")

    df_emo = df.dropna(subset=["emotion"])
    n_total = len(df)
    n_with_emo = len(df_emo)
    pct_emo = n_with_emo / n_total * 100

    if pct_emo < 99:
        st.warning(
            f"GoEmotions covers **{n_with_emo:,} posts** ({pct_emo:.1f}%). "
            f"{n_total - n_with_emo:,} posts without emotion are excluded from the charts."
        )

    st.subheader("Global emotion distribution")
    emotion_counts = df_emo["emotion"].value_counts().reset_index()
    emotion_counts.columns = ["emotion", "count"]
    fig_emo_bar = px.bar(
        emotion_counts,
        x="emotion",
        y="count",
        text="count",
        labels={"emotion": "Emotion", "count": "Number of posts"},
    )
    fig_emo_bar.update_layout(xaxis={"categoryorder": "total descending"})
    fig_emo_bar.update_traces(textposition="outside")
    st.plotly_chart(fig_emo_bar, use_container_width=True)

    st.subheader("Emotions by subreddit (heatmap)")
    df_emo_reddit = df_emo[df_emo["subreddit"] != "unknown"]
    if df_emo_reddit.empty:
        st.info("No Reddit posts with emotion in the current selection.")
    else:
        top_subs_emo = df_emo_reddit["subreddit"].value_counts().head(15).index.tolist()
        heat_df = df_emo_reddit[df_emo_reddit["subreddit"].isin(top_subs_emo)].copy()
        pivot = heat_df.groupby(["subreddit", "emotion"]).size().reset_index(name="n")
        pivot["pct"] = pivot.groupby("subreddit")["n"].transform(lambda x: x / x.sum() * 100)
        heatmap_data = pivot.pivot(index="subreddit", columns="emotion", values="pct").fillna(0)
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns.tolist(),
                y=heatmap_data.index.tolist(),
                colorscale="Blues",
                text=np.round(heatmap_data.values, 1),
                texttemplate="%{text}%",
                hovertemplate="Subreddit: %{y}<br>Emotion: %{x}<br>%{z:.1f}%<extra></extra>",
            )
        )
        fig_heat.update_layout(height=500)
        st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Emotions: othering vs non-othering")
    df_emo_split = df_emo.copy()
    df_emo_split["group"] = df_emo_split["othering_predicted"].map({1: "Othering", 0: "Non-othering"})
    emo_split = (
        df_emo_split.groupby(["group", "emotion"])
        .size()
        .reset_index(name="n")
    )
    emo_split["pct"] = emo_split.groupby("group")["n"].transform(lambda x: x / x.sum() * 100).round(2)
    fig_grouped = px.bar(
        emo_split,
        x="emotion",
        y="pct",
        color="group",
        barmode="group",
        labels={"pct": "% of posts", "emotion": "Emotion", "group": ""},
    )
    fig_grouped.update_layout(xaxis={"categoryorder": "total descending"})
    st.plotly_chart(fig_grouped, use_container_width=True)


# ---------------------------------------------------------------------------
elif page == "Othering & Topics":
    st.title("Othering & Topics")

    st.subheader("% othering by subreddit")
    df_sub_oth = df[df["subreddit"] != "unknown"]
    sub_othering = (
        df_sub_oth.groupby("subreddit")["othering_predicted"]
        .agg(["sum", "count"])
        .reset_index()
    )
    sub_othering.columns = ["subreddit", "othering_posts", "total_posts"]
    sub_othering["pct_othering"] = sub_othering["othering_posts"] / sub_othering["total_posts"] * 100
    sub_othering = sub_othering[sub_othering["total_posts"] >= 20].sort_values("pct_othering", ascending=True)
    if sub_othering.empty:
        st.info("Not enough Reddit posts for this calculation.")
    else:
        fig_oth_sub = px.bar(
            sub_othering,
            x="pct_othering",
            y="subreddit",
            orientation="h",
            text=sub_othering["pct_othering"].round(1).astype(str) + "%",
            labels={"pct_othering": "% othering", "subreddit": "Subreddit"},
        )
        fig_oth_sub.update_traces(textposition="outside")
        st.plotly_chart(fig_oth_sub, use_container_width=True)

    st.subheader("Post examples")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        score_min = float(df["othering_score"].min()) if not df["othering_score"].isna().all() else 0.0
        score_max = float(df["othering_score"].max()) if not df["othering_score"].isna().all() else 3.0
        score_range = st.slider(
            "othering_score",
            min_value=score_min,
            max_value=score_max,
            value=(score_min, score_max),
            step=1.0,
        )
    with col_f2:
        sub_filter = st.multiselect(
            "Subreddit",
            options=known_subreddits,
            default=[],
            placeholder="All",
        )
    with col_f3:
        othering_only = st.checkbox("Detected othering only", value=False)

    examples_df = df.dropna(subset=["othering_score"])
    examples_df = examples_df[
        (examples_df["othering_score"] >= score_range[0])
        & (examples_df["othering_score"] <= score_range[1])
    ]
    if sub_filter:
        examples_df = examples_df[examples_df["subreddit"].isin(sub_filter)]
    if othering_only:
        examples_df = examples_df[examples_df["othering_predicted"] == 1]

    st.caption(f"{len(examples_df):,} posts match the current filters.")
    display_cols = ["text", "source", "subreddit", "othering_score", "othering_proba", "matched_patterns", "toxicity"]
    st.dataframe(
        examples_df[display_cols].head(200).reset_index(drop=True),
        use_container_width=True,
    )

    st.markdown("---")

    if has_topics and "topic" in df.columns and "topic_name" in df.columns:
        st.subheader("BERTopic topics — Top 20 by size")

        topic_df = df.dropna(subset=["topic", "topic_name"]).copy()
        topic_df["topic"] = topic_df["topic"].astype(int)

        topic_stats = (
            topic_df[topic_df["topic"] >= 0]
            .groupby(["topic", "topic_name"])
            .agg(
                nb_posts=("topic", "count"),
                mean_toxicity=("toxicity", "mean"),
                pct_othering=("othering_predicted", "mean"),
            )
            .reset_index()
        )
        topic_stats["pct_othering"] = (topic_stats["pct_othering"] * 100).round(1)
        topic_stats["mean_toxicity"] = topic_stats["mean_toxicity"].round(3)
        topic_stats["label"] = topic_stats["topic"].astype(str) + " — " + topic_stats["topic_name"]

        top20 = topic_stats.nlargest(20, "nb_posts")
        fig_topics_bar = px.bar(
            top20.sort_values("nb_posts", ascending=True),
            x="nb_posts",
            y="label",
            orientation="h",
            text="nb_posts",
            labels={"nb_posts": "Number of posts", "label": "Topic"},
        )
        fig_topics_bar.update_traces(textposition="outside")
        fig_topics_bar.update_layout(height=600)
        st.plotly_chart(fig_topics_bar, use_container_width=True)

        st.subheader("Topics: toxicity vs % othering (top 50)")
        top50 = topic_stats.nlargest(50, "nb_posts")
        fig_topics_scatter = px.scatter(
            top50,
            x="mean_toxicity",
            y="pct_othering",
            size="nb_posts",
            hover_name="label",
            hover_data={"nb_posts": True, "mean_toxicity": True, "pct_othering": True, "label": False},
            labels={
                "mean_toxicity": "Mean toxicity",
                "pct_othering": "% othering",
                "nb_posts": "Nb posts",
            },
            size_max=60,
        )
        fig_topics_scatter.update_layout(height=600)
        st.plotly_chart(fig_topics_scatter, use_container_width=True)

    else:
        st.info("BERTopic topics are being generated — relaunch the dashboard once dataset_final.csv is available.")
