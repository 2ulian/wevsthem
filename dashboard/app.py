import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import re

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from cleaning import clean_text
from othering import apply_othering

st.set_page_config(layout="wide", page_title="We vs Them — Dashboard", page_icon="🔍")

FINAL_CSV      = BASE_DIR / "data" / "processed" / "dataset_final.csv"
CLASSIFIED_CSV = BASE_DIR / "data" / "processed" / "dataset_classified.csv"
DATASETS_DIR   = BASE_DIR / "data" / "Datasets"

WE_WORDS     = ["we", "us", "our", "ours", "ourselves"]
THEM_WORDS   = ["they", "them", "their", "theirs"]
THEM_PHRASES = ["those people", "these people", "people like them"]

TEXT_ALIASES = [
    "text", "Text", "TEXT",
    "caption", "Caption",
    "tweet", "Tweet", "tweet_text",
    "comment", "Comment", "comment_text",
    "body", "Body", "content", "Content",
    "message", "Message", "description",
    "transcription", "video_transcription_text",
    "parentText", "childCommentText",
    "post", "post_text", "selftext", "title",
]


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def tag_pronouns(text: str) -> dict:
    if not isinstance(text, str):
        return {"has_we": False, "has_them": False, "we_count": 0, "them_count": 0, "pronoun_type": "none"}
    t = text.lower()
    wc = sum(len(re.findall(r"\b" + w + r"\b", t)) for w in WE_WORDS)
    tc = sum(len(re.findall(r"\b" + w + r"\b", t)) for w in THEM_WORDS)
    tc += sum(t.count(p) for p in THEM_PHRASES)
    has_we, has_them = wc > 0, tc > 0
    ptype = "both" if has_we and has_them else "we_only" if has_we else "them_only" if has_them else "none"
    return {"has_we": has_we, "has_them": has_them, "we_count": wc, "them_count": tc, "pronoun_type": ptype}


def detect_text_column(df: pd.DataFrame):
    cols = df.columns.tolist()
    for alias in TEXT_ALIASES:
        if alias in cols:
            return alias, "exact"
    lower_map = {c.lower(): c for c in cols}
    for alias in TEXT_ALIASES:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()], "case-insensitive"
    str_cols = df.select_dtypes(include="object").columns.tolist()
    avg_lens = {c: df[c].dropna().astype(str).str.len().mean() for c in str_cols}
    long_cols = sorted([c for c, l in avg_lens.items() if l > 30], key=lambda c: -avg_lens[c])
    if len(long_cols) == 1:
        return long_cols[0], "inferred"
    if long_cols:
        return long_cols[0], "ambiguous"
    return None, "not_found"


def fast_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "source"    not in df.columns: df["source"]    = "external"
    if "subreddit" not in df.columns: df["subreddit"] = "unknown"
    if "clean_text" not in df.columns:
        df["clean_text"] = df["text"].fillna("").apply(clean_text)
    if "pronoun_type" not in df.columns:
        pron = pd.DataFrame(df["clean_text"].apply(tag_pronouns).tolist(), index=df.index)
        df = pd.concat([df, pron], axis=1)
    if "has_othering" not in df.columns:
        df = apply_othering(df, text_col="clean_text")
    if "othering_predicted" not in df.columns:
        df["othering_predicted"] = df["has_othering"].astype(int)
    if "othering_proba" not in df.columns:
        df["othering_proba"] = df["othering_score"] / 4.0
    for col in ["toxicity", "severe_toxicity", "identity_attack", "insult", "threat", "emotion", "emotion_score"]:
        if col not in df.columns:
            df[col] = np.nan
    df["subreddit"]    = df["subreddit"].fillna("unknown")
    df["source"]       = df["source"].fillna("external")
    df["pronoun_type"] = df["pronoun_type"].fillna("none")
    return df


def run_full_pipeline(df: pd.DataFrame, run_detoxify: bool, run_emotions: bool, progress) -> pd.DataFrame:
    df = fast_pipeline(df)
    progress.progress(0.35, "Base analysis done.")
    if run_detoxify and df["toxicity"].isna().all():
        progress.progress(0.40, "Running Detoxify...")
        try:
            from detoxify import Detoxify
            model = Detoxify("original")
            texts = df["clean_text"].tolist()
            tox_cols = ["toxicity", "severe_toxicity", "identity_attack", "insult", "threat"]
            all_res  = {c: [] for c in tox_cols}
            for i in range(0, len(texts), 64):
                res = model.predict(texts[i:i + 64])
                for c in tox_cols:
                    all_res[c].extend(res[c])
                progress.progress(0.40 + 0.35 * min((i + 64) / len(texts), 1.0),
                                   f"Detoxify — {min(i+64, len(texts))}/{len(texts)}")
            for c in tox_cols:
                df[c] = all_res[c]
        except ImportError:
            st.warning("Detoxify not installed.")
    if run_emotions and df["emotion"].isna().all():
        progress.progress(0.78, "Running GoEmotions...")
        try:
            from transformers import pipeline as hf_pipeline
            pipe = hf_pipeline("text-classification",
                               model="monologg/bert-base-cased-goemotions-original", top_k=1)
            texts = df["clean_text"].str[:512].tolist()
            emotions, scores = [], []
            for i in range(0, len(texts), 16):
                res = pipe(texts[i:i + 16])
                emotions.extend(r[0]["label"] for r in res)
                scores.extend(r[0]["score"]   for r in res)
                progress.progress(0.78 + 0.18 * min((i + 16) / len(texts), 1.0),
                                   f"GoEmotions — {min(i+16, len(texts))}/{len(texts)}")
            df["emotion"]       = emotions
            df["emotion_score"] = scores
        except ImportError:
            st.warning("Transformers not installed.")
    progress.progress(1.0, "Done.")
    return df


# ── Dataset loading ───────────────────────────────────────────────────────────

def discover_datasets() -> dict:
    groups = {}
    defaults = []
    if FINAL_CSV.exists():
        defaults.append(("dataset_final.csv", FINAL_CSV))
    if CLASSIFIED_CSV.exists():
        defaults.append(("dataset_classified.csv", CLASSIFIED_CSV))
    if defaults:
        groups["Default"] = defaults
    if DATASETS_DIR.exists():
        for folder in sorted(DATASETS_DIR.iterdir()):
            if not folder.is_dir():
                continue
            files = sorted(f for f in folder.iterdir() if f.suffix.lower() in (".csv", ".xlsx"))
            if files:
                groups[folder.name.capitalize()] = [(f.name, f) for f in files]
    return groups


@st.cache_data(show_spinner=False)
def load_single(path_str: str, is_default: bool) -> pd.DataFrame:
    path = Path(path_str)
    if is_default:
        df = pd.read_csv(path, low_memory=False)
        for col in ["othering_predicted", "othering_proba", "othering_score", "toxicity"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "othering_predicted" in df.columns:
            df["othering_predicted"] = df["othering_predicted"].fillna(0).astype(int)
        for col_name, default in [("subreddit", "unknown"), ("source", "unknown"),
                                   ("pronoun_type", "unknown")]:
            if col_name not in df.columns:
                df[col_name] = default
            else:
                df[col_name] = df[col_name].fillna(default)
        for col in ["toxicity", "emotion", "emotion_score"]:
            if col not in df.columns:
                df[col] = np.nan
    else:
        df = pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path, low_memory=False)
        col, _ = detect_text_column(df)
        if col and col != "text":
            df = df.rename(columns={col: "text"})
        elif col is None:
            df["text"] = ""
        df = fast_pipeline(df)
    return df


def build_combined() -> pd.DataFrame:
    groups   = discover_datasets()
    frames   = []
    errors   = []
    for group, files in groups.items():
        is_def = group == "Default"
        for name, path in files:
            label = f"{group}  ·  {name}"
            try:
                df = load_single(str(path), is_def)
                df = df.copy()
                df["dataset"] = label
                frames.append(df)
            except Exception as e:
                errors.append(f"{label}: {e}")
    if errors:
        st.sidebar.warning("Some datasets failed to load:\n" + "\n".join(errors))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    # Ensure consistent types
    combined["othering_predicted"] = pd.to_numeric(
        combined.get("othering_predicted", 0), errors="coerce").fillna(0).astype(int)
    combined["othering_score"] = pd.to_numeric(
        combined.get("othering_score", 0), errors="coerce").fillna(0)
    return combined


# ── Build combined dataset (cached in session_state across reruns) ─────────────

if "df_combined" not in st.session_state:
    with st.spinner("Loading all datasets..."):
        st.session_state["df_combined"] = build_combined()

df_all = st.session_state["df_combined"]

# Merge uploaded dataset if present
if "uploaded_df" in st.session_state:
    udf = st.session_state["uploaded_df"].copy()
    udf["dataset"] = f"📎  {st.session_state.get('upload_name', 'Uploaded')}"
    if udf["dataset"].iloc[0] not in df_all["dataset"].values:
        df_all = pd.concat([udf, df_all], ignore_index=True)
        st.session_state["df_combined"] = df_all

all_dataset_labels = sorted(df_all["dataset"].unique().tolist())


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("We vs Them")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Sentiment & Toxicity", "Emotions", "Othering & Topics", "Upload & Analyze"],
)

st.sidebar.markdown(" ")

if page != "Upload & Analyze":

    selected_datasets = st.sidebar.multiselect(
        "Datasets",
        options=all_dataset_labels,
        default=all_dataset_labels,
        placeholder="Select datasets...",
    )

    if not selected_datasets:
        st.warning("Select at least one dataset.")
        st.stop()

    df_full = df_all[df_all["dataset"].isin(selected_datasets)].copy()

    st.sidebar.markdown(" ")
    keyword = st.sidebar.text_input("Keyword search", placeholder="e.g. immigrant, they...")
    known_subreddits = sorted(
        s for s in df_full["subreddit"].unique()
        if s not in ("unknown", "none", "") and pd.notna(s)
    )
    selected_subreddits = st.sidebar.multiselect(
        "Filter by subreddit / account",
        options=known_subreddits, default=[], placeholder="All",
    )

    df = df_full.copy()
    if keyword.strip():
        df = df[df["text"].fillna("").str.contains(keyword.strip(), case=False, na=False)]
    if selected_subreddits:
        df = df[df["subreddit"].isin(selected_subreddits)]

    if len(df) == 0:
        st.warning("No results for these filters.")
        st.stop()

    has_topics = "topic" in df_full.columns and "topic_name" in df_full.columns


# ── Upload & Analyze ──────────────────────────────────────────────────────────

if page == "Upload & Analyze":
    st.title("Upload & Analyze")
    st.write("Drop any CSV — the text column is detected automatically. The result is added to all other pages.")

    uploaded = st.file_uploader("Drag & drop your CSV here", type=["csv"],
                                label_visibility="collapsed")

    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded, low_memory=False)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        st.success(f"File loaded: **{uploaded.name}** — {len(raw):,} rows, {len(raw.columns)} columns")

        detected_col, confidence = detect_text_column(raw)
        str_cols = raw.select_dtypes(include="object").columns.tolist()

        if confidence == "not_found":
            st.error("No text column found. Select one manually.")
            text_col = st.selectbox("Column to use as text", options=raw.columns.tolist())
        elif confidence == "ambiguous":
            st.warning(f"Multiple text-like columns found. Best guess: **`{detected_col}`**")
            text_col = st.selectbox("Column to use as text", options=str_cols,
                                    index=str_cols.index(detected_col) if detected_col in str_cols else 0)
        else:
            label = {"exact": "exact match", "case-insensitive": "case-insensitive match",
                     "inferred": "inferred from content"}[confidence]
            st.info(f"Text column detected: **`{detected_col}`** ({label})")
            text_col = st.selectbox("Column to use as text", options=str_cols,
                                    index=str_cols.index(detected_col) if detected_col in str_cols else 0,
                                    label_visibility="collapsed")

        if text_col != "text":
            raw = raw.rename(columns={text_col: "text"})

        with st.expander("Preview (first 5 rows)", expanded=False):
            st.dataframe(raw.head(), use_container_width=True)

        already_processed = all(c in raw.columns for c in ["has_othering", "othering_predicted", "pronoun_type"])
        if already_processed:
            st.info("This file already contains analysis columns — no reprocessing needed.")

        col1, col2 = st.columns(2)
        with col1:
            max_rows = st.slider("Max rows to process",
                                 min_value=100, max_value=min(50_000, len(raw)),
                                 value=min(5_000, len(raw)), step=100,
                                 disabled=already_processed)
        with col2:
            run_detoxify = st.checkbox("Run Detoxify (slow)", value=False,
                                       disabled=already_processed or "toxicity" in raw.columns)
            run_emotions = st.checkbox("Run GoEmotions (slow)", value=False,
                                       disabled=already_processed or "emotion" in raw.columns)

        if st.button("Run analysis", type="primary"):
            df_input = raw if already_processed else raw.head(max_rows)
            progress = st.progress(0.0, "Starting...")
            try:
                if already_processed:
                    result = df_input.copy()
                    for col in ["toxicity", "othering_proba", "othering_score"]:
                        if col in result.columns:
                            result[col] = pd.to_numeric(result[col], errors="coerce")
                    progress.progress(1.0, "Done.")
                else:
                    result = run_full_pipeline(df_input, run_detoxify, run_emotions, progress)

                st.session_state["uploaded_df"]  = result
                st.session_state["upload_name"]  = uploaded.name
                # Inject into combined dataframe
                udf = result.copy()
                udf["dataset"] = f"📎  {uploaded.name}"
                existing = st.session_state["df_combined"]
                existing = existing[existing["dataset"] != udf["dataset"].iloc[0]]
                st.session_state["df_combined"] = pd.concat([udf, existing], ignore_index=True)
                st.success(f"Done — {len(result):,} posts added to the dashboard. Switch to any page to explore.")
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline error: {e}")

        if "uploaded_df" in st.session_state and st.session_state.get("upload_name") == uploaded.name:
            result = st.session_state["uploaded_df"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{len(result):,}")
            c2.metric("% othering", f"{result['othering_predicted'].mean()*100:.1f}%")
            tox = result["toxicity"].mean() if result["toxicity"].notna().any() else None
            c3.metric("Avg toxicity", f"{tox:.3f}" if tox else "n/a")
            c4.metric("% them markers",
                      f"{result['pronoun_type'].isin(['them_only','both']).mean()*100:.1f}%")
            with st.expander("Download processed CSV"):
                st.download_button("Download result.csv",
                                   data=result.to_csv(index=False).encode("utf-8"),
                                   file_name="result_analyzed.csv", mime="text/csv")

    if st.button("Reload all datasets", help="Force reload if you added new files to data/Datasets"):
        del st.session_state["df_combined"]
        st.rerun()


# ── Overview ──────────────────────────────────────────────────────────────────

elif page == "Overview":
    st.title("Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total posts", f"{len(df):,}")
    c2.metric("% Othering detected", f"{df['othering_predicted'].mean()*100:.1f}%")
    tox_ok = df["toxicity"].notna().any()
    c3.metric("Highly toxic (>0.5)",
              f"{(df['toxicity'].dropna() > 0.5).mean()*100:.1f}%" if tox_ok else "n/a")

    st.subheader("Posts by dataset")
    ds_counts = df["dataset"].value_counts().reset_index()
    ds_counts.columns = ["dataset", "count"]
    ds_counts["pct"] = (ds_counts["count"] / ds_counts["count"].sum() * 100).round(1)
    fig = px.bar(ds_counts, x="dataset", y="count",
                 text=ds_counts.apply(lambda r: f"{r['count']:,} ({r['pct']}%)", axis=1),
                 labels={"dataset": "", "count": "Posts"})
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Pronoun type distribution")
        pron = (df[~df["pronoun_type"].isin(["unknown", "none", ""])]
                ["pronoun_type"].value_counts().reset_index())
        pron.columns = ["pronoun_type", "count"]
        if pron.empty:
            st.info("No pronoun data.")
        else:
            fig2 = px.bar(pron, x="pronoun_type", y="count", text="count")
            fig2.update_traces(textposition="outside")
            st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        st.subheader("Othering rate by dataset")
        oth_by_ds = (df.groupby("dataset")["othering_predicted"]
                     .agg(["sum", "count"]).reset_index())
        oth_by_ds.columns = ["dataset", "othering", "total"]
        oth_by_ds["pct"] = oth_by_ds["othering"] / oth_by_ds["total"] * 100
        oth_by_ds = oth_by_ds.sort_values("pct", ascending=True)
        fig3 = px.bar(oth_by_ds, x="pct", y="dataset", orientation="h",
                      text=oth_by_ds["pct"].round(1).astype(str) + "%",
                      labels={"pct": "% othering", "dataset": ""})
        fig3.update_traces(textposition="outside")
        st.plotly_chart(fig3, use_container_width=True)


# ── Sentiment & Toxicity ──────────────────────────────────────────────────────

elif page == "Sentiment & Toxicity":
    st.title("Sentiment & Toxicity")

    df_tox = df.dropna(subset=["toxicity"])
    if df_tox.empty:
        st.info("No toxicity data in the selected datasets. Run Detoxify via Upload & Analyze, or select a default dataset.")
    else:
        st.subheader("Toxicity distribution")
        fig = px.histogram(df_tox, x="toxicity", color="dataset", nbins=60, barmode="overlay",
                           opacity=0.6, labels={"toxicity": "Toxicity score"})
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Average toxicity by dataset")
        tox_ds = (df_tox.groupby("dataset")["toxicity"]
                  .agg(["mean", "median", "count"]).reset_index()
                  .rename(columns={"mean": "Mean", "median": "Median", "count": "Posts"}))
        tox_ds[["Mean", "Median"]] = tox_ds[["Mean", "Median"]].round(4)
        st.dataframe(tox_ds, use_container_width=True, hide_index=True)

        st.subheader("Toxicity by othering detection")
        box = df_tox.copy()
        box["Othering"] = box["othering_predicted"].map({0: "No", 1: "Yes"})
        fig2 = px.box(box, x="Othering", y="toxicity", color="dataset",
                      labels={"toxicity": "Toxicity score"}, points=False)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Top 10 most toxic posts")
        top10 = (df_tox.nlargest(10, "toxicity")
                 [["text", "dataset", "toxicity", "othering_predicted"]]
                 .reset_index(drop=True))
        top10.index += 1
        st.dataframe(top10, use_container_width=True)


# ── Emotions ──────────────────────────────────────────────────────────────────

elif page == "Emotions":
    st.title("Emotions")
    df_emo = df.dropna(subset=["emotion"])

    if df_emo.empty:
        st.info("No emotion data in the selected datasets. Run GoEmotions via Upload & Analyze, or select a default dataset.")
    else:
        pct = len(df_emo) / len(df) * 100
        if pct < 99:
            st.warning(f"Emotion data covers **{len(df_emo):,} posts** ({pct:.1f}%). "
                       f"{len(df) - len(df_emo):,} posts excluded.")

        st.subheader("Emotion distribution by dataset")
        emo = df_emo.groupby(["dataset", "emotion"]).size().reset_index(name="count")
        emo["pct"] = emo.groupby("dataset")["count"].transform(lambda x: x / x.sum() * 100).round(1)
        fig = px.bar(emo, x="emotion", y="pct", color="dataset", barmode="group",
                     labels={"pct": "% of posts", "emotion": "Emotion"})
        fig.update_layout(xaxis={"categoryorder": "total descending"})
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Emotions: othering vs non-othering")
        split = df_emo.copy()
        split["group"] = split["othering_predicted"].map({1: "Othering", 0: "Non-othering"})
        es = split.groupby(["group", "emotion"]).size().reset_index(name="n")
        es["pct"] = es.groupby("group")["n"].transform(lambda x: x / x.sum() * 100).round(2)
        fig2 = px.bar(es, x="emotion", y="pct", color="group", barmode="group",
                      labels={"pct": "% of posts", "emotion": "Emotion", "group": ""})
        fig2.update_layout(xaxis={"categoryorder": "total descending"})
        st.plotly_chart(fig2, use_container_width=True)


# ── Othering & Topics ─────────────────────────────────────────────────────────

elif page == "Othering & Topics":
    st.title("Othering & Topics")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Othering rate by dataset")
        oth_ds = (df.groupby("dataset")["othering_predicted"]
                  .agg(["sum", "count"]).reset_index())
        oth_ds.columns = ["dataset", "othering", "total"]
        oth_ds["pct"] = oth_ds["othering"] / oth_ds["total"] * 100
        oth_ds = oth_ds.sort_values("pct", ascending=True)
        fig = px.bar(oth_ds, x="pct", y="dataset", orientation="h",
                     text=oth_ds["pct"].round(1).astype(str) + "%",
                     labels={"pct": "% othering", "dataset": ""})
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Othering rate by pronoun type")
        oth_pron = (df[~df["pronoun_type"].isin(["unknown", "none", ""])]
                    .groupby("pronoun_type")["othering_predicted"]
                    .agg(["sum", "count"]).reset_index())
        oth_pron.columns = ["pronoun_type", "othering", "total"]
        oth_pron["pct"] = oth_pron["othering"] / oth_pron["total"] * 100
        fig2 = px.bar(oth_pron, x="pronoun_type", y="pct",
                      text=oth_pron["pct"].round(1).astype(str) + "%",
                      labels={"pct": "% othering", "pronoun_type": "Pronoun type"})
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Post examples")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        smin = float(df["othering_score"].min()) if not df["othering_score"].isna().all() else 0.0
        smax = float(df["othering_score"].max()) if not df["othering_score"].isna().all() else 3.0
        score_range = st.slider("othering_score", min_value=smin, max_value=smax,
                                value=(smin, smax), step=1.0)
    with c2:
        ds_filter = st.multiselect("Dataset", options=sorted(df["dataset"].unique()),
                                   default=[], placeholder="All")
    with c3:
        ks = sorted(s for s in df["subreddit"].unique() if s not in ("unknown", "none", ""))
        sub_filter = st.multiselect("Subreddit / account", options=ks, default=[], placeholder="All")
    with c4:
        othering_only = st.checkbox("Detected othering only", value=False)

    ex = df.dropna(subset=["othering_score"])
    ex = ex[(ex["othering_score"] >= score_range[0]) & (ex["othering_score"] <= score_range[1])]
    if ds_filter:     ex = ex[ex["dataset"].isin(ds_filter)]
    if sub_filter:    ex = ex[ex["subreddit"].isin(sub_filter)]
    if othering_only: ex = ex[ex["othering_predicted"] == 1]

    st.caption(f"{len(ex):,} posts match the current filters.")
    show_cols = [c for c in ["text", "dataset", "othering_score", "othering_proba",
                             "matched_patterns", "toxicity"] if c in ex.columns]
    st.dataframe(ex[show_cols].head(200).reset_index(drop=True), use_container_width=True)

    if has_topics and "topic" in df.columns and "topic_name" in df.columns:
        st.subheader("BERTopic topics — top 20 by size")
        topic_df = df.dropna(subset=["topic", "topic_name"]).copy()
        topic_df["topic"] = topic_df["topic"].astype(int)
        tstats = (topic_df[topic_df["topic"] >= 0]
                  .groupby(["topic", "topic_name"])
                  .agg(nb_posts=("topic", "count"),
                       mean_toxicity=("toxicity", "mean"),
                       pct_othering=("othering_predicted", "mean"))
                  .reset_index())
        tstats["pct_othering"]  = (tstats["pct_othering"] * 100).round(1)
        tstats["mean_toxicity"] = tstats["mean_toxicity"].round(3)
        tstats["label"] = tstats["topic"].astype(str) + " — " + tstats["topic_name"]
        top20 = tstats.nlargest(20, "nb_posts")
        fig3 = px.bar(top20.sort_values("nb_posts", ascending=True),
                      x="nb_posts", y="label", orientation="h", text="nb_posts",
                      labels={"nb_posts": "Posts", "label": "Topic"})
        fig3.update_traces(textposition="outside")
        fig3.update_layout(height=600)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No BERTopic data in the current selection.")
