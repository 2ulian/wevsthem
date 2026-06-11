import sys
import re
import json as _json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from cleaning import clean_text
from othering import apply_othering
from classifier import load_model

_CLF_PATH = BASE_DIR / "models" / "othering_classifier.pkl"
_MTR_PATH = BASE_DIR / "models" / "othering_metrics.json"

try:
    _CLASSIFIER = load_model(str(_CLF_PATH))
except Exception:
    _CLASSIFIER = None

try:
    with open(_MTR_PATH) as _f:
        _CLF_METRICS = _json.load(_f)
except Exception:
    _CLF_METRICS = None

FINAL_CSV      = BASE_DIR / "data" / "processed" / "dataset_final.csv"
CLASSIFIED_CSV = BASE_DIR / "data" / "processed" / "dataset_classified.csv"
DATASETS_DIR   = BASE_DIR / "data" / "datasets"
CUSTOM_DIR     = DATASETS_DIR / "imported"

WE_WORDS     = ["we", "us", "our", "ours", "ourselves"]
THEM_WORDS   = ["they", "them", "their", "theirs"]
THEM_PHRASES = ["those people", "these people", "people like them"]

TEXT_ALIASES = [
    "text", "Text", "TEXT", "caption", "Caption",
    "tweet", "Tweet", "tweet_text", "comment", "Comment", "comment_text",
    "body", "Body", "content", "Content", "message", "Message", "description",
    "transcription", "video_transcription_text",
    "parentText", "childCommentText", "post", "post_text", "selftext", "title",
]

PLATFORM_EMOJI = {
    "Default":   "◈",
    "Instagram": "◎",
    "Sentiment": "◉",
    "Tiktok":    "▶",
    "Twitter":   "◆",
    "Imported":  "▲",
}

EMOTION_VALENCE = {
    "admiration": "positive",  "amusement": "positive",   "approval": "positive",
    "caring":     "positive",  "desire":    "positive",   "excitement": "positive",
    "gratitude":  "positive",  "joy":       "positive",   "love":      "positive",
    "optimism":   "positive",  "pride":     "positive",   "relief":    "positive",
    "anger":      "negative",  "annoyance": "negative",   "confusion": "negative",
    "disappointment": "negative", "disapproval": "negative", "disgust": "negative",
    "embarrassment":  "negative", "fear":    "negative",  "grief":     "negative",
    "nervousness": "negative", "remorse":   "negative",   "sadness":   "negative",
    "curiosity":  "neutral",   "realization": "neutral",  "surprise":  "neutral",
    "neutral":    "neutral",
}
VALENCE_COLOR = {"positive": "#10b981", "negative": "#e11d48", "neutral": "#7070a0"}


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


def fast_pipeline(df: pd.DataFrame, run_othering: bool = True) -> pd.DataFrame:
    df = df.copy()
    if "source"    not in df.columns: df["source"]    = "external"
    if "subreddit" not in df.columns: df["subreddit"] = "unknown"
    if "clean_text" not in df.columns:
        df["clean_text"] = df["text"].fillna("").apply(clean_text)
    if "pronoun_type" not in df.columns:
        pron = pd.DataFrame(df["clean_text"].apply(tag_pronouns).tolist(), index=df.index)
        df = pd.concat([df, pron], axis=1)
    if run_othering and "has_othering" not in df.columns:
        df = apply_othering(df, text_col="clean_text")
    if "othering_predicted" not in df.columns:
        if _CLASSIFIER is not None and run_othering:
            _vect  = _CLASSIFIER.get("vectorizer")
            _mdl   = _CLASSIFIER.get("model")
            _texts = df["clean_text"].fillna("").tolist()
            _X     = _vect.transform(_texts)
            df["othering_predicted"] = _mdl.predict(_X).astype(int)
            df["othering_proba"]     = _mdl.predict_proba(_X)[:, 1].round(4)
        else:
            df["othering_predicted"] = df["has_othering"].astype(int) if "has_othering" in df.columns else 0
    if "othering_proba" not in df.columns:
        df["othering_proba"] = df["othering_score"] / 4.0 if "othering_score" in df.columns else 0.0
    for col in ["toxicity", "severe_toxicity", "identity_attack", "insult", "threat", "emotion", "emotion_score"]:
        if col not in df.columns:
            df[col] = np.nan
    df["subreddit"]    = df["subreddit"].fillna("unknown")
    df["source"]       = df["source"].fillna("external")
    df["pronoun_type"] = df["pronoun_type"].fillna("none")
    return df


def run_full_pipeline(df: pd.DataFrame, run_othering: bool, run_detoxify: bool,
                      run_emotions: bool, run_bertopic: bool, progress) -> pd.DataFrame:
    n_steps = 1 + run_detoxify + run_emotions + run_bertopic
    step = 0

    def _p(frac, msg):
        progress.progress((step + frac) / n_steps, msg)

    df = fast_pipeline(df, run_othering=run_othering)
    step += 1
    progress.progress(step / n_steps, "Base analysis done.")

    if run_detoxify and df["toxicity"].isna().all():
        _p(0.0, "Running Detoxify...")
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
                _p(min((i + 64) / len(texts), 1.0),
                   f"Detoxify — {min(i+64, len(texts))}/{len(texts)}")
            for c in tox_cols:
                df[c] = all_res[c]
        except ImportError:
            st.warning("Detoxify not installed.")
        step += 1
        progress.progress(step / n_steps, "Detoxify done.")

    if run_emotions and df["emotion"].isna().all():
        _p(0.0, "Running GoEmotions...")
        try:
            import torch
            from transformers import pipeline as hf_pipeline
            _device = 0 if torch.cuda.is_available() else -1
            _batch  = 128 if _device == 0 else 64
            pipe = hf_pipeline("text-classification",
                               model="monologg/bert-base-cased-goemotions-original",
                               top_k=1, device=_device,
                               truncation=True, max_length=128)
            texts = df["clean_text"].str[:512].tolist()
            emotions, scores = [], []
            for i in range(0, len(texts), _batch):
                res = pipe(texts[i:i + _batch])
                emotions.extend(r[0]["label"] for r in res)
                scores.extend(r[0]["score"]   for r in res)
                _p(min((i + _batch) / len(texts), 1.0),
                   f"GoEmotions — {min(i+_batch, len(texts))}/{len(texts)}")
            df["emotion"]       = emotions
            df["emotion_score"] = scores
        except ImportError:
            st.warning("Transformers not installed.")
        step += 1
        progress.progress(step / n_steps, "GoEmotions done.")

    if run_bertopic:
        _p(0.0, "Running BERTopic — encoding texts...")
        try:
            from sentence_transformers import SentenceTransformer
            from bertopic import BERTopic
            from hdbscan import HDBSCAN
            from umap import UMAP

            text_col = "clean_text" if "clean_text" in df.columns else "text"
            docs = df[text_col].fillna("").tolist()

            st_model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = st_model.encode(docs, batch_size=32, show_progress_bar=False,
                                         convert_to_numpy=True)
            _p(0.35, "BERTopic — fitting UMAP...")

            umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                              metric="cosine", random_state=42)
            umap_embeddings = umap_model.fit_transform(embeddings)
            _p(0.65, "BERTopic — clustering...")

            min_topic_size = max(5, len(docs) // 150)
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
                verbose=False,
                calculate_probabilities=False,
            )
            topics, _ = topic_model.fit_transform(docs, embeddings=umap_embeddings)
            _p(0.88, "BERTopic — reducing outliers...")
            topics = topic_model.reduce_outliers(docs, topics, strategy="c-tf-idf")
            topic_model.update_topics(docs, topics=topics)

            df["topic"] = topics

            def _topic_name(tid):
                if tid == -1:
                    return "outlier"
                words = topic_model.get_topic(tid)
                return "_".join(w for w, _ in words[:3]) if words else f"topic_{tid}"

            df["topic_name"] = df["topic"].apply(_topic_name)
        except ImportError as e:
            st.warning(f"BERTopic not installed: {e}")
        step += 1
        progress.progress(step / n_steps, "BERTopic done.")

    progress.progress(1.0, "Done.")
    return df


def discover_datasets() -> dict:
    groups = {}
    defaults = []
    if FINAL_CSV.exists():
        defaults.append(("dataset_final.csv", FINAL_CSV))
    if defaults:
        groups["Default"] = defaults
    if DATASETS_DIR.exists():
        for folder in sorted(DATASETS_DIR.iterdir()):
            if not folder.is_dir():
                continue
            files = sorted(f for f in folder.iterdir() if f.suffix.lower() == ".csv")
            if files:
                groups[folder.name.capitalize()] = [(f.name, f) for f in files]
    if "Imported" in groups:
        groups["Imported"] = groups.pop("Imported")
    return groups


_DATE_COL_NAMES = [
    "createTimeISO", "createdAt", "Timestamp", "timestamp", "created_at",
    "date", "Date", "datetime", "childCommentDate", "date Parent", "posted_at",
]
_DATE_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
    "%a %b %d %H:%M:%S %z %Y",
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
]


def _parse_date_col(df: pd.DataFrame) -> "pd.Series | None":
    for col in _DATE_COL_NAMES:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        parsed = pd.to_datetime(s, format="mixed", errors="coerce", utc=True)
        if parsed.notna().sum() / len(s) >= 0.5:
            result = pd.to_datetime(df[col], format="mixed", errors="coerce", utc=True)
            return result.dt.tz_localize(None)
    return None


@st.cache_data(show_spinner=False)
def load_single(path_str: str, is_default: bool) -> pd.DataFrame:
    path = Path(path_str)
    already_analyzed = is_default or (CUSTOM_DIR in path.parents)
    df = pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path, low_memory=False)
    col, _ = detect_text_column(df)
    if col and col != "text":
        df = df.rename(columns={col: "text"})
    elif col is None:
        df["text"] = ""
    if already_analyzed and "othering_predicted" in df.columns:
        for col in ["othering_predicted", "othering_proba", "othering_score", "toxicity"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
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
        df = fast_pipeline(df)
    if "post_date" not in df.columns:
        parsed = _parse_date_col(df)
        if parsed is not None:
            df["post_date"] = parsed
    return df


def build_combined() -> pd.DataFrame:
    groups = discover_datasets()
    frames, errors = [], []
    for group, files in groups.items():
        is_def = group == "Default"
        for name, path in files:
            label = f"{group}  ·  {name}"
            try:
                df = load_single(str(path), is_def)
                df = df.copy()
                df = df.loc[:, ~df.columns.duplicated()]
                df["dataset"] = label
                frames.append(df)
            except Exception as e:
                errors.append(f"{label}: {e}")
    if errors:
        st.sidebar.warning("Load errors:\n" + "\n".join(errors))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["othering_predicted"] = pd.to_numeric(
        combined.get("othering_predicted", 0), errors="coerce").fillna(0).astype(int)
    combined["othering_score"] = pd.to_numeric(
        combined.get("othering_score", 0), errors="coerce").fillna(0)
    return combined
