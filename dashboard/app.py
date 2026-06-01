import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import re
import html as _html
import ast as _ast

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from cleaning import clean_text
from othering import apply_othering

st.set_page_config(layout="wide", page_title="We vs Them", page_icon="⚡")

# ── Design system ─────────────────────────────────────────────────────────────

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg:       #0c0c10;
  --surface:  #13131a;
  --card:     #1a1a26;
  --border:   #2a2a3a;
  --accent:   #7c3aed;
  --accent2:  #e11d48;
  --accent3:  #0ea5e9;
  --text:     #e8e8f0;
  --muted:    #7070a0;
  --success:  #10b981;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
}

[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

h1, h2, h3, h4 {
  font-family: 'Syne', sans-serif !important;
  letter-spacing: -0.02em;
}

.mono { font-family: 'IBM Plex Mono', monospace !important; }

/* KPI cards */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 32px; }
.kpi-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px 24px;
  position: relative;
  overflow: hidden;
}
.kpi-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: var(--accent-color, var(--accent));
}
.kpi-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 8px;
}
.kpi-value {
  font-family: 'Syne', sans-serif;
  font-size: 32px;
  font-weight: 800;
  color: var(--text);
  line-height: 1;
  margin-bottom: 4px;
}
.kpi-sub {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  color: var(--muted);
}

/* Section headers */
.section-header {
  font-family: 'Syne', sans-serif;
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 16px;
  margin-top: 8px;
}

/* Post cards */
.post-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: 8px;
  padding: 14px 16px;
  margin-bottom: 10px;
  font-size: 14px;
  line-height: 1.6;
}
.post-card.toxic { border-left-color: var(--accent2); }
.post-card .meta {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  color: var(--muted);
  margin-top: 8px;
  display: flex;
  gap: 16px;
}
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  font-weight: 500;
}
.badge-red   { background: rgba(225,29,72,0.15);  color: #fb7185; }
.badge-purple{ background: rgba(124,58,237,0.15); color: #a78bfa; }
.badge-blue  { background: rgba(14,165,233,0.15); color: #38bdf8; }
.badge-green { background: rgba(16,185,129,0.15); color: #34d399; }
.badge-gray  { background: rgba(112,112,160,0.15);color: #a0a0c0; }

/* Nav pills */
.nav-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border-radius: 8px;
  font-family: 'Syne', sans-serif;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s;
  border: 1px solid transparent;
  width: 100%;
  margin-bottom: 4px;
}
.nav-pill.active {
  background: rgba(124,58,237,0.2);
  border-color: rgba(124,58,237,0.4);
  color: #c4b5fd !important;
}
.nav-pill:not(.active) {
  color: var(--muted) !important;
}
.nav-pill:not(.active):hover {
  background: rgba(255,255,255,0.04);
}

/* Dataset pills */
.ds-pill-container { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
.ds-pill {
  padding: 4px 10px;
  border-radius: 20px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  cursor: pointer;
  transition: all 0.15s;
  border: 1px solid var(--border);
  white-space: nowrap;
}
.ds-pill.on  { background: rgba(124,58,237,0.25); border-color: rgba(124,58,237,0.5); color: #c4b5fd; }
.ds-pill.off { background: transparent; color: var(--muted); }

/* Upload zone */
.upload-zone {
  border: 2px dashed var(--border);
  border-radius: 16px;
  padding: 48px;
  text-align: center;
  background: var(--card);
  transition: all 0.2s;
}
.upload-zone:hover { border-color: var(--accent); }

/* Page title */
.page-title {
  font-family: 'Syne', sans-serif;
  font-size: 28px;
  font-weight: 800;
  color: var(--text);
  letter-spacing: -0.03em;
  margin-bottom: 4px;
}
.page-subtitle {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 12px;
  color: var(--muted);
  margin-bottom: 28px;
}

/* Streamlit overrides */
.stButton button {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 12px !important;
  border-radius: 6px !important;
  transition: all 0.15s !important;
}
.stButton button:hover {
  border-color: var(--accent) !important;
  color: #c4b5fd !important;
}
[data-testid="stMetric"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 16px !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Syne', sans-serif !important;
  font-size: 28px !important;
  color: var(--text) !important;
}
[data-testid="stMetricLabel"] {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 11px !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
}
div[data-testid="stDataFrame"] { border-radius: 8px !important; overflow: hidden; }
.stSelectbox > div, .stMultiSelect > div, .stTextInput > div > div {
  background: var(--card) !important;
  border-color: var(--border) !important;
}
.stAlert { border-radius: 8px !important; }
div[data-baseweb="tab-list"] { background: var(--surface) !important; }

/* Sidebar title */
.sidebar-brand {
  font-family: 'Syne', sans-serif;
  font-size: 18px;
  font-weight: 800;
  letter-spacing: -0.02em;
  padding: 8px 0 20px 0;
  display: flex;
  align-items: center;
  gap: 8px;
}
.sidebar-brand .we { color: var(--accent3); }
.sidebar-brand .vs { color: var(--muted); font-size: 14px; }
.sidebar-brand .them { color: var(--accent2); }

hr.divider {
  border: none;
  border-top: 1px solid var(--border);
  margin: 16px 0;
}

/* Plotly chart containers */
.js-plotly-plot .plotly { background: transparent !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────

PALETTE   = ["#7c3aed", "#e11d48", "#0ea5e9", "#10b981", "#f59e0b", "#ec4899", "#06b6d4", "#84cc16"]
CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", size=11, color="#7070a0"),
    title=dict(text="", font=dict(family="Syne, sans-serif", size=14, color="#e8e8f0")),
    margin=dict(l=0, r=0, t=36, b=0),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    colorway=PALETTE,
    xaxis=dict(gridcolor="#1e1e2e", linecolor="#2a2a3a", zeroline=False, title=""),
    yaxis=dict(gridcolor="#1e1e2e", linecolor="#2a2a3a", zeroline=False, title=""),
)

def apply_theme(fig, height=380):
    fig.update_layout(**CHART_LAYOUT, height=height)
    return fig

def hex_rgba(hex_color: str, alpha: float = 0.2) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Constants ─────────────────────────────────────────────────────────────────

FINAL_CSV      = BASE_DIR / "data" / "processed" / "dataset_final.csv"
CLASSIFIED_CSV = BASE_DIR / "data" / "processed" / "dataset_classified.csv"
DATASETS_DIR   = BASE_DIR / "data" / "Datasets"

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
    if defaults:
        groups["Default"] = defaults
    if DATASETS_DIR.exists():
        for folder in sorted(DATASETS_DIR.iterdir()):
            if not folder.is_dir():
                continue
            files = sorted(f for f in folder.iterdir() if f.suffix.lower() == ".csv")
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


# ── Bootstrap data ────────────────────────────────────────────────────────────

if "df_combined" not in st.session_state:
    with st.spinner("Loading datasets..."):
        st.session_state["df_combined"] = build_combined()

df_all = st.session_state["df_combined"]

if "uploaded_df" in st.session_state:
    udf = st.session_state["uploaded_df"].copy()
    udf["dataset"] = f"📎  {st.session_state.get('upload_name', 'Uploaded')}"
    if udf["dataset"].iloc[0] not in df_all["dataset"].values:
        df_all = pd.concat([udf, df_all], ignore_index=True)
        st.session_state["df_combined"] = df_all

all_dataset_labels = sorted(df_all["dataset"].unique().tolist())


# ── Sidebar ───────────────────────────────────────────────────────────────────

PAGES = [
    ("Overview",  "◈"),
    ("Toxicity",  "◎"),
    ("Emotions",  "◉"),
    ("Othering",  "◆"),
    ("Upload",    "▲"),
]

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
      <span class="we">We</span>
      <span class="vs">vs</span>
      <span class="them">Them</span>
    </div>
    """, unsafe_allow_html=True)

    if "page" not in st.session_state:
        st.session_state["page"] = "Overview"

    for pname, picon in PAGES:
        active = st.session_state["page"] == pname
        cls = "nav-pill active" if active else "nav-pill"
        if st.button(f"{picon}  {pname}", key=f"nav_{pname}", use_container_width=True):
            st.session_state["page"] = pname
            st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Dataset selector
    if st.session_state["page"] != "Upload":
        st.markdown('<div class="section-header">Datasets</div>', unsafe_allow_html=True)

        _b1, _b2 = st.columns(2)
        if _b1.button("All", use_container_width=True, key="_ds_btn_all"):
            for _lbl in all_dataset_labels:
                st.session_state[f"_ds_{_lbl}"] = True
            st.rerun()
        if _b2.button("None", use_container_width=True, key="_ds_btn_none"):
            for _lbl in all_dataset_labels:
                st.session_state[f"_ds_{_lbl}"] = False
            st.rerun()

        st.markdown(" ")
        _ds_counts = df_all["dataset"].value_counts().to_dict() if not df_all.empty else {}
        _groups    = discover_datasets()
        selected_datasets = []

        for _group, _files in _groups.items():
            _emoji = PLATFORM_EMOJI.get(_group, "◈")
            _labels = [
                f"{_group}  ·  {_name}"
                for _name, _ in _files
                if f"{_group}  ·  {_name}" in set(all_dataset_labels)
            ]
            if not _labels:
                continue
            st.markdown(f'<div class="section-header" style="font-size:10px;">{_emoji} {_group.upper()}</div>',
                        unsafe_allow_html=True)
            for _lbl in _labels:
                _count = _ds_counts.get(_lbl, 0)
                _short = re.sub(r"\.(csv|xlsx)$", "", _lbl.split("  ·  ")[-1])
                _display = f"{_short}" + (f"  ·  {_count:,}" if _count else "")
                if st.checkbox(_display, value=True, key=f"_ds_{_lbl}"):
                    selected_datasets.append(_lbl)
            st.markdown(" ")

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Search & filter
        st.markdown('<div class="section-header">Filters</div>', unsafe_allow_html=True)
        keyword = st.text_input("", placeholder="Search keyword...", label_visibility="collapsed")

        _df_for_subs = df_all[df_all["dataset"].isin(selected_datasets)] if selected_datasets else df_all
        known_subreddits = sorted(
            s for s in _df_for_subs["subreddit"].unique()
            if s not in ("unknown", "none", "") and pd.notna(s)
        )
        selected_subreddits = st.multiselect(
            "", options=known_subreddits, default=[], placeholder="Subreddit / account",
            label_visibility="collapsed",
        )

page = st.session_state["page"]


# ── Upload page ───────────────────────────────────────────────────────────────

if page == "Upload":
    st.markdown('<div class="page-title">Upload & Analyze</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Drop any CSV — text column detected automatically</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded, low_memory=False)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        st.markdown(f"""
        <div class="post-card">
          <strong>{uploaded.name}</strong>
          <div class="meta">
            <span>{len(raw):,} rows</span>
            <span>{len(raw.columns)} columns</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        detected_col, confidence = detect_text_column(raw)
        str_cols = raw.select_dtypes(include="object").columns.tolist()

        if confidence == "not_found":
            st.error("No text column found. Select one manually.")
            text_col = st.selectbox("Column to use as text", options=raw.columns.tolist())
        elif confidence == "ambiguous":
            st.warning(f"Multiple text-like columns. Best guess: `{detected_col}`")
            text_col = st.selectbox("Column to use as text", options=str_cols,
                                    index=str_cols.index(detected_col) if detected_col in str_cols else 0)
        else:
            lbl = {"exact": "exact match", "case-insensitive": "case-insensitive",
                   "inferred": "inferred from content"}[confidence]
            st.info(f"Text column: **`{detected_col}`** ({lbl})")
            text_col = st.selectbox("Column", options=str_cols,
                                    index=str_cols.index(detected_col) if detected_col in str_cols else 0,
                                    label_visibility="collapsed")

        if text_col != "text":
            raw = raw.rename(columns={text_col: "text"})

        with st.expander("Preview (5 rows)"):
            st.dataframe(raw.head(), use_container_width=True)

        already_processed = all(c in raw.columns for c in ["has_othering", "othering_predicted", "pronoun_type"])
        if already_processed:
            st.info("File already contains analysis columns.")

        col1, col2 = st.columns(2)
        with col1:
            max_rows = st.slider("Max rows", min_value=100, max_value=min(50_000, len(raw)),
                                 value=min(5_000, len(raw)), step=100, disabled=already_processed)
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

                st.session_state["uploaded_df"] = result
                st.session_state["upload_name"] = uploaded.name
                udf = result.copy()
                udf["dataset"] = f"📎  {uploaded.name}"
                existing = st.session_state["df_combined"]
                existing = existing[existing["dataset"] != udf["dataset"].iloc[0]]
                st.session_state["df_combined"] = pd.concat([udf, existing], ignore_index=True)
                st.success(f"Done — {len(result):,} posts added. Navigate to any page to explore.")
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline error: {e}")

        if "uploaded_df" in st.session_state and st.session_state.get("upload_name") == uploaded.name:
            result = st.session_state["uploaded_df"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{len(result):,}")
            c2.metric("% Othering", f"{result['othering_predicted'].mean()*100:.1f}%")
            tox = result["toxicity"].mean() if result["toxicity"].notna().any() else None
            c3.metric("Avg toxicity", f"{tox:.3f}" if tox else "n/a")
            c4.metric("% Them markers",
                      f"{result['pronoun_type'].isin(['them_only','both']).mean()*100:.1f}%")
            with st.expander("Download result"):
                st.download_button("Download result_analyzed.csv",
                                   data=result.to_csv(index=False).encode("utf-8"),
                                   file_name="result_analyzed.csv", mime="text/csv")

    st.markdown(" ")
    if st.button("Reload all datasets"):
        del st.session_state["df_combined"]
        st.rerun()

    st.stop()


# ── Gate: dataset selection ───────────────────────────────────────────────────

if not selected_datasets:
    st.warning("Select at least one dataset from the sidebar.")
    st.stop()

df_full = df_all[df_all["dataset"].isin(selected_datasets)].copy()

df = df_full.copy()
if keyword.strip():
    df = df[df["text"].fillna("").str.contains(keyword.strip(), case=False, na=False)]
if selected_subreddits:
    df = df[df["subreddit"].isin(selected_subreddits)]

if len(df) == 0:
    st.warning("No results for the current filters.")
    st.stop()

has_topics = "topic" in df_full.columns and "topic_name" in df_full.columns


# ── Overview ──────────────────────────────────────────────────────────────────

if page == "Overview":
    st.markdown('<div class="page-title">Overview</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{len(df):,} posts across {df["dataset"].nunique()} dataset(s)</div>',
                unsafe_allow_html=True)

    # KPI row
    total     = len(df)
    pct_oth   = df["othering_predicted"].mean() * 100
    tox_ok    = df["toxicity"].notna().any()
    avg_tox   = df["toxicity"].mean() if tox_ok else None
    pct_toxic = (df["toxicity"].dropna() > 0.5).mean() * 100 if tox_ok else None
    top_emo   = df["emotion"].mode()[0] if df["emotion"].notna().any() else "n/a"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total posts",     f"{total:,}")
    c2.metric("Othering rate",   f"{pct_oth:.1f}%")
    c3.metric("High toxicity",   f"{pct_toxic:.1f}%" if pct_toxic is not None else "n/a")
    c4.metric("Top emotion",     top_emo)

    st.markdown(" ")

    # Posts by dataset — horizontal bar with pastel colors and ticktext labels
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown('<div class="section-header">Posts by dataset</div>', unsafe_allow_html=True)
        ds_counts = df["dataset"].value_counts().reset_index()
        ds_counts.columns = ["dataset", "count"]
        ds_counts["pct"] = (ds_counts["count"] / ds_counts["count"].sum() * 100).round(1)
        ds_counts["short"] = ds_counts["dataset"].apply(
            lambda x: re.sub(r"\.(csv|xlsx)$", "", x.split("  ·  ")[-1]) if "  ·  " in x else x)
        ds_counts["oth_rate"] = ds_counts["dataset"].map(
            df.groupby("dataset")["othering_predicted"].mean().mul(100).round(1)
        ).fillna(0)
        ds_counts = ds_counts.sort_values("count", ascending=True)
        _PASTEL = ["#c4b5fd","#fda4af","#7dd3fc","#6ee7b7","#fcd34d","#f9a8d4","#67e8f9","#bef264"]
        ds_counts = ds_counts.reset_index(drop=True)
        _colors   = [_PASTEL[i % len(_PASTEL)] for i in range(len(ds_counts))]
        _ticktext = [f"{r['short']}  ·  {r['count']:,}" for _, r in ds_counts.iterrows()]
        fig = go.Figure(go.Bar(
            y=ds_counts["dataset"], x=ds_counts["count"],
            orientation="h",
            marker_color=_colors,
            marker_line_width=0,
            customdata=list(zip(ds_counts["pct"], ds_counts["oth_rate"])),
            hovertemplate="%{x:,} posts · %{customdata[0]:.1f}% of corpus<br>othering: %{customdata[1]:.1f}%<extra></extra>",
        ))
        apply_theme(fig, height=max(200, len(ds_counts) * 44 + 60))
        fig.update_layout(
            xaxis=dict(showticklabels=False),
            yaxis=dict(tickvals=ds_counts["dataset"].tolist(), ticktext=_ticktext),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Pronoun type vs othering</div>', unsafe_allow_html=True)
        pron_agg = (df[~df["pronoun_type"].isin(["unknown", "none", ""])]
                    .groupby("pronoun_type")
                    .agg(count=("text", "count"), oth_rate=("othering_predicted", "mean"))
                    .reset_index())
        if pron_agg.empty:
            st.info("No pronoun data.")
        else:
            pron_agg["pct_posts"] = (pron_agg["count"] / pron_agg["count"].sum() * 100).round(1)
            pron_agg["oth_pct"]   = (pron_agg["oth_rate"] * 100).round(1)
            pron_agg = pron_agg.sort_values("pct_posts", ascending=False)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                name="% of posts", x=pron_agg["pronoun_type"], y=pron_agg["pct_posts"],
                marker_color=hex_rgba(PALETTE[2], 0.7),
                marker_line_color=PALETTE[2], marker_line_width=1,
                text=(pron_agg["pct_posts"].round(1).astype(str) + "%").tolist(),
                textposition="outside",
                textfont=dict(family="IBM Plex Mono, monospace", size=10),
            ))
            fig2.add_trace(go.Bar(
                name="othering rate", x=pron_agg["pronoun_type"], y=pron_agg["oth_pct"],
                marker_color=hex_rgba(PALETTE[1], 0.7),
                marker_line_color=PALETTE[1], marker_line_width=1,
                text=(pron_agg["oth_pct"].round(1).astype(str) + "%").tolist(),
                textposition="outside",
                textfont=dict(family="IBM Plex Mono, monospace", size=10),
            ))
            apply_theme(fig2, height=260)
            fig2.update_layout(
                barmode="group",
                yaxis=dict(title="%", gridcolor="#1e1e2e", zeroline=False),
                legend=dict(orientation="h", y=1.14, x=0.5, xanchor="center"),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Othering rate by dataset — lollipop style
    st.markdown('<div class="section-header">Othering rate by dataset</div>', unsafe_allow_html=True)
    oth = (df.groupby("dataset")["othering_predicted"].agg(["sum", "count"]).reset_index())
    oth.columns = ["dataset", "othering", "total"]
    oth["pct"] = oth["othering"] / oth["total"] * 100
    oth["short"] = oth["dataset"].apply(
        lambda x: re.sub(r"\.(csv|xlsx)$", "", x.split("  ·  ")[-1]) if "  ·  " in x else x)
    oth = oth.sort_values("pct", ascending=True)

    fig3 = go.Figure()
    for i, row in oth.iterrows():
        fig3.add_trace(go.Scatter(
            x=[0, row["pct"]], y=[row["short"], row["short"]],
            mode="lines",
            line=dict(color="#2a2a3a", width=2),
            showlegend=False,
        ))
    global_avg = oth["othering"].sum() / oth["total"].sum() * 100
    fig3.add_trace(go.Scatter(
        x=oth["pct"], y=oth["short"],
        mode="markers+text",
        marker=dict(size=12, color=PALETTE[1], line=dict(color="#0c0c10", width=2)),
        text=[f"{p:.1f}%  n={n:,}" for p, n in zip(oth["pct"].fillna(0), oth["total"])],
        textposition="middle right",
        textfont=dict(family="IBM Plex Mono, monospace", size=11, color="#e8e8f0"),
        showlegend=False,
    ))
    fig3.add_shape(type="line",
                   x0=global_avg, x1=global_avg, y0=-0.5, y1=len(oth) - 0.5,
                   line=dict(color="#7070a0", width=1, dash="dot"))
    fig3.add_annotation(x=global_avg, y=len(oth) - 0.5,
                        text=f"avg {global_avg:.1f}%",
                        showarrow=False, yanchor="bottom", xanchor="left",
                        font=dict(family="IBM Plex Mono,monospace", size=10, color="#7070a0"))
    apply_theme(fig3, height=max(200, len(oth) * 36 + 60))
    pct_max = oth["pct"].max() if oth["pct"].notna().any() else 10
    fig3.update_layout(xaxis_title="% othering", yaxis_title="",
                       xaxis=dict(range=[0, max(pct_max * 1.65, 10)]))
    st.plotly_chart(fig3, use_container_width=True)


# ── Toxicity ──────────────────────────────────────────────────────────────────

elif page == "Toxicity":
    st.markdown('<div class="page-title">Toxicity</div>', unsafe_allow_html=True)

    df_tox = df.dropna(subset=["toxicity"])
    if df_tox.empty:
        st.info("No toxicity data. Run Detoxify via Upload, or select a default dataset.")
        st.stop()

    st.markdown(f'<div class="page-subtitle">{len(df_tox):,} posts with toxicity scores</div>',
                unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg toxicity",    f"{df_tox['toxicity'].mean():.3f}")
    c2.metric("Median toxicity", f"{df_tox['toxicity'].median():.3f}")
    c3.metric("High (> 0.5)",    f"{(df_tox['toxicity'] > 0.5).mean()*100:.1f}%")
    c4.metric("Max score",       f"{df_tox['toxicity'].max():.3f}")

    st.markdown(" ")
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="section-header">Distribution by dataset</div>', unsafe_allow_html=True)
        datasets_in_tox = df_tox["dataset"].unique()
        short_map = {d: re.sub(r"\.(csv|xlsx)$", "", d.split("  ·  ")[-1]) if "  ·  " in d else d
                     for d in datasets_in_tox}
        fig = go.Figure()
        for i, ds in enumerate(datasets_in_tox):
            vals = np.sort(df_tox[df_tox["dataset"] == ds]["toxicity"].dropna().values)
            if len(vals) == 0:
                continue
            ecdf_y = np.arange(1, len(vals) + 1) / len(vals)
            fig.add_trace(go.Scatter(
                x=vals, y=ecdf_y,
                mode="lines",
                name=short_map[ds],
                line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            ))
        apply_theme(fig, height=300)
        fig.update_layout(
            xaxis_title="Toxicity score",
            yaxis=dict(title="Cumulative %", tickformat=".0%",
                       gridcolor="#1e1e2e", zeroline=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Toxicity vs othering</div>', unsafe_allow_html=True)
        box = df_tox.copy()
        box["Group"] = box["othering_predicted"].map({1: "Othering", 0: "Non-othering"})
        fig2 = go.Figure()
        for i, grp in enumerate(["Othering", "Non-othering"]):
            vals = box[box["Group"] == grp]["toxicity"].dropna()
            fig2.add_trace(go.Histogram(
                x=vals, name=grp,
                histnorm="probability density",
                nbinsx=60,
                marker_color=hex_rgba(PALETTE[i], 0.55),
                marker_line_color=PALETTE[i], marker_line_width=1,
                opacity=0.75,
            ))
        apply_theme(fig2, height=300)
        fig2.update_layout(
            barmode="overlay",
            xaxis_title="Toxicity score",
            yaxis=dict(title="Density", gridcolor="#1e1e2e", zeroline=False),
            legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Toxicity × othering scatter
    st.markdown('<div class="section-header">Toxicity × othering score</div>', unsafe_allow_html=True)
    _sdf = df_tox.copy()
    _sdf["othering_score"] = pd.to_numeric(_sdf["othering_score"], errors="coerce").fillna(0)
    if len(_sdf) > 4000:
        _sdf = _sdf.sample(4000, random_state=42)
    _rng = np.random.default_rng(42)
    _sdf["_jitter"] = _sdf["othering_score"] + _rng.uniform(-0.18, 0.18, len(_sdf))
    _sdf["_color"]  = _sdf["othering_predicted"].map({1: PALETTE[1], 0: PALETTE[0]})
    fig_s = go.Figure(go.Scatter(
        x=_sdf["toxicity"], y=_sdf["_jitter"],
        mode="markers",
        marker=dict(size=4, color=_sdf["_color"].tolist(), opacity=0.35),
        hovertemplate="tox: %{x:.3f}<br>score: %{y:.1f}<extra></extra>",
        showlegend=False,
    ))
    apply_theme(fig_s, height=260)
    fig_s.update_layout(
        xaxis_title="Toxicity score",
        yaxis=dict(title="Othering score (0–4)", tickvals=[0, 1, 2, 3, 4], zeroline=False),
    )
    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#7070a0;margin-bottom:6px;">violet = othering · blue = non-othering</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_s, use_container_width=True)

    # Top toxic posts
    st.markdown('<div class="section-header">Most toxic posts</div>', unsafe_allow_html=True)
    top_n = st.slider("Show top N", 5, 50, 10, key="tox_top_n")
    top_posts = df_tox.nlargest(top_n, "toxicity")

    for _, row in top_posts.iterrows():
        tox_val  = row["toxicity"]
        oth_val  = row.get("othering_predicted", 0)
        raw_text = str(row.get("text", ""))
        ds_short = re.sub(r"\.(csv|xlsx)$", "", row["dataset"].split("  ·  ")[-1]) if "  ·  " in row["dataset"] else row["dataset"]
        tox_cls  = "badge-red" if tox_val > 0.7 else "badge-purple" if tox_val > 0.4 else "badge-gray"
        text_safe = _html.escape(raw_text[:280]) + ("…" if len(raw_text) > 280 else "")
        badges = f'<span class="badge {tox_cls}">tox {tox_val:.3f}</span>'
        if oth_val:
            badges += ' <span class="badge badge-purple">othering</span>'
        badges += f' <span class="badge badge-gray">{_html.escape(ds_short)}</span>'
        st.markdown(
            f'<div class="post-card toxic"><span style="font-size:13px;">{text_safe}</span>'
            f'<div class="meta">{badges}</div></div>',
            unsafe_allow_html=True,
        )


# ── Emotions ──────────────────────────────────────────────────────────────────

elif page == "Emotions":
    st.markdown('<div class="page-title">Emotions</div>', unsafe_allow_html=True)
    df_emo = df.dropna(subset=["emotion"])

    if df_emo.empty:
        st.info("No emotion data. Run GoEmotions via Upload, or select a default dataset.")
        st.stop()

    pct_cov = len(df_emo) / len(df) * 100
    st.markdown(f'<div class="page-subtitle">{len(df_emo):,} posts with emotion labels ({pct_cov:.1f}% coverage)</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 3])

    with col_a:
        st.markdown('<div class="section-header">Overall distribution</div>', unsafe_allow_html=True)
        emo_counts = df_emo["emotion"].value_counts().reset_index()
        emo_counts.columns = ["emotion", "count"]
        emo_counts["pct"] = (emo_counts["count"] / emo_counts["count"].sum() * 100).round(1)
        emo_counts = emo_counts.sort_values("count", ascending=True)
        emo_counts["valence"] = emo_counts["emotion"].map(EMOTION_VALENCE).fillna("neutral")
        fig = go.Figure(go.Bar(
            y=emo_counts["emotion"], x=emo_counts["count"],
            orientation="h",
            marker_color=emo_counts["valence"].map(VALENCE_COLOR).tolist(),
            text=(emo_counts["pct"].fillna(0).round(1).astype(str) + "%").tolist(),
            textposition="outside",
            textfont=dict(family="IBM Plex Mono, monospace", size=10, color="#7070a0"),
        ))
        apply_theme(fig, height=max(300, len(emo_counts) * 24 + 60))
        fig.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Δ% othering − non-othering</div>', unsafe_allow_html=True)
        split = df_emo.copy()
        split["group"] = split["othering_predicted"].map({1: "Othering", 0: "Non-othering"})
        es = split.groupby(["group", "emotion"]).size().reset_index(name="n")
        es["pct"] = es.groupby("group")["n"].transform(lambda x: x / x.sum() * 100).round(2)
        epivot = es.pivot(index="emotion", columns="group", values="pct").fillna(0)
        for _col in ["Othering", "Non-othering"]:
            if _col not in epivot.columns:
                epivot[_col] = 0.0
        epivot["delta"] = (epivot["Othering"] - epivot["Non-othering"]).round(2)
        epivot = epivot.sort_values("delta", ascending=True)
        _div_colors = [PALETTE[1] if d > 0 else PALETTE[2] for d in epivot["delta"]]
        fig2 = go.Figure(go.Bar(
            y=epivot.index.tolist(),
            x=epivot["delta"].tolist(),
            orientation="h",
            marker_color=_div_colors,
            marker_line_width=0,
            text=[f"+{d:.1f}%" if d > 0 else f"{d:.1f}%" for d in epivot["delta"]],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono, monospace", size=10, color="#7070a0"),
        ))
        apply_theme(fig2, height=max(300, len(epivot) * 24 + 80))
        fig2.update_layout(
            xaxis=dict(zeroline=True, zerolinecolor="#3a3a5a", zerolinewidth=2, title="Δ%"),
            yaxis_title="",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Emotion by dataset — heatmap
    st.markdown('<div class="section-header">Emotion by dataset</div>', unsafe_allow_html=True)
    emo_ds = df_emo.groupby(["dataset", "emotion"]).size().reset_index(name="count")
    emo_ds["pct"] = emo_ds.groupby("dataset")["count"].transform(lambda x: x / x.sum() * 100).round(1)
    emo_ds["short"] = emo_ds["dataset"].apply(
        lambda x: re.sub(r"\.(csv|xlsx)$", "", x.split("  ·  ")[-1]) if "  ·  " in x else x)
    _top_emos = emo_ds.groupby("emotion")["count"].sum().nlargest(15).index.tolist()
    emo_ds_top = emo_ds[emo_ds["emotion"].isin(_top_emos)]
    emo_heat = emo_ds_top.pivot_table(index="emotion", columns="short", values="pct", aggfunc="mean").fillna(0)
    _emo_order = [e for e in _top_emos if e in emo_heat.index]
    emo_heat = emo_heat.reindex(_emo_order[::-1])

    fig3 = go.Figure(go.Heatmap(
        z=emo_heat.values,
        x=emo_heat.columns.tolist(),
        y=emo_heat.index.tolist(),
        colorscale=[[0, "#13131a"], [0.5, "rgba(124,58,237,0.4)"], [1, "#7c3aed"]],
        texttemplate="%{z:.1f}%",
        textfont=dict(family="IBM Plex Mono, monospace", size=9),
        showscale=False,
        hoverongaps=False,
    ))
    apply_theme(fig3, height=max(320, len(emo_heat) * 28 + 80))
    fig3.update_layout(xaxis_title="", yaxis_title="")
    st.plotly_chart(fig3, use_container_width=True)


# ── Othering ──────────────────────────────────────────────────────────────────

elif page == "Othering":
    st.markdown('<div class="page-title">Othering</div>', unsafe_allow_html=True)

    total_oth = df["othering_predicted"].sum()
    pct_oth   = df["othering_predicted"].mean() * 100
    st.markdown(f'<div class="page-subtitle">{total_oth:,} othering posts detected ({pct_oth:.1f}%)</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Othering posts", f"{total_oth:,}")
    c2.metric("Othering rate",  f"{pct_oth:.1f}%")
    score_mean = df["othering_score"].mean()
    c3.metric("Avg score",      f"{score_mean:.2f} / 4")
    both_pct = (df["pronoun_type"] == "both").mean() * 100
    c4.metric("Both we + them", f"{both_pct:.1f}%")

    st.markdown(" ")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Rate by pronoun type</div>', unsafe_allow_html=True)
        oth_pron = (df[~df["pronoun_type"].isin(["unknown", "none", ""])]
                    .groupby("pronoun_type")["othering_predicted"]
                    .agg(["sum", "count"]).reset_index())
        oth_pron.columns = ["pronoun_type", "othering", "total"]
        oth_pron["pct"] = oth_pron["othering"] / oth_pron["total"] * 100
        oth_pron = oth_pron.sort_values("pct", ascending=False)
        fig = go.Figure(go.Bar(
            x=oth_pron["pronoun_type"], y=oth_pron["pct"],
            marker=dict(
                color=PALETTE[:len(oth_pron)],
                line=dict(color="#0c0c10", width=1),
            ),
            text=(oth_pron["pct"].fillna(0).round(1).astype(str) + "%").tolist(),
            textposition="outside",
            textfont=dict(family="IBM Plex Mono, monospace", size=11, color="#e8e8f0"),
        ))
        apply_theme(fig, height=280)
        fig.update_layout(xaxis_title="", yaxis_title="% othering", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Pronoun × dataset heatmap</div>', unsafe_allow_html=True)
        valid_prons = ["we_only", "them_only", "both"]
        heat_df = (df[df["pronoun_type"].isin(valid_prons)]
                   .groupby(["dataset", "pronoun_type"])["othering_predicted"]
                   .mean().mul(100).round(1).reset_index())
        heat_df["short"] = heat_df["dataset"].apply(
            lambda x: re.sub(r"\.(csv|xlsx)$", "", x.split("  ·  ")[-1]) if "  ·  " in x else x)
        heat_pivot = heat_df.pivot_table(index="short", columns="pronoun_type", values="othering_predicted", aggfunc="mean").fillna(0)
        _sort_col = "both" if "both" in heat_pivot.columns else heat_pivot.columns[0]
        heat_pivot = heat_pivot.sort_values(_sort_col, ascending=False)
        fig2 = go.Figure(go.Heatmap(
            z=heat_pivot.values,
            x=heat_pivot.columns.tolist(),
            y=heat_pivot.index.tolist(),
            colorscale=[[0, "#13131a"], [0.5, "rgba(124,58,237,0.33)"], [1, "#e11d48"]],
            texttemplate="%{z:.1f}%",
            textfont=dict(family="IBM Plex Mono, monospace", size=11),
            showscale=False,
        ))
        apply_theme(fig2, height=max(240, len(heat_pivot) * 36 + 80))
        fig2.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    # Pattern families — grouped by dataset
    st.markdown('<div class="section-header">Pattern families by dataset</div>', unsafe_allow_html=True)
    _FAMILY_ORDER  = ["dehumanizing", "moral_exclusion", "generalization", "threat_framing"]
    _FAMILY_COLORS = dict(zip(_FAMILY_ORDER, [PALETTE[1], PALETTE[0], PALETTE[2], PALETTE[3]]))
    _fam_rows = []
    try:
        from othering import ALL_FAMILIES as _AF
        _p2f = {lbl: fam for fam, pats in _AF.items() for lbl, _ in pats}
        _oth_posts = df[df["othering_predicted"] == 1].copy()
        if "matched_patterns" in _oth_posts.columns and not _oth_posts.empty:
            def _extract_fams(val):
                if isinstance(val, list): pats = val
                elif isinstance(val, str):
                    try: pats = _ast.literal_eval(val)
                    except: pats = []
                else: pats = []
                return {_p2f[p] for p in pats if p in _p2f}
            _oth_posts["_fams"] = _oth_posts["matched_patterns"].apply(_extract_fams)
            _oth_posts["short"] = _oth_posts["dataset"].apply(
                lambda x: re.sub(r"\.(csv|xlsx)$", "", x.split("  ·  ")[-1]) if "  ·  " in x else x)
            _ds_totals = _oth_posts.groupby("short").size()
            for _fam in _FAMILY_ORDER:
                _fam_hits = _oth_posts[_oth_posts["_fams"].apply(lambda f: _fam in f)].groupby("short").size()
                for _ds, _cnt in _fam_hits.items():
                    _fam_rows.append({"dataset": _ds, "family": _fam,
                                      "pct": round(_cnt / _ds_totals.get(_ds, 1) * 100, 1)})
    except Exception:
        pass

    if _fam_rows:
        _fbd = pd.DataFrame(_fam_rows)
        fig3 = go.Figure()
        for _fam in _FAMILY_ORDER:
            _sub = _fbd[_fbd["family"] == _fam]
            fig3.add_trace(go.Bar(
                name=_fam.replace("_", " "),
                x=_sub["dataset"], y=_sub["pct"],
                marker_color=_FAMILY_COLORS[_fam],
                marker_line_width=0,
            ))
        apply_theme(fig3, height=300)
        fig3.update_layout(
            barmode="group", xaxis_title="", yaxis_title="% of othering posts",
            legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No matched_patterns data available.")

    # Pattern co-occurrence
    st.markdown('<div class="section-header">Pattern co-occurrence</div>', unsafe_allow_html=True)
    try:
        from othering import ALL_FAMILIES as _AF_co
        _all_labels = [lbl for _fam, pats in _AF_co.items() for lbl, _ in pats]
        _oth_df = df[df["othering_predicted"] == 1].dropna(subset=["matched_patterns"])
        _co = pd.DataFrame(0, index=_all_labels, columns=_all_labels, dtype=int)
        for _val in _oth_df["matched_patterns"]:
            if isinstance(_val, list): _pats = _val
            elif isinstance(_val, str):
                try: _pats = _ast.literal_eval(_val)
                except: _pats = []
            else: _pats = []
            _present = [p for p in _pats if p in _all_labels]
            for _a in _present:
                for _b in _present:
                    if _a != _b:
                        _co.loc[_a, _b] += 1
        _mask = _co.sum(axis=1) > 0
        _co = _co.loc[_mask, _mask]
        if not _co.empty:
            fig_co = go.Figure(go.Heatmap(
                z=_co.values,
                x=_co.columns.tolist(),
                y=_co.index.tolist(),
                colorscale=[[0, "#13131a"], [0.3, "rgba(14,165,233,0.3)"], [1, "#0ea5e9"]],
                texttemplate="%{z}",
                textfont=dict(family="IBM Plex Mono, monospace", size=9),
                showscale=False,
                hoverongaps=False,
            ))
            apply_theme(fig_co, height=max(320, len(_co) * 26 + 80))
            fig_co.update_layout(xaxis=dict(tickangle=-45), yaxis_title="")
            st.plotly_chart(fig_co, use_container_width=True)
    except Exception:
        pass

    # Post examples as cards
    st.markdown('<div class="section-header">Post examples</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        smin = float(df["othering_score"].min()) if not df["othering_score"].isna().all() else 0.0
        smax = float(df["othering_score"].max()) if not df["othering_score"].isna().all() else 4.0
        score_range = st.slider("Score range", min_value=smin, max_value=smax,
                                value=(smin, smax), step=1.0)
    with c2:
        othering_only = st.checkbox("Othering only", value=True)
    with c3:
        n_cards = st.slider("Cards to show", 5, 40, 10)

    ex = df.dropna(subset=["othering_score"])
    ex = ex[(ex["othering_score"] >= score_range[0]) & (ex["othering_score"] <= score_range[1])]
    if othering_only:
        ex = ex[ex["othering_predicted"] == 1]

    st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#7070a0;margin-bottom:12px;">{len(ex):,} posts match</div>',
                unsafe_allow_html=True)

    for _, row in ex.head(n_cards).iterrows():
        score    = int(row.get("othering_score", 0))
        tox_v    = row.get("toxicity", None)
        raw_text = str(row.get("text", ""))
        ds_s     = re.sub(r"\.(csv|xlsx)$", "", row["dataset"].split("  ·  ")[-1]) if "  ·  " in row["dataset"] else row["dataset"]
        patterns_raw = row.get("matched_patterns", "[]")
        try:
            patterns = eval(patterns_raw) if isinstance(patterns_raw, str) else patterns_raw
        except Exception:
            patterns = []
        score_colors = ["badge-gray", "badge-blue", "badge-purple", "badge-red", "badge-red"]
        text_safe = _html.escape(raw_text[:300]) + ("…" if len(raw_text) > 300 else "")
        badges = f'<span class="badge {score_colors[min(score,4)]}">{score}/4</span>'
        if pd.notna(tox_v):
            badges += f' <span class="badge badge-red">tox {tox_v:.3f}</span>'
        for p in (patterns or [])[:4]:
            badges += f' <span class="badge badge-purple">{_html.escape(str(p))}</span>'
        badges += f' <span class="badge badge-gray">{_html.escape(ds_s)}</span>'
        st.markdown(
            f'<div class="post-card"><span style="font-size:13px;line-height:1.6;">{text_safe}</span>'
            f'<div class="meta">{badges}</div></div>',
            unsafe_allow_html=True,
        )

    if has_topics and "topic" in df.columns and "topic_name" in df.columns:
        st.markdown('<div class="section-header">BERTopic — top 20 topics</div>', unsafe_allow_html=True)
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
        tstats["label"] = tstats["topic"].fillna(0).astype(int).astype(str) + "  " + tstats["topic_name"].fillna("").astype(str)
        top20 = tstats.nlargest(20, "nb_posts").sort_values("nb_posts", ascending=True)
        fig_t = go.Figure(go.Bar(
            y=top20["label"], x=top20["nb_posts"],
            orientation="h",
            marker=dict(
                color=top20["pct_othering"],
                colorscale=[[0, "#2a2a3a"], [1, "#e11d48"]],
                showscale=True,
                colorbar=dict(title="% othering", tickfont=dict(family="IBM Plex Mono,monospace", size=10)),
            ),
            text=top20["nb_posts"],
            textposition="outside",
        ))
        apply_theme(fig_t, height=600)
        fig_t.update_layout(yaxis_title="", xaxis_title="Posts")
        st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("No BERTopic data in the current selection.")
