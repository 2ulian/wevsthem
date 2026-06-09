import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import io
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
from classifier import load_model

import json as _json

# Load trained classifier once at startup (None if not found)
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
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 14px;
  border-radius: 8px;
  font-family: 'Syne', sans-serif;
  font-size: 13px;
  font-weight: 600;
  border: 1px solid transparent;
  width: 100%;
  margin-bottom: 4px;
  box-sizing: border-box;
}
.nav-pill.active {
  background: rgba(124,58,237,0.18);
  border-color: rgba(124,58,237,0.35);
  border-left: 3px solid var(--accent);
  color: #c4b5fd !important;
  cursor: default;
  padding-left: 11px;
}

/* Sidebar nav buttons */
[data-testid="stSidebar"] .stButton button {
  font-family: 'Syne', sans-serif !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  color: var(--muted) !important;
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 8px 14px !important;
  width: 100% !important;
  text-align: left !important;
  margin-bottom: 4px !important;
  transition: all 0.15s !important;
}
[data-testid="stSidebar"] .stButton button:hover {
  background: rgba(124,58,237,0.08) !important;
  border-color: var(--accent) !important;
  color: var(--text) !important;
}
/* All / None buttons inside the dataset expander */
[data-testid="stSidebar"] details .stButton button {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-size: 11px !important;
  padding: 3px 10px !important;
  border-radius: 4px !important;
  width: 100% !important;
  margin-bottom: 2px !important;
}
[data-testid="stSidebar"] details .stButton button:hover {
  border-color: var(--accent) !important;
  color: #c4b5fd !important;
  background: rgba(124,58,237,0.08) !important;
}

/* Active nav button (Streamlit 1.39+ : data-testid + class on the <button> element) */
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"],
[data-testid="stSidebar"] button.stBaseButton-primary {
  color: #c4b5fd !important;
  background: rgba(124,58,237,0.18) !important;
  border: 1px solid rgba(124,58,237,0.35) !important;
  border-left: 3px solid var(--accent) !important;
  padding-left: 11px !important;
}
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]:hover,
[data-testid="stSidebar"] button.stBaseButton-primary:hover {
  color: #ede9fe !important;
  background: rgba(124,58,237,0.28) !important;
  border-color: rgba(124,58,237,0.55) !important;
  border-left: 3px solid var(--accent) !important;
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

/* Analysis option cards */
.ac-card {
  border-radius: 8px;
  padding: 10px 8px 10px 8px;
  margin-bottom: 8px;
  text-align: center;
  transition: background 0.2s;
  position: relative;
}
.ac-info-btn {
  position: absolute;
  top: 7px;
  right: 7px;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  border: 1px solid rgba(112,112,160,0.35);
  background: rgba(112,112,160,0.08);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-family: 'IBM Plex Mono', monospace;
  color: #70709f;
  cursor: pointer;
  user-select: none;
  transition: border-color 0.15s, background 0.15s, color 0.15s;
  line-height: 1;
}
.ac-info-btn:hover {
  border-color: #7c3aed;
  background: rgba(124,58,237,0.15);
  color: #c4b5fd;
}
.ac-info-toggle { display: none !important; }
.ac-info-popup {
  display: none;
  position: fixed;
  inset: 0;
  z-index: 9999;
  background: rgba(0,0,0,0.55);
  align-items: center;
  justify-content: center;
}
.ac-info-toggle:checked ~ .ac-info-popup { display: flex; }
.ac-info-backdrop {
  position: absolute;
  inset: 0;
  cursor: pointer;
}
.ac-info-box {
  position: relative;
  background: #16161f;
  border: 1px solid #2a2a3a;
  border-radius: 12px;
  padding: 22px 26px 18px;
  max-width: 420px;
  width: 90%;
  color: #e2e2f0;
  font-size: 12px;
  line-height: 1.65;
  z-index: 1;
}
.ac-info-box h3 {
  font-family: 'Syne', sans-serif;
  font-size: 13px;
  font-weight: 700;
  color: #c4b5fd;
  margin: 0 0 10px;
}
.ac-info-box ul { padding-left: 16px; margin: 6px 0; }
.ac-info-box li { margin-bottom: 3px; }
.ac-info-box p { margin: 6px 0; }
.ac-info-box b { color: #c4b5fd; }
.ac-info-box em { color: #a0a0c0; }
.ac-info-box code { background: rgba(124,58,237,0.12); padding: 1px 4px; border-radius: 3px; font-size: 11px; }
.ac-info-close {
  position: absolute;
  top: 10px; right: 12px;
  cursor: pointer;
  color: #70709f;
  font-size: 12px;
  width: 22px; height: 22px;
  display: flex; align-items: center; justify-content: center;
  border-radius: 50%;
  border: 1px solid rgba(112,112,160,0.2);
  transition: color 0.15s, border-color 0.15s;
}
.ac-info-close:hover { color: #e2e2f0; border-color: rgba(112,112,160,0.5); }
.ac-icon  { font-size: 20px; margin-bottom: 6px; }
.ac-name  {
  font-family: 'Syne', sans-serif;
  font-weight: 700;
  font-size: 12px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 6px;
}
.ac-badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 20px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.04em;
  margin-bottom: 2px;
}

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
  background: var(--surface) !important;
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
  background: rgba(124,58,237,0.08) !important;
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
/* Run analysis button — red */
[data-baseweb="tab-panel"] [data-testid="baseButton-primary"] {
  background: #dc2626 !important;
  border-color: #b91c1c !important;
}
/* Tab buttons — bordered boxes */
[data-baseweb="tab-list"] {
  gap: 4px !important;
  overflow: visible !important;
}
[data-baseweb="tab"] {
  border: 1px solid var(--border) !important;
  border-radius: 8px 8px 0 0 !important;
  padding: 8px 18px !important;
  background: transparent !important;
}
[data-baseweb="tab"][aria-selected="true"] {
  border-color: #7c3aed !important;
  background: rgba(124,58,237,0.12) !important;
}
[data-baseweb="tab"][aria-selected="true"] p {
  color: #c4b5fd !important;
}
/* Tab panel frame */
div[data-baseweb="tab-panel"] {
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 10px 10px !important;
  padding: 20px 20px 16px !important;
  background: var(--card) !important;
}

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
DATASETS_DIR   = BASE_DIR / "data" / "datasets"

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

CUSTOM_DIR = DATASETS_DIR / "imported"

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


_AC_INFO_HTML = {
    "Othering": (
        "<h3>◆ Othering detector</h3>"
        "<p>Regex matching across <b>4 pattern families</b>:</p>"
        "<ul>"
        "<li><b>Dehumanising metaphors</b> — <em>invasion, swarm, flood, vermin…</em></li>"
        "<li><b>Moral exclusion</b> — <em>go back, don't belong, send them back…</em></li>"
        "<li><b>Generalisations</b> — <em>they all, these people always…</em></li>"
        "<li><b>Threat framing</b> — <em>replacing us, taking over, great replacement…</em></li>"
        "</ul>"
        "<p>Each match adds <b>+1</b> to the othering score (0–4). "
        "Also tags pronoun usage: <em>us-only</em>, <em>them-only</em>, or <em>both</em>.</p>"
    ),
    "Toxicity": (
        "<h3>◎ Toxicity — Detoxify</h3>"
        "<p>Multilingual <b>BERT model</b> fine-tuned on the Jigsaw Unintended Bias dataset. "
        "Returns 6 continuous scores (0–1) per text:</p>"
        "<ul>"
        "<li>toxicity · severe_toxicity · obscene</li>"
        "<li>identity_attack · insult · threat</li>"
        "</ul>"
        "<p>Fast CPU inference (~1 000 rows/min).</p>"
    ),
    "Emotions": (
        "<h3>◉ Emotions — GoEmotions</h3>"
        "<p>HuggingFace pipeline — BERT fine-tuned on Google's GoEmotions corpus (58K Reddit comments).</p>"
        "<p>Returns the <b>top emotion</b> + confidence score per text across <b>28 categories</b>: "
        "<em>admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, "
        "desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, "
        "gratitude, grief, joy, love, nervousness, optimism, pride, realisation, relief, "
        "remorse, sadness, surprise, neutral.</em></p>"
        "<p>⚠ Slow on CPU — use the row slider to limit volume.</p>"
    ),
    "BERTopic": (
        "<h3>▶ BERTopic</h3>"
        "<p>Fully unsupervised topic modelling pipeline:</p>"
        "<ul>"
        "<li><b>SentenceTransformer</b> encodes each text into a dense embedding</li>"
        "<li><b>UMAP</b> reduces dimensions (n_components=5, n_neighbors=15)</li>"
        "<li><b>HDBSCAN</b> clusters the reduced embeddings</li>"
        "<li><b>c-TF-IDF</b> extracts the most representative keywords per cluster</li>"
        "</ul>"
        "<p>Number of topics is <b>data-driven</b> — no manual setting needed. "
        "Outliers are labelled topic -1.</p>"
    ),
}



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
    # Always put Import last
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

def _parse_date_col(df: pd.DataFrame) -> pd.Series | None:
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


# ── Bootstrap data ────────────────────────────────────────────────────────────

if "df_combined" not in st.session_state:
    with st.spinner("Loading datasets..."):
        st.session_state["df_combined"] = build_combined()

df_all = st.session_state["df_combined"]

if "uploaded_df" in st.session_state:
    udf = st.session_state["uploaded_df"].copy()
    udf["dataset"] = f"Imported  ·  {st.session_state.get('upload_name', 'Uploaded')}"
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
    ("Temporal",  "◷"),
]
PAGE_UPLOAD = ("Upload", "▲")

# Detect nav clicks BEFORE the sidebar renders so type= is already correct this run.
# st.session_state[f"nav_{pname}"] is True at the start of the run if that button was clicked.
if "page" not in st.session_state:
    st.session_state["page"] = "Overview"
for _pname, _ in PAGES + [PAGE_UPLOAD]:
    if st.session_state.get(f"nav_{_pname}"):
        st.session_state["page"] = _pname
        break

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
      <span class="we">We</span>
      <span class="vs">vs</span>
      <span class="them">Them</span>
    </div>
    """, unsafe_allow_html=True)

    for pname, picon in PAGES:
        _active = st.session_state["page"] == pname
        st.button(f"{picon}  {pname}", key=f"nav_{pname}", use_container_width=True,
                  type="primary" if _active else "secondary")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    pname, picon = PAGE_UPLOAD
    _active = st.session_state["page"] == pname
    st.button(f"{picon}  {pname}", key=f"nav_{pname}", use_container_width=True,
              type="primary" if _active else "secondary")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Dataset selector (always visible, all pages)
    _ds_counts = df_all["dataset"].value_counts().to_dict() if not df_all.empty else {}
    _groups    = discover_datasets()

    # Pre-detect All/None clicks (same pattern as nav buttons) so propagation
    # and label count are correct in this very run.
    if st.session_state.get("_ds_btn_all"):
        for _lbl in all_dataset_labels:
            st.session_state[f"_ds_{_lbl}"] = True
    if st.session_state.get("_ds_btn_none"):
        for _lbl in all_dataset_labels:
            st.session_state[f"_ds_{_lbl}"] = False

    def _propagate_group(grp_key, was_key, labels):
        _now = st.session_state.get(grp_key, True)
        _was = st.session_state.get(was_key)
        if _was is not None and _now != _was:
            for _l in labels:
                st.session_state[f"_ds_{_l}"] = _now
        else:
            st.session_state[grp_key] = all(
                st.session_state.get(f"_ds_{_l}", True) for _l in labels
            )
        st.session_state[was_key] = st.session_state[grp_key]

    _all_set = set(all_dataset_labels)
    for _group, _files in _groups.items():
        _labels = [f"{_group}  ·  {n}" for n, _ in _files if f"{_group}  ·  {n}" in _all_set]
        if _labels:
            _propagate_group(f"_grp_{_group}", f"_grp_was_{_group}", _labels)

    _n_total   = len(all_dataset_labels)
    _n_checked = sum(1 for _l in all_dataset_labels if st.session_state.get(f"_ds_{_l}", True))
    _ds_label  = f"Datasets · {_n_checked} / {_n_total}"
    selected_datasets = []

    with st.expander(_ds_label, expanded=False, key="_ds_expander"):
        _b1, _b2 = st.columns(2)
        _b1.button("All",  use_container_width=True, key="_ds_btn_all")
        _b2.button("None", use_container_width=True, key="_ds_btn_none")

        st.markdown(" ")

        for _group, _files in _groups.items():
            _emoji = PLATFORM_EMOJI.get(_group, "◈")
            _labels = [
                f"{_group}  ·  {_name}"
                for _name, _ in _files
                if f"{_group}  ·  {_name}" in _all_set
            ]
            if not _labels:
                continue
            st.checkbox(f"{_emoji} {_group.upper()}", key=f"_grp_{_group}")
            _, _col = st.columns([0.08, 0.92])
            with _col:
                for _lbl in _labels:
                    _count = _ds_counts.get(_lbl, 0)
                    _short = re.sub(r"\.(csv|xlsx)$", "", _lbl.split("  ·  ")[-1])
                    _display = f"{_short}" + (f"  ·  {_count:,}" if _count else "")
                    if st.checkbox(_display, value=True, key=f"_ds_{_lbl}"):
                        selected_datasets.append(_lbl)
            st.markdown(" ")

    keyword = ""
    selected_subreddits = []

page = st.session_state["page"]


# ── Upload page ───────────────────────────────────────────────────────────────

if page == "Upload":
    st.markdown('<div class="page-title">Upload & Analyze</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Drop one or more CSVs — text column auto-detected</div>',
                unsafe_allow_html=True)

    # ── File uploader ─────────────────────────────────────────────────────────
    _cur_uploads = st.file_uploader("", type=["csv"], accept_multiple_files=True,
                                    label_visibility="collapsed")

    if _cur_uploads:
        def _tab_icon(fname):
            fl = fname.lower()
            if "tiktok" in fl:   return "▶ "
            if "twitter" in fl or "tweet" in fl or "_x_" in fl: return "◆ "
            if "instagram" in fl or "insta" in fl: return "◎ "
            if "reddit" in fl:   return "◈ "
            if "youtube" in fl:  return "▷ "
            if "facebook" in fl or "fb_" in fl: return "◉ "
            return ""

        _tab_labels = []
        for _uf in _cur_uploads:
            _short = _uf.name[:-4] if _uf.name.endswith(".csv") else _uf.name
            _short = (_short[:24] + "…") if len(_short) > 24 else _short
            _tab_labels.append(f"{_tab_icon(_uf.name)}{_short}")

        _tabs = st.tabs(_tab_labels)

        _AC_DESC = {
            "Othering":  "Detects us-vs-them language — dehumanisation, moral exclusion, generalisations",
            "Toxicity":  "Scores toxicity, insults & threats using Detoxify",
            "Emotions":  "Labels the dominant emotion per text (28 categories via GoEmotions)",
            "BERTopic":  "Finds recurring topics via semantic clustering",
        }

        def _ac_html(icon, name, done, active, uid=""):
            if active:
                bg = "rgba(124,58,237,0.18)"
                badge_bg, badge_color = "rgba(124,58,237,0.35)", "#c4b5fd"
                status = "→ will run"
            elif done:
                bg = "rgba(16,185,129,0.12)"
                badge_bg, badge_color = "rgba(16,185,129,0.22)", "#34d399"
                status = "✓ done"
            else:
                bg = "rgba(112,112,160,0.06)"
                badge_bg, badge_color = "rgba(112,112,160,0.10)", "var(--muted)"
                status = "optional"
            _iid = f"aci_{name.lower()}_{uid}"
            _info = _AC_INFO_HTML.get(name, "")
            return (
                f'<div class="ac-card" style="background:{bg};">'
                f'<input type="checkbox" id="{_iid}" class="ac-info-toggle">'
                f'<label for="{_iid}" class="ac-info-btn">?</label>'
                f'<div class="ac-icon">{icon}</div>'
                f'<div class="ac-name">{name}</div>'
                f'<div class="ac-badge" style="background:{badge_bg};color:{badge_color};">{status}</div>'
                f'<div class="ac-info-popup">'
                f'<label for="{_iid}" class="ac-info-backdrop"></label>'
                f'<div class="ac-info-box">{_info}'
                f'<label for="{_iid}" class="ac-info-close">✕</label>'
                f'</div></div>'
                f'</div>'
            )

        _dyn_css = (
            '[data-testid="stVerticalBlock"]:has(.ac-card){'
            'border:1px solid var(--border);border-radius:10px;padding:12px 12px 10px;}'
            '[data-testid="stVerticalBlock"]:has(.ac-card) .stCheckbox{'
            'background:transparent!important;display:flex!important;justify-content:center!important;}'
        )
        st.markdown(f'<style>{_dyn_css}</style>', unsafe_allow_html=True)

        for _tab, _uf in zip(_tabs, _cur_uploads):
            with _tab:
                _fname = _uf.name
                _fkey = re.sub(r"[^a-zA-Z0-9]", "_", _fname)
                try:
                    raw = pd.read_csv(io.BytesIO(_uf.getvalue()), low_memory=False)
                except Exception as e:
                    st.error(f"Could not read {_fname}: {e}")
                    continue

                # Text column detection
                detected_col, confidence = detect_text_column(raw)
                str_cols = raw.select_dtypes(include="object").columns.tolist()

                if confidence == "not_found":
                    _hint = "❌  No column detected — select manually"
                    _hint_color = "var(--danger, #f87171)"
                    _col_opts = raw.columns.tolist()
                    _col_idx = 0
                elif confidence == "ambiguous":
                    _hint = f"⚠  Multiple candidates — best guess: <code>{detected_col}</code>"
                    _hint_color = "#f59e0b"
                    _col_opts = str_cols
                    _col_idx = str_cols.index(detected_col) if detected_col in str_cols else 0
                else:
                    _lbl = {"exact": "exact match", "case-insensitive": "case-insensitive",
                            "inferred": "inferred from content"}[confidence]
                    _hint = f"✓  Auto-detected — <code>{detected_col}</code> ({_lbl})"
                    _hint_color = "#34d399"
                    _col_opts = str_cols
                    _col_idx = str_cols.index(detected_col) if detected_col in str_cols else 0

                st.markdown(f"""
                <div style="margin:16px 0 4px 0;">
                  <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:12px;
                              letter-spacing:.06em;text-transform:uppercase;color:var(--text);
                              margin-bottom:4px;">Text column</div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:{_hint_color};">
                    {_hint}</div>
                </div>
                """, unsafe_allow_html=True)
                text_col = st.selectbox("Text column", options=_col_opts, index=_col_idx,
                                        label_visibility="collapsed", key=f"tc_{_fkey}")

                if text_col != "text":
                    if "text" in raw.columns:
                        raw = raw.drop(columns=["text"])
                    raw = raw.rename(columns={text_col: "text"})

                with st.expander("Preview (5 rows)"):
                    st.dataframe(raw.head(), use_container_width=True)

                already_processed = all(c in raw.columns for c in ["has_othering", "othering_predicted", "pronoun_type"])
                _has_tox = "toxicity" in raw.columns
                _has_emo  = "emotion" in raw.columns
                _has_ber  = "topic" in raw.columns and "topic_name" in raw.columns

                _slider_key = f"mr_{_fkey}"
                _n = len(raw)
                if _n > 100:
                    _maxv = min(50_000, _n)
                    if _slider_key not in st.session_state:
                        st.session_state[_slider_key] = _maxv
                    max_rows = st.slider("Max rows", min_value=100, max_value=_maxv,
                                         step=100, key=_slider_key)
                else:
                    max_rows = _n

                _run_oth = st.session_state.get(f"_tog_othering_{_fkey}", False) and not already_processed
                _run_det = st.session_state.get(f"_tog_detoxify_{_fkey}", False) and not _has_tox
                _run_emo = st.session_state.get(f"_tog_emotions_{_fkey}",  False) and not _has_emo
                _run_ber = st.session_state.get(f"_tog_bertopic_{_fkey}",  False) and not _has_ber

                st.markdown(" ")
                _ca, _cb, _cc, _cd = st.columns(4)
                with _ca:
                    st.markdown(_ac_html("◆", "Othering", already_processed, _run_oth, _fkey), unsafe_allow_html=True)
                    run_othering = st.toggle("Run Othering",   value=False, disabled=already_processed, key=f"_tog_othering_{_fkey}")
                with _cb:
                    st.markdown(_ac_html("◎", "Toxicity", _has_tox, _run_det, _fkey), unsafe_allow_html=True)
                    run_detoxify = st.toggle("Run Detoxify",   value=False, disabled=_has_tox,           key=f"_tog_detoxify_{_fkey}")
                with _cc:
                    st.markdown(_ac_html("◉", "Emotions", _has_emo, _run_emo, _fkey), unsafe_allow_html=True)
                    run_emotions = st.toggle("Run GoEmotions", value=False, disabled=_has_emo,           key=f"_tog_emotions_{_fkey}")
                with _cd:
                    st.markdown(_ac_html("▶", "BERTopic", _has_ber, _run_ber, _fkey), unsafe_allow_html=True)
                    run_bertopic = st.toggle("Run BERTopic",   value=False, disabled=_has_ber,           key=f"_tog_bertopic_{_fkey}")

                _, _run_col, _ = st.columns([1, 2, 1])
                with _run_col:
                    _do_run = st.button("Run analysis", type="primary", key=f"run_{_fkey}", use_container_width=True)
                components.html(
                    "<script>"
                    "var p=window.parent.document;"
                    "function paint(){"
                    "p.querySelectorAll('[data-testid=\"baseButton-primary\"]').forEach(function(b){"
                    "b.style.setProperty('background-color','#dc2626','important');"
                    "b.style.setProperty('border-color','#b91c1c','important');});}"
                    "paint();setTimeout(paint,200);"
                    "</script>",
                    height=0
                )
                if _do_run:
                    df_input = raw.head(max_rows)
                    progress = st.progress(0.0, "Starting...")
                    try:
                        result = run_full_pipeline(df_input, run_othering, run_detoxify, run_emotions, run_bertopic, progress)
                        st.session_state[f"uploaded_df_{_fkey}"] = result
                        udf = result.copy()
                        udf["dataset"] = f"Imported  ·  {_fname}"
                        existing = st.session_state["df_combined"]
                        existing = existing[existing["dataset"] != udf["dataset"].iloc[0]]
                        st.session_state["df_combined"] = pd.concat([udf, existing], ignore_index=True)
                        CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
                        save_path = CUSTOM_DIR / _fname
                        if save_path.exists():
                            st.session_state[f"_pending_overwrite_{_fkey}"] = result
                            st.session_state[f"_pending_overwrite_path_{_fkey}"] = str(save_path)
                        else:
                            result.to_csv(save_path, index=False)
                            st.success(f"Done — {len(result):,} rows added. Saved to data/datasets/imported/{_fname}.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Pipeline error: {e}")

                if st.session_state.get(f"_pending_overwrite_{_fkey}") is not None:
                    save_path = Path(st.session_state[f"_pending_overwrite_path_{_fkey}"])
                    st.warning(f"**{save_path.name}** already exists. Overwrite?")
                    _, _ow1, _ow2, _ = st.columns([1, 2, 2, 1])
                    if _ow1.button("Overwrite", type="secondary", key=f"_btn_overwrite_{_fkey}", use_container_width=True):
                        st.session_state[f"_pending_overwrite_{_fkey}"].to_csv(save_path, index=False)
                        st.session_state.pop(f"_pending_overwrite_{_fkey}", None)
                        st.session_state.pop(f"_pending_overwrite_path_{_fkey}", None)
                        st.success(f"Saved to data/datasets/imported/{save_path.name}.")
                        st.rerun()
                    if _ow2.button("Cancel", key=f"_btn_cancel_overwrite_{_fkey}", use_container_width=True):
                        st.session_state.pop(f"_pending_overwrite_{_fkey}", None)
                        st.session_state.pop(f"_pending_overwrite_path_{_fkey}", None)
                        st.rerun()

                if f"uploaded_df_{_fkey}" in st.session_state:
                    result = st.session_state[f"uploaded_df_{_fkey}"]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Rows", f"{len(result):,}")
                    c2.metric("% Othering", f"{result['othering_predicted'].mean()*100:.1f}%")
                    tox = result["toxicity"].mean() if result["toxicity"].notna().any() else None
                    c3.metric("Avg toxicity", f"{tox:.3f}" if tox else "n/a")
                    c4.metric("% Them markers",
                              f"{result['pronoun_type'].isin(['them_only','both']).mean()*100:.1f}%")
                    with st.expander("Download result"):
                        st.download_button("Download CSV",
                                           data=result.to_csv(index=False).encode("utf-8"),
                                           file_name=f"{_fname[:-4]}_analyzed.csv",
                                           mime="text/csv", key=f"dl_{_fkey}")

    # ── Manage imported datasets ──────────────────────────────────────────────
    imported_files = sorted(CUSTOM_DIR.glob("*.csv")) if CUSTOM_DIR.exists() else []
    if imported_files:
        st.markdown(" ")
        st.markdown('<div class="section-header">Imported datasets</div>', unsafe_allow_html=True)

        for _f in imported_files:
            if st.session_state.get(f"_del_{_f.name}"):
                _f.unlink()
                del st.session_state["df_combined"]
                st.session_state.pop(f"_del_{_f.name}", None)
                st.rerun()

        _hdr_file, _hdr_oth, _hdr_tox, _hdr_emo, _hdr_ber, _hdr_del = st.columns([3, 1, 1, 1, 1, 1])
        _hdr_file.markdown("**File**")
        _hdr_oth.markdown("**Othering**")
        _hdr_tox.markdown("**Toxicity**")
        _hdr_emo.markdown("**Emotions**")
        _hdr_ber.markdown("**BERTopic**")

        for _f in imported_files:
            try:
                _cols = set(pd.read_csv(_f, nrows=0).columns)
            except Exception:
                _cols = set()
            _has_oth = {"has_othering", "othering_predicted", "pronoun_type"}.issubset(_cols)
            _has_tox = "toxicity" in _cols
            _has_emo = "emotion" in _cols
            _has_ber = "topic" in _cols and "topic_name" in _cols
            _c_file, _c_oth, _c_tox, _c_emo, _c_ber, _c_del = st.columns([3, 1, 1, 1, 1, 1])
            _c_file.markdown(f"`{_f.name}`")
            _c_oth.markdown("✓" if _has_oth else "–")
            _c_tox.markdown("✓" if _has_tox else "–")
            _c_emo.markdown("✓" if _has_emo else "–")
            _c_ber.markdown("✓" if _has_ber else "–")
            _c_del.button("Delete", key=f"_del_{_f.name}", use_container_width=True)

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

    st.markdown(f'<div class="page-subtitle">{len(df_tox):,} posts with toxicity scores · {df["dataset"].nunique()} dataset(s)</div>',
                unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg toxicity",    f"{df_tox['toxicity'].mean():.3f}")
    c2.metric("Median toxicity", f"{df_tox['toxicity'].median():.3f}")
    c3.metric("High (> 0.5)",    f"{(df_tox['toxicity'] > 0.5).mean()*100:.1f}%")
    c4.metric("Max score",       f"{df_tox['toxicity'].max():.3f}")

    st.markdown(" ")
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
    st.markdown(f'<div class="page-subtitle">{len(df_emo):,} posts with emotion labels ({pct_cov:.1f}% coverage) · {df["dataset"].nunique()} dataset(s)</div>',
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
    st.markdown(f'<div class="page-subtitle">{total_oth:,} othering posts detected ({pct_oth:.1f}%) · {df["dataset"].nunique()} dataset(s)</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Othering posts", f"{total_oth:,}")
    c2.metric("Othering rate",  f"{pct_oth:.1f}%")
    score_mean = df["othering_score"].mean()
    c3.metric("Avg score",      f"{score_mean:.2f} / 4")
    both_pct = (df["pronoun_type"] == "both").mean() * 100
    c4.metric("Both we + them", f"{both_pct:.1f}%")

    # ML Classifier metrics block
    if _CLF_METRICS:
        st.markdown('<div class="section-header">ML Classifier — othering detection</div>', unsafe_allow_html=True)
        _m = _CLF_METRICS
        _banner_color = "rgba(124,58,237,0.12)"
        st.markdown(
            f'<div style="background:{_banner_color};border:1px solid rgba(124,58,237,0.3);border-radius:8px;'
            f'padding:10px 16px;font-family:IBM Plex Mono,monospace;font-size:11px;color:#c4b5fd;margin-bottom:12px;">'
            f'Active model: <b>{_m["model_name"]}</b> · '
            f'trained on {_m["n_train"]:,} samples ({_m["n_pos_train"]:,} othering) · '
            f'silver labels from rule-based detector</div>',
            unsafe_allow_html=True,
        )
        _mc1, _mc2, _mc3, _mc4 = st.columns(4)
        _mc1.metric("Precision", f'{_m["precision"]:.3f}')
        _mc2.metric("Recall",    f'{_m["recall"]:.3f}')
        _mc3.metric("F1",        f'{_m["f1"]:.3f}')
        _mc4.metric("Test set",  f'{_m["n_test"]:,}')

        _clf_col_a, _clf_col_b = st.columns(2)
        with _clf_col_a:
            st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#70709f;margin-bottom:6px;">Model comparison</div>', unsafe_allow_html=True)
            _cmp = pd.DataFrame(_m["all_models"]).sort_values("f1", ascending=False)
            _cmp_fig = go.Figure(go.Bar(
                x=_cmp["f1"], y=[f'{r["model"]} / {r["features"]}' for _, r in _cmp.iterrows()],
                orientation="h",
                marker=dict(color=["#7c3aed" if i == 0 else "#2a2a3a" for i in range(len(_cmp))],
                            line_width=0),
                text=[f'F1 {v:.4f}' for v in _cmp["f1"]],
                textposition="inside",
                textfont=dict(family="IBM Plex Mono, monospace", size=10),
            ))
            apply_theme(_cmp_fig, height=max(120, len(_cmp) * 48 + 40))
            _cmp_fig.update_layout(xaxis=dict(range=[0.9, 1.0], title="F1"), yaxis_title="", margin=dict(l=0))
            st.plotly_chart(_cmp_fig, use_container_width=True)

        with _clf_col_b:
            st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#70709f;margin-bottom:6px;">Confusion matrix (test set)</div>', unsafe_allow_html=True)
            _cm = _m["confusion_matrix"]
            _tn, _fp, _fn, _tp = _cm[0][0], _cm[0][1], _cm[1][0], _cm[1][1]
            _cell_labels = [
                [f"<b>{_tn:,}</b><br><span style='font-size:10px;color:#70709f'>Correct non-othering<br>(true negatives)</span>",
                 f"<b>{_fp:,}</b><br><span style='font-size:10px;color:#e11d48'>Non-othering flagged<br>(false positives)</span>"],
                [f"<b>{_fn:,}</b><br><span style='font-size:10px;color:#f59e0b'>Othering missed<br>(false negatives)</span>",
                 f"<b>{_tp:,}</b><br><span style='font-size:10px;color:#10b981'>Correct othering<br>(true positives)</span>"],
            ]
            _cell_colors = [
                ["rgba(16,185,129,0.15)", "rgba(225,29,72,0.25)"],
                ["rgba(245,158,11,0.25)", "rgba(124,58,237,0.3)"],
            ]
            _cm_fig = go.Figure(go.Heatmap(
                z=[[_tn, _fp], [_fn, _tp]],
                x=["Predicted: No", "Predicted: Yes"],
                y=["Actual: No", "Actual: Yes"],
                text=_cell_labels,
                texttemplate="%{text}",
                textfont=dict(family="IBM Plex Mono, monospace", size=12, color="#e2e2f0"),
                colorscale=[[0, "#13131a"], [1, "#1e1e2e"]],
                showscale=False,
            ))
            apply_theme(_cm_fig, height=240)
            _cm_fig.update_layout(
                xaxis=dict(side="top"),
                yaxis=dict(autorange="reversed"),
                margin=dict(t=60),
            )
            st.plotly_chart(_cm_fig, use_container_width=True)

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


# ── Temporal ──────────────────────────────────────────────────────────────────

elif page == "Temporal":
    from events import event_study as _event_study

    st.markdown('<div class="page-title">Temporal Analysis</div>', unsafe_allow_html=True)

    if "post_date" not in df.columns:
        st.info("No date column detected in the current dataset selection. Load a dataset with a date column (TikTok, Twitter, Instagram).")
        st.stop()

    df_t = df.dropna(subset=["post_date"]).copy()
    df_t["post_date"] = pd.to_datetime(df_t["post_date"], errors="coerce")
    df_t = df_t.dropna(subset=["post_date"])

    if df_t.empty:
        st.info("Date column found but no parseable values.")
        st.stop()

    date_min = df_t["post_date"].min().date()
    date_max = df_t["post_date"].max().date()
    n_with_date = len(df_t)

    st.markdown(
        f'<div class="page-subtitle">'
        f'{n_with_date:,} posts with dates · {date_min} → {date_max} · '
        f'{df_t["dataset"].nunique()} dataset(s)'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Load curated events CSV
    _evt_df = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "title": pd.Series(dtype=str),
                             "category": pd.Series(dtype=str), "country": pd.Series(dtype=str),
                             "region": pd.Series(dtype=str), "source_type": pd.Series(dtype=str)})
    _curated_path = BASE_DIR / "data" / "events" / "curated_events.csv"
    if _curated_path.exists():
        try:
            _evt_df = pd.read_csv(_curated_path, parse_dates=["date"])
            _evt_df["date"] = pd.to_datetime(_evt_df["date"], errors="coerce")
        except Exception as _e:
            st.sidebar.warning(f"Events load failed: {_e}")

    _evt_in_range = _evt_df[
        (_evt_df["date"].dt.date >= date_min) &
        (_evt_df["date"].dt.date <= date_max)
    ].copy()

    _colors = ["#7c3aed", "#0ea5e9", "#10b981", "#f59e0b", "#e11d48", "#a855f7"]

    _tab_timeline, _tab_study = st.tabs(["Timeline", "Event Study"])

    # ── TAB 1 : Timeline ─────────────────────────────────────────────────────
    with _tab_timeline:
        _tc1, _tc2 = st.columns(2)
        with _tc1:
            granularity = st.selectbox("Granularity", ["Week", "Month", "Day"], index=1)
        with _tc2:
            show_events = st.checkbox("Show public events", value=True)

        _freq = {"Day": "D", "Week": "W", "Month": "M"}[granularity]
        df_t["period"] = df_t["post_date"].dt.to_period(_freq).dt.to_timestamp()

        _agg = (
            df_t.groupby(["period", "dataset"])
            .agg(posts=("text", "count"),
                 othering_rate=("othering_predicted", "mean"),
                 toxicity_mean=("toxicity", "mean"))
            .reset_index()
        )
        _agg["short_ds"] = _agg["dataset"].apply(
            lambda x: re.sub(r"\.(csv|xlsx)$", "", x.split("  ·  ")[-1]) if "  ·  " in x else x
        )
        _datasets = sorted(_agg["short_ds"].unique())
        _ds_color = {ds: _colors[i % len(_colors)] for i, ds in enumerate(_datasets)}

        def _add_event_lines(fig):
            if show_events and not _evt_in_range.empty:
                for _, _ev in _evt_in_range.iterrows():
                    fig.add_vline(x=_ev["date"], line=dict(color="rgba(226,226,240,0.18)", width=1, dash="dot"))
                    fig.add_annotation(x=_ev["date"], y=1, yref="paper", text=_ev["title"],
                        textangle=-90, font=dict(size=9, color="#70709f"),
                        showarrow=False, xanchor="left", yanchor="top")

        # Volume
        st.markdown('<div class="section-header">Post volume over time</div>', unsafe_allow_html=True)
        fig_vol = go.Figure()
        for ds in _datasets:
            _sub = _agg[_agg["short_ds"] == ds].sort_values("period")
            _c = _ds_color[ds]
            _rgba = f"rgba({int(_c[1:3],16)},{int(_c[3:5],16)},{int(_c[5:7],16)},0.07)"
            fig_vol.add_trace(go.Scatter(x=_sub["period"], y=_sub["posts"],
                mode="lines", name=ds, line=dict(color=_c, width=2),
                fill="tozeroy", fillcolor=_rgba))
        _add_event_lines(fig_vol)
        apply_theme(fig_vol, height=280)
        fig_vol.update_layout(xaxis_title="", yaxis_title="Posts", legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig_vol, use_container_width=True)

        # Othering rate
        st.markdown('<div class="section-header">Othering rate over time</div>', unsafe_allow_html=True)
        fig_oth = go.Figure()
        for ds in _datasets:
            _sub = _agg[_agg["short_ds"] == ds].sort_values("period")
            _y = _sub["othering_rate"] * 100
            if _y.notna().sum() == 0:
                continue
            _mean, _std = _y.mean(), _y.std()
            fig_oth.add_trace(go.Scatter(x=_sub["period"], y=_y, mode="lines", name=ds,
                line=dict(color=_ds_color[ds], width=2)))
            _spikes = _sub[_y > _mean + 1.5 * _std]
            if not _spikes.empty:
                fig_oth.add_trace(go.Scatter(x=_spikes["period"], y=_spikes["othering_rate"]*100,
                    mode="markers", showlegend=False,
                    marker=dict(color="#e11d48", size=8, symbol="diamond")))
        _add_event_lines(fig_oth)
        apply_theme(fig_oth, height=280)
        fig_oth.update_layout(xaxis_title="", yaxis_title="% othering", legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig_oth, use_container_width=True)

        # Toxicity
        if df_t["toxicity"].notna().any():
            st.markdown('<div class="section-header">Mean toxicity over time</div>', unsafe_allow_html=True)
            fig_tox = go.Figure()
            for ds in _datasets:
                _sub = _agg[(_agg["short_ds"] == ds) & _agg["toxicity_mean"].notna()].sort_values("period")
                if _sub.empty:
                    continue
                fig_tox.add_trace(go.Scatter(x=_sub["period"], y=_sub["toxicity_mean"],
                    mode="lines", name=ds, line=dict(color=_ds_color[ds], width=2)))
            _add_event_lines(fig_tox)
            apply_theme(fig_tox, height=260)
            fig_tox.update_layout(xaxis_title="", yaxis_title="Mean toxicity", legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig_tox, use_container_width=True)

        # Emotion distribution over time
        if "emotion" in df_t.columns and df_t["emotion"].notna().any():
            st.markdown('<div class="section-header">Emotion distribution over time</div>', unsafe_allow_html=True)
            _EMO_PALETTE = {
                "neutral":       "#70709f",
                "joy":           "#10b981",
                "admiration":    "#0ea5e9",
                "gratitude":     "#34d399",
                "optimism":      "#fbbf24",
                "love":          "#f472b6",
                "approval":      "#6ee7b7",
                "amusement":     "#22d3ee",
                "anger":         "#e11d48",
                "fear":          "#f59e0b",
                "sadness":       "#7c3aed",
                "disgust":       "#8b5cf6",
                "disapproval":   "#ef4444",
                "disappointment":"#a78bfa",
                "annoyance":     "#fb923c",
                "nervousness":   "#facc15",
            }
            _emo_tl = df_t[df_t["emotion"].notna()].copy()
            _emo_tl["period"] = _emo_tl["post_date"].dt.to_period(_freq).dt.to_timestamp()
            _top_emos_tl = _emo_tl["emotion"].value_counts().head(8).index.tolist()
            _emo_counts = (
                _emo_tl[_emo_tl["emotion"].isin(_top_emos_tl)]
                .groupby(["period", "emotion"]).size().reset_index(name="n")
            )
            _emo_totals = _emo_tl.groupby("period").size().reset_index(name="total")
            _emo_counts = _emo_counts.merge(_emo_totals, on="period")
            _emo_counts["pct"] = _emo_counts["n"] / _emo_counts["total"] * 100
            fig_emo = go.Figure()
            for _emo in _top_emos_tl:
                _sub_emo = _emo_counts[_emo_counts["emotion"] == _emo].sort_values("period")
                if _sub_emo.empty:
                    continue
                fig_emo.add_trace(go.Scatter(
                    x=_sub_emo["period"], y=_sub_emo["pct"],
                    mode="lines", name=_emo,
                    line=dict(color=_EMO_PALETTE.get(_emo, "#9ca3af"), width=1.5),
                    stackgroup="one",
                    groupnorm="percent",
                    hovertemplate="%{y:.1f}%<extra>" + _emo + "</extra>",
                ))
            _add_event_lines(fig_emo)
            apply_theme(fig_emo, height=280)
            fig_emo.update_layout(
                xaxis_title="", yaxis_title="Share of emotional posts (%)",
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_emo, use_container_width=True)

        # Spike table
        st.markdown('<div class="section-header">Detected spikes (othering rate > mean + 1.5σ)</div>', unsafe_allow_html=True)
        _spike_rows = []
        for ds in _datasets:
            _sub = _agg[_agg["short_ds"] == ds].sort_values("period").copy()
            _y = _sub["othering_rate"] * 100
            if _y.std() == 0 or _y.notna().sum() < 3:
                continue
            _mean, _std = _y.mean(), _y.std()
            for _, _row in _sub[_y > _mean + 1.5 * _std].iterrows():
                _ts = pd.Timestamp(_row["period"])
                _near = _evt_df.copy()
                _near["_dist"] = (_near["date"] - _ts).abs().dt.days
                _near = _near[_near["_dist"] <= 45]
                _evt_label = _near.sort_values("_dist")["title"].iloc[0] if not _near.empty else ""
                _spike_rows.append({
                    "Period": str(_row["period"])[:10],
                    "Dataset": ds,
                    "Posts": int(_row["posts"]),
                    "Othering %": f'{_row["othering_rate"]*100:.1f}%',
                    "vs mean": f'+{_row["othering_rate"]*100 - _mean:.1f}pp',
                    "Nearest event (±45d)": _evt_label,
                })
        if _spike_rows:
            st.dataframe(pd.DataFrame(_spike_rows), use_container_width=True, hide_index=True)
        else:
            st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#70709f;">No significant spikes detected.</div>', unsafe_allow_html=True)

    # ── TAB 2 : Event Study ──────────────────────────────────────────────────
    with _tab_study:
        st.markdown(
            '<div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#70709f;margin-bottom:12px;">'
            'For each selected event, shows daily metric deviation from the global mean '
            'in a ±N day window. Reveals whether language spikes before, during, or after events.'
            '</div>',
            unsafe_allow_html=True,
        )

        _es1, _es2 = st.columns(2)
        with _es1:
            _es_window = st.slider("Window (days each side)", 3, 30, 14)
        with _es2:
            _es_base_metrics = [m for m in ["othering_predicted", "toxicity"]
                                if m in df_t.columns and df_t[m].notna().any()]
            _es_emo_opts = []
            if "emotion" in df_t.columns and df_t["emotion"].notna().any():
                _avail_emos = df_t["emotion"].value_counts().index.tolist()
                for _e in ["anger", "fear", "sadness", "disgust", "disapproval",
                           "annoyance", "nervousness", "grief", "remorse"]:
                    if _e in _avail_emos:
                        _es_emo_opts.append(f"emo:{_e}")
            _es_all_metrics = _es_base_metrics + _es_emo_opts

            def _fmt_es_metric(x):
                if x == "othering_predicted": return "Othering rate"
                if x == "toxicity":           return "Toxicity"
                return f"Emotion: {x.split(':', 1)[1]}"

            _es_metric = st.selectbox("Metric", _es_all_metrics, format_func=_fmt_es_metric)

        _evt_options = _evt_df.sort_values("date")["title"].tolist()

        if not _evt_options:
            st.info("No events available.")
        else:
            # Score each event: lift in othering + toxicity at/after event vs before
            _df_t_day = df_t.copy()
            _df_t_day["_day"] = _df_t_day["post_date"].dt.normalize()
            _score_cols = [c for c in ["othering_predicted", "toxicity"] if c in _df_t_day.columns and _df_t_day[c].notna().any()]
            _daily = _df_t_day.groupby("_day")[_score_cols].mean() if _score_cols else pd.DataFrame()

            def _score_event(evt_date):
                if _daily.empty:
                    return 0
                pre_start  = evt_date - pd.Timedelta(days=_es_window)
                pre_end    = evt_date - pd.Timedelta(days=1)
                post_end   = evt_date + pd.Timedelta(days=7)
                pre_days   = _daily.loc[pre_start:pre_end]
                post_days  = _daily.loc[evt_date:post_end]
                if len(pre_days) < 2 or len(post_days) < 1:
                    return 0
                lift = 0
                for col in _score_cols:
                    pre_mean  = pre_days[col].mean()
                    post_mean = post_days[col].mean()
                    pre_std   = pre_days[col].std() + 1e-6
                    lift += max(0, (post_mean - pre_mean) / pre_std)
                return lift

            _evt_scores = _evt_df.copy()
            _evt_scores["_score"] = _evt_scores["date"].apply(_score_event)
            _suggested = _evt_scores.nlargest(3, "_score")["title"].tolist()
            _suggested = [t for t in _suggested if _evt_scores.loc[_evt_scores["title"]==t, "_score"].values[0] > 0]

            _selected_evts = st.multiselect(
                "Select events to analyse",
                _evt_options,
                default=_suggested if _suggested else _evt_options[:min(3, len(_evt_options))],
                help="Pre-selection: events with the most posts and signal in your loaded datasets.",
            )

            if _selected_evts:
                _evts_to_study = _evt_df[_evt_df["title"].isin(_selected_evts)]

                # For emotion metrics, materialise a binary column before passing to event_study
                _df_t_es = df_t.copy()
                if _es_metric and _es_metric.startswith("emo:"):
                    _emo_name = _es_metric.split(":", 1)[1]
                    _df_t_es[_es_metric] = np.where(
                        _df_t_es["emotion"].isna(), np.nan,
                        (_df_t_es["emotion"] == _emo_name).astype(float),
                    )

                _es_result = _event_study(_df_t_es, _evts_to_study,
                                          window_days=_es_window,
                                          metric_cols=[_es_metric])

                if _es_result.empty:
                    st.info("Not enough post data in the windows around the selected events.")
                else:
                    _metric_dev = f"{_es_metric}_deviation"
                    _metric_val = _es_metric
                    if _es_metric == "othering_predicted":
                        _ylabel_dev = "Othering rate deviation (pp)"
                        _scale = 100
                    elif _es_metric.startswith("emo:"):
                        _ylabel_dev = f"{_es_metric.split(':',1)[1].capitalize()} rate deviation (pp)"
                        _scale = 100
                    else:
                        _ylabel_dev = "Toxicity deviation"
                        _scale = 1

                    # Individual event windows
                    st.markdown('<div class="section-header">Deviation from global mean per event</div>', unsafe_allow_html=True)
                    fig_es = go.Figure()
                    for i, evt_title in enumerate(_selected_evts):
                        _sub = _es_result[_es_result["event_title"] == evt_title].sort_values("day_offset")
                        if _sub.empty:
                            continue
                        _col = _colors[i % len(_colors)]
                        fig_es.add_trace(go.Scatter(
                            x=_sub["day_offset"],
                            y=_sub[_metric_dev] * _scale,
                            mode="lines+markers",
                            name=evt_title[:50],
                            line=dict(color=_col, width=2),
                            marker=dict(size=4),
                        ))
                    fig_es.add_hline(y=0, line=dict(color="rgba(226,226,240,0.3)", width=1, dash="dash"))
                    fig_es.add_vline(x=0, line=dict(color="rgba(226,226,240,0.5)", width=1, dash="dot"))
                    fig_es.add_annotation(x=0, y=1, yref="paper", text="event day",
                        font=dict(size=9, color="#70709f"), showarrow=False, xanchor="left")
                    apply_theme(fig_es, height=340)
                    fig_es.update_layout(
                        xaxis_title="Days relative to event",
                        yaxis_title=_ylabel_dev,
                        legend=dict(orientation="h", y=1.1),
                    )
                    st.plotly_chart(fig_es, use_container_width=True)

                    # Summary table
                    st.markdown('<div class="section-header">Peak deviation per event</div>', unsafe_allow_html=True)
                    _summary_rows = []
                    for evt_title in _selected_evts:
                        _sub = _es_result[_es_result["event_title"] == evt_title]
                        if _sub.empty:
                            continue
                        _dev_series = (_sub[_metric_dev] * _scale).abs()
                        if _dev_series.isna().all():
                            continue
                        _peak_row = _sub.loc[_dev_series.idxmax()]
                        _summary_rows.append({
                            "Event": evt_title[:60],
                            "Date": str(_peak_row["event_date"])[:10],
                            "Peak day offset": int(_peak_row["day_offset"]),
                            "Peak deviation": f'{_peak_row[_metric_dev] * _scale:+.3f}',
                            "Posts in window": int(_sub["n_posts"].sum()),
                        })
                    if _summary_rows:
                        st.dataframe(pd.DataFrame(_summary_rows), use_container_width=True, hide_index=True)
