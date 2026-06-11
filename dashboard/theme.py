import plotly.graph_objects as go

PALETTE = ["#7c3aed", "#e11d48", "#0ea5e9", "#10b981", "#f59e0b", "#ec4899", "#06b6d4", "#84cc16"]

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

/* Active nav button */
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
/* Run analysis button - red */
[data-baseweb="tab-panel"] [data-testid="baseButton-primary"] {
  background: #dc2626 !important;
  border-color: #b91c1c !important;
}
/* Tab buttons */
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


def apply_theme(fig, height=380):
    fig.update_layout(**CHART_LAYOUT, height=height)
    return fig


def hex_rgba(hex_color: str, alpha: float = 0.2) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
