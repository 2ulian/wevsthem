import re
import html as _html

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from theme import PALETTE, apply_theme


def render(df):
    st.markdown('<div class="page-title">Toxicity</div>', unsafe_allow_html=True)

    df_tox = df.dropna(subset=["toxicity"])
    if df_tox.empty:
        st.info("No toxicity data. Run Detoxify via Upload, or select a default dataset.")
        st.stop()

    st.markdown(f'<div class="page-subtitle">{len(df_tox):,} posts with toxicity scores · {df["dataset"].nunique()} dataset(s)</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg toxicity",    f"{df_tox['toxicity'].mean():.3f}")
    c2.metric("Median toxicity", f"{df_tox['toxicity'].median():.3f}")
    c3.metric("High (> 0.5)",    f"{(df_tox['toxicity'] > 0.5).mean()*100:.1f}%")
    c4.metric("Max score",       f"{df_tox['toxicity'].max():.3f}")

    st.markdown(" ")
    st.markdown('<div class="section-header">Toxicity × othering score</div>', unsafe_allow_html=True)
    _sdf = df_tox.copy()
    _sdf["othering_score"] = (
        __import__("pandas").to_numeric(_sdf["othering_score"], errors="coerce").fillna(0)
        if "othering_score" in _sdf.columns else 0
    )
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
        yaxis=dict(title="Othering score (0-4)", tickvals=[0, 1, 2, 3, 4], zeroline=False),
    )
    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#7070a0;margin-bottom:6px;">violet = othering · blue = non-othering</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_s, use_container_width=True)

    st.markdown('<div class="section-header">Most toxic posts</div>', unsafe_allow_html=True)
    top_n = st.slider("Show top N", 5, 50, 10, key="tox_top_n")
    top_posts = df_tox.nlargest(top_n, "toxicity")

    for _, row in top_posts.iterrows():
        tox_val  = row["toxicity"]
        oth_val  = row.get("othering_predicted", 0)
        raw_text = str(row.get("text", ""))
        ds_short = re.sub(r"\.(csv|xlsx)$", "", row["dataset"].split("  ·  ")[-1]) if "  ·  " in row["dataset"] else row["dataset"]
        tox_cls  = "badge-red" if tox_val > 0.7 else "badge-purple" if tox_val > 0.4 else "badge-gray"
        text_safe = _html.escape(raw_text[:280]) + ("..." if len(raw_text) > 280 else "")
        badges = f'<span class="badge {tox_cls}">tox {tox_val:.3f}</span>'
        if oth_val:
            badges += ' <span class="badge badge-purple">othering</span>'
        badges += f' <span class="badge badge-gray">{_html.escape(ds_short)}</span>'
        st.markdown(
            f'<div class="post-card toxic"><span style="font-size:13px;">{text_safe}</span>'
            f'<div class="meta">{badges}</div></div>',
            unsafe_allow_html=True,
        )
