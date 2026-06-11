import re

import plotly.graph_objects as go
import streamlit as st

from theme import PALETTE, apply_theme
from data import EMOTION_VALENCE, VALENCE_COLOR


def render(df):
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
        st.markdown('<div class="section-header">Delta% othering - non-othering</div>', unsafe_allow_html=True)
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
            xaxis=dict(zeroline=True, zerolinecolor="#3a3a5a", zerolinewidth=2, title="Delta%"),
            yaxis_title="",
        )
        st.plotly_chart(fig2, use_container_width=True)

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
