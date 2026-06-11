import re

import plotly.graph_objects as go
import streamlit as st

from theme import PALETTE, apply_theme, hex_rgba


def render(df):
    st.markdown('<div class="page-title">Overview</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{len(df):,} posts across {df["dataset"].nunique()} dataset(s)</div>',
                unsafe_allow_html=True)

    total     = len(df)
    pct_oth   = df["othering_predicted"].mean() * 100
    tox_ok    = df["toxicity"].notna().any()
    avg_tox   = df["toxicity"].mean() if tox_ok else None
    pct_toxic = (df["toxicity"].dropna() > 0.5).mean() * 100 if tox_ok else None
    top_emo   = df["emotion"].mode()[0] if df["emotion"].notna().any() else "n/a"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total posts",   f"{total:,}")
    c2.metric("Othering rate", f"{pct_oth:.1f}%")
    c3.metric("High toxicity", f"{pct_toxic:.1f}%" if pct_toxic is not None else "n/a")
    c4.metric("Top emotion",   top_emo)

    st.markdown(" ")

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
