import numpy as np
import plotly.graph_objects as go
import streamlit as st

from theme import PALETTE, apply_theme


def render(df):
    st.markdown('<div class="page-title">Topics</div>', unsafe_allow_html=True)

    has_topics = "topic" in df.columns and "topic_name" in df.columns
    if not has_topics:
        st.info("No BERTopic data available. Run the pipeline with BERTopic enabled on the Upload page.")
        return

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
    tstats["label"] = tstats["topic"].astype(int).astype(str) + "  " + tstats["topic_name"].fillna("").astype(str)

    n_topics  = len(tstats)
    top_topic = tstats.nlargest(1, "nb_posts").iloc[0]
    top_other = tstats.nlargest(1, "pct_othering").iloc[0]
    outlier_n = (topic_df["topic"] == -1).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Topics found",        f"{n_topics}")
    c2.metric("Largest topic",       top_topic["topic_name"].split("_")[0] + "...", f"{top_topic['nb_posts']:,} posts")
    c3.metric("Most othering topic", top_other["topic_name"].split("_")[0] + "...", f"{top_other['pct_othering']:.1f}%")
    c4.metric("Outlier posts (-1)",  f"{outlier_n:,}")

    st.markdown('<div class="section-header">Top 20 topics by size - colour = othering rate</div>', unsafe_allow_html=True)
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
        customdata=np.stack([top20["pct_othering"], top20["mean_toxicity"]], axis=1),
        hovertemplate="<b>%{y}</b><br>Posts: %{x:,}<br>Othering: %{customdata[0]:.1f}%<br>Toxicity: %{customdata[1]:.3f}<extra></extra>",
    ))
    apply_theme(fig_t, height=600)
    fig_t.update_layout(yaxis_title="", xaxis_title="Number of posts")
    st.plotly_chart(fig_t, use_container_width=True)

    st.markdown('<div class="section-header">Top 15 topics by othering rate</div>', unsafe_allow_html=True)
    top_oth = tstats[tstats["nb_posts"] >= 10].nlargest(15, "pct_othering").sort_values("pct_othering", ascending=True)
    fig_o = go.Figure(go.Bar(
        y=top_oth["label"], x=top_oth["pct_othering"],
        orientation="h",
        marker=dict(color="#e11d48", opacity=0.8),
        text=top_oth["pct_othering"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Othering: %{x:.1f}%<extra></extra>",
    ))
    apply_theme(fig_o, height=500)
    fig_o.update_layout(yaxis_title="", xaxis_title="Othering rate (%)")
    st.plotly_chart(fig_o, use_container_width=True)

    st.markdown('<div class="section-header">All topics - detail</div>', unsafe_allow_html=True)
    display_df = tstats.sort_values("nb_posts", ascending=False).rename(columns={
        "topic":         "ID",
        "topic_name":    "Topic",
        "nb_posts":      "Posts",
        "pct_othering":  "Othering %",
        "mean_toxicity": "Avg toxicity",
    })[["ID", "Topic", "Posts", "Othering %", "Avg toxicity"]]
    st.dataframe(display_df, use_container_width=True, hide_index=True)
