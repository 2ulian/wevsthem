import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme import PALETTE, apply_theme
from data import BASE_DIR


def render(df):
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
        f'{n_with_date:,} posts with dates · {date_min} to {date_max} · '
        f'{df_t["dataset"].nunique()} dataset(s)'
        f'</div>',
        unsafe_allow_html=True,
    )

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

    _colors = PALETTE

    _tab_timeline, _tab_study = st.tabs(["Timeline", "Event Study"])

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

        st.markdown('<div class="section-header">Detected spikes (othering rate > mean + 1.5 sigma)</div>', unsafe_allow_html=True)
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
                    "Nearest event (+-45d)": _evt_label,
                })
        if _spike_rows:
            st.dataframe(pd.DataFrame(_spike_rows), use_container_width=True, hide_index=True)
        else:
            st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#70709f;">No significant spikes detected.</div>', unsafe_allow_html=True)

    with _tab_study:
        st.markdown(
            '<div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#70709f;margin-bottom:12px;">'
            'For each selected event, shows daily metric deviation from the global mean '
            'in a +/-N day window. Reveals whether language spikes before, during, or after events.'
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
                    if _es_metric == "othering_predicted":
                        _ylabel_dev = "Othering rate deviation (pp)"
                        _scale = 100
                    elif _es_metric.startswith("emo:"):
                        _ylabel_dev = f"{_es_metric.split(':',1)[1].capitalize()} rate deviation (pp)"
                        _scale = 100
                    else:
                        _ylabel_dev = "Toxicity deviation"
                        _scale = 1

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
