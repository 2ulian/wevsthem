import re
import ast as _ast
import html as _html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme import PALETTE, apply_theme
from data import _CLF_METRICS


def render(df):
    st.markdown('<div class="page-title">Othering</div>', unsafe_allow_html=True)

    total_oth = df["othering_predicted"].sum()
    pct_oth   = df["othering_predicted"].mean() * 100
    st.markdown(f'<div class="page-subtitle">{total_oth:,} othering posts detected ({pct_oth:.1f}%) · {df["dataset"].nunique()} dataset(s)</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Othering posts", f"{total_oth:,}")
    c2.metric("Othering rate",  f"{pct_oth:.1f}%")
    score_mean = df["othering_score"].mean() if "othering_score" in df.columns else 0
    c3.metric("Avg score",      f"{score_mean:.2f} / 4")
    both_pct = (df["pronoun_type"] == "both").mean() * 100
    c4.metric("Both we + them", f"{both_pct:.1f}%")

    if _CLF_METRICS:
        _m = _CLF_METRICS
        st.markdown('<div class="section-header">ML Classifier - othering detection</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:rgba(124,58,237,0.12);border:1px solid rgba(124,58,237,0.3);border-radius:8px;'
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
        st.markdown('<div class="section-header">Pronoun x dataset heatmap</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="section-header">Post examples</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        smin = float(df["othering_score"].min()) if "othering_score" in df.columns and not df["othering_score"].isna().all() else 0.0
        smax = float(df["othering_score"].max()) if "othering_score" in df.columns and not df["othering_score"].isna().all() else 4.0
        score_range = st.slider("Score range", min_value=smin, max_value=smax,
                                value=(smin, smax), step=1.0)
    with c2:
        othering_only = st.checkbox("Othering only", value=True)
    with c3:
        n_cards = st.slider("Cards to show", 5, 40, 10)

    ex = df.dropna(subset=["othering_score"]) if "othering_score" in df.columns else df.copy()
    if "othering_score" in ex.columns:
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
        text_safe = _html.escape(raw_text[:300]) + ("..." if len(raw_text) > 300 else "")
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


