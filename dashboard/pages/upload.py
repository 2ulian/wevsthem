import io
import re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from data import CUSTOM_DIR, detect_text_column, fast_pipeline, run_full_pipeline, build_combined

_AC_INFO_HTML = {
    "Othering": (
        "<h3>Othering detector</h3>"
        "<p>Regex matching across <b>4 pattern families</b>:</p>"
        "<ul>"
        "<li><b>Dehumanising metaphors</b> - <em>invasion, swarm, flood, vermin...</em></li>"
        "<li><b>Moral exclusion</b> - <em>go back, don't belong, send them back...</em></li>"
        "<li><b>Generalisations</b> - <em>they all, these people always...</em></li>"
        "<li><b>Threat framing</b> - <em>replacing us, taking over, great replacement...</em></li>"
        "</ul>"
        "<p>Each match adds <b>+1</b> to the othering score (0-4). "
        "Also tags pronoun usage: <em>us-only</em>, <em>them-only</em>, or <em>both</em>.</p>"
    ),
    "Toxicity": (
        "<h3>Toxicity - Detoxify</h3>"
        "<p>Multilingual <b>BERT model</b> fine-tuned on the Jigsaw Unintended Bias dataset. "
        "Returns 6 continuous scores (0-1) per text:</p>"
        "<ul>"
        "<li>toxicity · severe_toxicity · obscene</li>"
        "<li>identity_attack · insult · threat</li>"
        "</ul>"
        "<p>Fast CPU inference (~1 000 rows/min).</p>"
    ),
    "Emotions": (
        "<h3>Emotions - GoEmotions</h3>"
        "<p>HuggingFace pipeline - BERT fine-tuned on Google's GoEmotions corpus (58K Reddit comments).</p>"
        "<p>Returns the <b>top emotion</b> + confidence score per text across <b>28 categories</b>: "
        "<em>admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, "
        "desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, "
        "gratitude, grief, joy, love, nervousness, optimism, pride, realisation, relief, "
        "remorse, sadness, surprise, neutral.</em></p>"
        "<p>Slow on CPU - use the row slider to limit volume.</p>"
    ),
    "BERTopic": (
        "<h3>BERTopic</h3>"
        "<p>Fully unsupervised topic modelling pipeline:</p>"
        "<ul>"
        "<li><b>SentenceTransformer</b> encodes each text into a dense embedding</li>"
        "<li><b>UMAP</b> reduces dimensions (n_components=5, n_neighbors=15)</li>"
        "<li><b>HDBSCAN</b> clusters the reduced embeddings</li>"
        "<li><b>c-TF-IDF</b> extracts the most representative keywords per cluster</li>"
        "</ul>"
        "<p>Number of topics is <b>data-driven</b> - no manual setting needed. "
        "Outliers are labelled topic -1.</p>"
    ),
}

_AC_DESC = {
    "Othering": "Detects us-vs-them language - dehumanisation, moral exclusion, generalisations",
    "Toxicity": "Scores toxicity, insults & threats using Detoxify",
    "Emotions": "Labels the dominant emotion per text (28 categories via GoEmotions)",
    "BERTopic": "Finds recurring topics via semantic clustering",
}


def render(df_all: pd.DataFrame):
    st.markdown('<div class="page-title">Upload & Analyze</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Drop one or more CSVs - text column auto-detected</div>',
                unsafe_allow_html=True)

    _cur_uploads = st.file_uploader("", type=["csv"], accept_multiple_files=True,
                                    label_visibility="collapsed")

    if _cur_uploads:
        def _tab_icon(fname):
            fl = fname.lower()
            if "tiktok"   in fl: return "  "
            if "twitter"  in fl or "tweet" in fl or "_x_" in fl: return "  "
            if "instagram" in fl or "insta" in fl: return "  "
            if "reddit"   in fl: return "  "
            if "youtube"  in fl: return "  "
            if "facebook" in fl or "fb_"  in fl: return "  "
            return ""

        _tab_labels = []
        for _uf in _cur_uploads:
            _short = _uf.name[:-4] if _uf.name.endswith(".csv") else _uf.name
            _short = (_short[:24] + "...") if len(_short) > 24 else _short
            _tab_labels.append(f"{_tab_icon(_uf.name)}{_short}")

        _tabs = st.tabs(_tab_labels)

        def _ac_html(icon, name, done, active, uid=""):
            if active:
                bg = "rgba(124,58,237,0.18)"
                badge_bg, badge_color = "rgba(124,58,237,0.35)", "#c4b5fd"
                status = "will run"
            elif done:
                bg = "rgba(16,185,129,0.12)"
                badge_bg, badge_color = "rgba(16,185,129,0.22)", "#34d399"
                status = "done"
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
                f'<label for="{_iid}" class="ac-info-close">x</label>'
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

                detected_col, confidence = detect_text_column(raw)
                str_cols = raw.select_dtypes(include="object").columns.tolist()

                if confidence == "not_found":
                    _hint = "No column detected - select manually"
                    _hint_color = "var(--danger, #f87171)"
                    _col_opts = raw.columns.tolist()
                    _col_idx = 0
                elif confidence == "ambiguous":
                    _hint = f"Multiple candidates - best guess: <code>{detected_col}</code>"
                    _hint_color = "#f59e0b"
                    _col_opts = str_cols
                    _col_idx = str_cols.index(detected_col) if detected_col in str_cols else 0
                else:
                    _lbl = {"exact": "exact match", "case-insensitive": "case-insensitive",
                            "inferred": "inferred from content"}[confidence]
                    _hint = f"Auto-detected - <code>{detected_col}</code> ({_lbl})"
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
                            st.success(f"Done - {len(result):,} rows added. Saved to data/datasets/imported/{_fname}.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Pipeline error: {e}")

                if st.session_state.get(f"_pending_overwrite_{_fkey}") is not None:
                    save_path = __import__("pathlib").Path(st.session_state[f"_pending_overwrite_path_{_fkey}"])
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
            _c_oth.markdown("ok" if _has_oth else "-")
            _c_tox.markdown("ok" if _has_tox else "-")
            _c_emo.markdown("ok" if _has_emo else "-")
            _c_ber.markdown("ok" if _has_ber else "-")
            _c_del.button("Delete", key=f"_del_{_f.name}", use_container_width=True)
