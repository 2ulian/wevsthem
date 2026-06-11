import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st

from theme import CSS
from data import (
    build_combined, discover_datasets, PLATFORM_EMOJI,
    CUSTOM_DIR, BASE_DIR,
)
import pages.overview  as page_overview
import pages.toxicity  as page_toxicity
import pages.emotions  as page_emotions
import pages.othering  as page_othering
import pages.topics    as page_topics
import pages.temporal  as page_temporal
import pages.upload    as page_upload

st.set_page_config(layout="wide", page_title="We vs Them", page_icon="⚡")
st.markdown(CSS, unsafe_allow_html=True)

# Bootstrap combined dataset
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

PAGES = [
    ("Overview", "◈"),
    ("Toxicity", "◎"),
    ("Emotions", "◉"),
    ("Othering", "◆"),
    ("Topics",   "◐"),
    ("Temporal", "◷"),
]
PAGE_UPLOAD = ("Upload", "▲")

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

    _ds_counts = df_all["dataset"].value_counts().to_dict() if not df_all.empty else {}
    _groups    = discover_datasets()

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

page = st.session_state["page"]

# Upload page (no dataset gate needed)
if page == "Upload":
    page_upload.render(df_all)
    st.stop()

# Dataset gate
if not selected_datasets:
    st.warning("Select at least one dataset from the sidebar.")
    st.stop()

df_full = df_all[df_all["dataset"].isin(selected_datasets)].copy()
df = df_full.copy()

if len(df) == 0:
    st.warning("No results for the current filters.")
    st.stop()

# Route to page
if   page == "Overview":  page_overview.render(df)
elif page == "Toxicity":  page_toxicity.render(df)
elif page == "Emotions":  page_emotions.render(df)
elif page == "Othering":  page_othering.render(df)
elif page == "Topics":    page_topics.render(df)
elif page == "Temporal":  page_temporal.render(df)
