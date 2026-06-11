# We vs Them

NLP pipeline for detecting othering language on social media (Twitter, Instagram, TikTok). Includes a classifier, topic modelling, event study analysis, and a Streamlit dashboard.

## Requirements

- Python 3.10+
- pip

No other system dependency is needed for a CPU-only setup.

## Installation

### 1. Clone the repo

```bash
git clone <repo-url>
cd wevsthem
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies

**CPU-only machine (no GPU):**
```bash
pip install -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

**Machine with NVIDIA GPU (CUDA):**
```bash
pip install -r requirements.txt
```

> The first run will download the sentence-transformer model (`all-MiniLM-L6-v2`, ~90 MB) and BERTopic embeddings automatically.

## Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

The dashboard hot-reloads on file save — no need to restart after code changes.

## Project structure

```
dashboard/
  app.py          Entry point — config, sidebar, routing
  theme.py        CSS and Plotly theme
  data.py         Constants, pipeline functions, dataset loaders
  pages/          One file per page (overview, toxicity, emotions, othering, topics, temporal, upload)
data/
  datasets/       Raw datasets (Twitter, Instagram, TikTok)
  processed/      Classified and enriched data
  events/         Curated event timeline
models/           Trained othering classifier (.pkl)
src/              Pipeline source code
  pipeline/       Data ingestion and processing scripts
reports/          Internship report (DOCX + PDF)
```

## Notes

- Datasets and the trained classifier are committed to the repo — no separate download needed.
- `data/cache/` (BERTopic embeddings) is gitignored and regenerated on first run. This takes a few minutes depending on dataset size.
- The GoEmotions and Detoxify models are only downloaded when you run them via the Upload page.
