# We vs Them

NLP pipeline for detecting othering language on social media (Twitter, Instagram, TikTok). Includes a classifier, topic modelling, event study analysis, and a Streamlit dashboard.

## Setup

```bash
git clone <repo>
cd wevsthem

# CPU-only machine
pip install -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Machine with GPU (CUDA)
pip install -r requirements.txt
```

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

## Project structure

```
dashboard/      Streamlit app
data/
  datasets/     Raw datasets (Twitter, Instagram, TikTok)
  processed/    Classified and enriched data
  events/       Curated event timeline
models/         Trained othering classifier (.pkl)
src/            Pipeline source code
  pipeline/     Data ingestion and processing scripts
reports/        Internship report (DOCX + PDF)
```

## Data

Datasets are committed to the repo. The `data/cache/` directory (BERTopic embeddings) is gitignored and will be regenerated on first run — this takes a few minutes.
