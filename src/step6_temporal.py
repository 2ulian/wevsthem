# Step 6: Temporal analysis and platform comparison
# Reads: data/processed/dataset_final.csv
# Writes: reports/figures/temporal_*.png, reports/platform_comparison.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data(path="data/processed/dataset_final.csv", max_rows=None):
    print(f"Loading {path}...")
    df = pd.read_csv(path, nrows=max_rows)
    print(f"  {len(df)} rows loaded.")
    return df


# ─── TASK 1 — Prepare time data ──────────────────────────────────────────────

def prepare_time_data(df):
    if "created_utc" not in df.columns:
        print("[WARNING] 'created_utc' column not found — skipping temporal analysis.")
        print("          To enable it, preserve created_utc through the full pipeline.")
        return None

    df = df[df["created_utc"] != 0].copy()
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
    df = df.dropna(subset=["created_utc"])
    df["date"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    df = df.dropna(subset=["date"])
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    print(f"  {len(df)} Reddit posts with valid timestamps.")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df


# ─── TASK 2 — Time series plots ──────────────────────────────────────────────

def plot_temporal_toxicity(df_time):
    monthly = df_time.groupby("year_month")["toxicity"].mean().reset_index()
    monthly = monthly.sort_values("year_month")

    plt.figure(figsize=(14, 5))
    plt.plot(monthly["year_month"], monthly["toxicity"], marker="o", linewidth=1.5)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.title("Average Toxicity Score per Month")
    plt.ylabel("Average Toxicity")
    plt.xlabel("Month")
    plt.tight_layout()
    out = f"{FIGURES_DIR}/temporal_toxicity.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")
    return monthly


def plot_temporal_othering(df_time):
    monthly = df_time.groupby("year_month")["has_othering"].mean().reset_index()
    monthly = monthly.sort_values("year_month")
    monthly["othering_rate"] = monthly["has_othering"] * 100

    plt.figure(figsize=(14, 5))
    plt.plot(monthly["year_month"], monthly["othering_rate"], marker="o",
             color="darkorange", linewidth=1.5)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.title("Othering Rate per Month (%)")
    plt.ylabel("% Posts with Othering")
    plt.xlabel("Month")
    plt.tight_layout()
    out = f"{FIGURES_DIR}/temporal_othering.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")
    return monthly


# ─── TASK 3 — Detect spikes ──────────────────────────────────────────────────

def detect_spikes(monthly_tox, monthly_other):
    print("\n--- Toxicity spikes (mean + 1.5 std) ---")
    mean_t = monthly_tox["toxicity"].mean()
    std_t = monthly_tox["toxicity"].std()
    threshold_t = mean_t + 1.5 * std_t
    spikes_t = monthly_tox[monthly_tox["toxicity"] > threshold_t]
    for _, row in spikes_t.iterrows():
        print(f"  [SPIKE] {row['year_month']} — toxicity: {row['toxicity']:.3f} → ADD EVENT HERE")
    if spikes_t.empty:
        print("  No toxicity spikes detected.")

    print("\n--- Othering rate spikes (mean + 1.5 std) ---")
    mean_o = monthly_other["othering_rate"].mean()
    std_o = monthly_other["othering_rate"].std()
    threshold_o = mean_o + 1.5 * std_o
    spikes_o = monthly_other[monthly_other["othering_rate"] > threshold_o]
    for _, row in spikes_o.iterrows():
        print(f"  [SPIKE] {row['year_month']} — othering: {row['othering_rate']:.1f}% → ADD EVENT HERE")
    if spikes_o.empty:
        print("  No othering spikes detected.")


# ─── TASK 4 — Platform comparison ────────────────────────────────────────────

def platform_comparison(df):
    print("\n--- Platform comparison by subreddit ---")

    # Keep only Reddit posts with a real subreddit
    reddit = df[df["source"] == "reddit"].copy()
    if reddit.empty:
        print("  No Reddit posts found.")
        return None

    def top_value(series):
        return series.value_counts().idxmax() if not series.empty else "N/A"

    def top_pattern(series):
        all_patterns = []
        for val in series.dropna():
            try:
                patterns = eval(val) if isinstance(val, str) else val
                if isinstance(patterns, list):
                    all_patterns.extend(patterns)
            except Exception:
                pass
        if not all_patterns:
            return "N/A"
        return pd.Series(all_patterns).value_counts().idxmax()

    rows = []
    for subreddit, group in reddit.groupby("subreddit"):
        rows.append({
            "subreddit": subreddit,
            "n_posts": len(group),
            "avg_toxicity": round(group["toxicity"].mean(), 4),
            "othering_rate_%": round(group["has_othering"].mean() * 100, 2),
            "top_emotion": top_value(group["emotion"]),
            "top_pattern": top_pattern(group["matched_patterns"]),
        })

    table = pd.DataFrame(rows).sort_values("n_posts", ascending=False)
    out = "reports/platform_comparison.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"\n  Saved {out}")

    # Bar chart: avg toxicity per subreddit
    plt.figure(figsize=(10, 5))
    sns.barplot(data=table, x="subreddit", y="avg_toxicity", palette="Reds_d")
    plt.title("Average Toxicity by Subreddit")
    plt.ylabel("Average Toxicity")
    plt.xlabel("")
    plt.tight_layout()
    out_fig = f"{FIGURES_DIR}/platform_toxicity.png"
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"  Saved {out_fig}")

    # Bar chart: othering rate per subreddit
    plt.figure(figsize=(10, 5))
    sns.barplot(data=table, x="subreddit", y="othering_rate_%", palette="Oranges_d")
    plt.title("Othering Rate by Subreddit (%)")
    plt.ylabel("% Posts with Othering")
    plt.xlabel("")
    plt.tight_layout()
    out_fig2 = f"{FIGURES_DIR}/platform_othering.png"
    plt.savefig(out_fig2, dpi=150)
    plt.close()
    print(f"  Saved {out_fig2}")

    return table


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main(max_rows=None):
    df = load_data(max_rows=max_rows)

    # Temporal analysis (only if created_utc is available)
    print("\n=== TEMPORAL ANALYSIS ===")
    df_time = prepare_time_data(df)
    if df_time is not None:
        monthly_tox = plot_temporal_toxicity(df_time)
        monthly_other = plot_temporal_othering(df_time)
        detect_spikes(monthly_tox, monthly_other)
    else:
        print("  Skipping time series plots.")

    # Platform comparison (always runs)
    print("\n=== PLATFORM COMPARISON ===")
    platform_comparison(df)

    print("\n=== DONE ===")
    print("  Figures saved to reports/figures/")
    print("  Platform table saved to reports/platform_comparison.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Limit rows for testing (e.g. --max_rows 5000)")
    args = parser.parse_args()
    main(max_rows=args.max_rows)
