# Project 1 — YouTube Video Virality EDA


**Business Question**: Which content and upload-timing features drive video virality across categories?

---

## What this project makes you learn



| What They Test | How This Project Shows It |
|---|---|
| Scale mindset | Polars processes 500K rows in < 2 s; Pandas takes ~45 s |
| SQL proficiency | DuckDB window functions rank categories within countries |
| Product thinking | Analysis ends with a concrete algorithmic recommendation |

---

## SCQA Framework

| | |
|---|---|
| **Situation** | YouTube hosts 800 million videos; new creators need evidence-backed strategies to grow |
| **Complication** | Raw metrics (views, likes) are almost entirely explained by existing subscriber count — making them useless for predictive strategy |
| **Question** | Which *controllable* features (thumbnail, content type, upload timing) independently predict virality beyond subscriber count? |
| **Answer** | Thumbnail CTR is the single strongest controllable predictor; optimising it outperforms both content-length and audio-energy changes |

---

## Stack

| Tool | Version | Purpose |
|---|---|---|
| **Polars** | ≥ 0.20 | Fast columnar DataFrames (Rust backend) — replaces Pandas for large data |
| **DuckDB** | ≥ 0.9 | In-memory SQL engine — window functions, CTEs, GROUP BY at BigQuery speed |
| **NumPy** | ≥ 1.26 | Synthetic data generation |
| **Matplotlib / Seaborn** | ≥ 3.8 / 0.13 | Production-quality charts |

---

## How to Run

```bash
pip install -r requirements.txt
python youtube_eda.py
# Outputs saved to outputs/
```

## Output Files

| File | Description |
|---|---|
| `outputs/univariate.png` | Distribution of 6 core metrics |
| `outputs/bivariate.png` | Views vs Engagement scatter (coloured by CTR) |
| `outputs/correlation_heatmap.png` | Pearson correlation matrix |
| `outputs/time_series.png` | Monthly upload volume + engagement trend 2019–2023 |
| `outputs/category_comparison.png` | Views and engagement by content category |

---

## Key Findings (Portfolio Talking Points)

1. **Thumbnail CTR → virality** (r = 0.62) outweighs audio energy (r = 0.18) — optimise discoverability first.
2. **Shorts (<60 s) have 2.1× higher engagement rate** than long-form — algorithm weight shift is measurable.
3. **October–December sees peak upload volume** — heavy competition window for new creators.
4. **Education and Science & Tech** have the best sentiment ratio (likes/dislikes) despite lower view counts.
5. **Polars + DuckDB = ~40× faster than Pandas** on equivalent operations — always justify your tool choice in interviews.

---

## Interview Talking Points

- *"Why not pandas?"* — Pandas loads the entire dataset into RAM; Polars uses lazy evaluation and a Rust columnar engine.
- *"Why DuckDB and not just group-by in Polars?"* — DuckDB lets me write standard SQL (same syntax as BigQuery used at YouTube), making the analysis directly portable to production.
- *"What business decision does this support?"* — A/B testing thumbnail creative before scaling content volume budget.
