# 🛒 QuickCommerce Analytics & Recommendation System

> **Big Data Analytics & Recommendation System for Quick-Commerce Platforms**
> Scoped to **Jayanagar, Bangalore** — comparing **Blinkit**, **Zepto**, and **Swiggy Instamart**

---

## 📌 Overview

A cloud-integrated analytics platform that:

- Generates **statistically realistic simulated data** for 51 products across 3 quick-commerce platforms at 15-minute intervals
- Processes data at scale using **Apache Spark (PySpark)** on Databricks
- Applies **Spark MLlib** for demand forecasting (GBTRegressor) and trending product classification (RandomForestClassifier)
- Surfaces insights through an interactive **Streamlit dashboard** with **Plotly** visualizations
- Offers a **Deepgram-powered voice interface** for natural language product search

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│  DATA GENERATION — dataset.py                        │
│  51 products · 3 platforms · 15-min cadence           │
└──────────────────┬───────────────────────────────────┘
                   │  Parquet files
                   ▼
┌──────────────────────────────────────────────────────┐
│  PREPROCESSING — preprocessing.py (PySpark)          │
│  Schema normalisation · Aggregation · Analytics       │
└──────────────────┬───────────────────────────────────┘
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
┌─────────────────┐  ┌─────────────────────────┐
│  ML PIPELINE    │  │  RECOMMENDATION ENGINE  │
│  GBTRegressor   │  │  Preference-Weighted    │
│  RandomForest   │  │  Scoring & Ranking      │
└────────┬────────┘  └───────────┬─────────────┘
         └──────────┬────────────┘
                    ▼
┌──────────────────────────────────────────────────────┐
│  STREAMLIT DASHBOARD — app.py                        │
│  Plotly charts · Preference sliders · Voice query     │
└──────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
code2.0/
├── dataset.py                 # Data simulator (51 products × 3 platforms × 35K+ timestamps)
├── config.py                  # Shared constants, paths, column mappings, product registry
├── preprocessing.py           # PySpark schema normalisation & analytics aggregations
├── ml_pipeline.py             # GBTRegressor (demand) + RandomForest (trend) training
├── recommendation_engine.py   # Preference-weighted composite scoring logic
├── voice_handler.py           # Deepgram STT, NLP intent parser, gTTS response
├── app.py                     # Streamlit dashboard (4 pages)
├── requirements.txt           # Python dependencies
├── .streamlit/
│   └── config.toml            # Streamlit dark theme config
├── output/
│   ├── raw/                   # Raw Parquet files (blinkit, zepto, swiggy)
│   ├── processed/             # Unified schema + analytics parquets
│   ├── curated/               # Demand forecasts + trend labels
│   └── models/                # Saved Spark ML models
└── QuickCommerce_Analytics_PRD_v1_4.md  # Product Requirements Document
```

---

## ⚙️ Setup & Installation

### Prerequisites

- **Python 3.11+**
- **Java 8 or 11** (required for PySpark)
- ~4 GB free RAM for Spark local mode

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Environment Variables (optional, for voice)

```bash
set DEEPGRAM_API_KEY=your_deepgram_api_key_here
```

---

## 🚀 Execution Order

### Step 1 — Generate Data

> ⚠️ Takes ~20–40 min. Raw Parquet files are already pre-generated in `output/raw/`.

```bash
python dataset.py
```

**Output:** `output/raw/blinkit_raw.parquet`, `zepto_raw.parquet`, `swiggy_raw.parquet` (~250 MB each)

### Step 2 — Preprocess & Normalise

```bash
python preprocessing.py
```

- Reads 3 platform-specific schemas
- Normalises column names, stock formats, delivery time strings
- Produces `output/processed/unified.parquet` + 4 analytics tables

### Step 3 — Train ML Models

```bash
python ml_pipeline.py
```

- **GBTRegressor** → demand forecasting (target MAPE ≤ 20%)
- **RandomForestClassifier** → trend classification (target F1 ≥ 0.80)
- Train: April–May | Test: June (time-based split)
- Outputs to `output/curated/` and `output/models/`

### Step 4 — Launch Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 📊 Dashboard Pages

| Page | Features |
|------|----------|
| **Product Search & Comparison** | Search bar, preference sliders (Price/Delivery/Quality), side-by-side platform table, 7-day price trend chart |
| **Category Trends** | Category selector, top 10 trending products, demand forecast charts |
| **Demand Forecasting** | Product-level forecast timeline, predicted vs actual overlay, stockout alerts |
| **Voice Query Center** | Microphone input → Deepgram STT → recommendation → spoken TTS response |

### Preference Sliders

Three sliders control platform scoring weights (normalised to sum = 1.0):

- 🏷️ **Price** (0–100, default: 50)
- 🚴 **Delivery Time** (0–100, default: 30)
- ⭐ **Product Quality** (0–100, default: 20)

---

## 🤖 ML Models

| Model | Algorithm | Target | Metric | Threshold |
|-------|-----------|--------|--------|-----------|
| Demand Forecast | `GBTRegressor` (MLlib) | `units_ordered` | MAPE | ≤ 20% |
| Trend Classifier | `RandomForestClassifier` (MLlib) | `trending / stable / declining` | Weighted F1 | ≥ 0.80 |

---

## 🎤 Voice Interface

Supports 5 intents via Deepgram Nova-2 STT:

| Intent | Example |
|--------|---------|
| `recommend` | "Where should I order milk from?" |
| `find_cheapest` | "Which is cheapest for 500g paneer?" |
| `check_delivery` | "Which platform is fastest right now?" |
| `trending` | "What's trending today?" |
| `compare` | "Compare Blinkit and Zepto for chips" |

Fallback: text input is always available if mic is unavailable.

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| Data Generation | Python, pandas, numpy, PyArrow |
| Processing | Apache Spark 3.5 (PySpark) |
| ML | Spark MLlib (GBTRegressor, RandomForestClassifier) |
| Dashboard | Streamlit 1.35+, Plotly 5.x |
| Voice STT | Deepgram API (Nova-2, en-IN) |
| Voice TTS | gTTS |
| Storage | Parquet (local / AWS S3) |

---

## 📝 Notes

- **Location:** Fixed to Jayanagar, Bangalore (single-location scope)
- **Data:** 100% simulated — no scraping or API access to any platform
- **Memory:** Dataset generator uses chunked batch writing (peak ≤ 200 MB/chunk)
- **S3:** For production, update paths in `config.py` to S3 URIs and use `boto3`
- All platform names are trademarks of their respective owners

---

## 📄 License

Academic project — RVCE, Department of CSE, 6th Semester BDT Lab.
