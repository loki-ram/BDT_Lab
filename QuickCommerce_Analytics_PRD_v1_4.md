# Product Requirements Document (PRD)
## Big Data Analytics & Recommendation System for Quick-Commerce Platforms

**Version:** 1.4
**Date:** May 2026
**Status:** Draft
**Document Type:** Product Requirements Document
**Location Scope:** Jayanagar, Bangalore

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Product Overview](#2-product-overview)
3. [Goals & Objectives](#3-goals--objectives)
4. [Target Users](#4-target-users)
5. [System Architecture Overview](#5-system-architecture-overview)
6. [Detailed Methodology & Process Flow](#6-detailed-methodology--process-flow)
   - 6.1 [Data Generation Layer](#61-data-generation-layer)
   - 6.2 [Cloud Storage Layer](#62-cloud-storage-layer)
   - 6.3 [Data Processing Layer](#63-data-processing-layer)
   - 6.4 [ML & Demand Forecasting Layer](#64-ml--demand-forecasting-layer)
   - 6.5 [Recommendation Engine](#65-recommendation-engine)
   - 6.6 [Visualization & Dashboard Layer](#66-visualization--dashboard-layer)
   - 6.7 [Voice Interaction Layer](#67-voice-interaction-layer)
7. [Functional Requirements](#7-functional-requirements)
8. [Non-Functional Requirements](#8-non-functional-requirements)
9. [Data Schema & Models](#9-data-schema--models)
10. [Technology Stack](#10-technology-stack)
11. [Platform Comparison Logic](#11-platform-comparison-logic)
12. [API & Integration Specifications](#12-api--integration-specifications)
13. [Security & Compliance](#13-security--compliance)
14. [Deployment Strategy](#14-deployment-strategy)
15. [Risks & Mitigations](#15-risks--mitigations)
16. [Success Metrics & KPIs](#16-success-metrics--kpis)
17. [Glossary](#17-glossary)

---

## 1. Executive Summary

This document defines the product requirements for a **cloud-integrated big data analytics and recommendation system** scoped exclusively to **Jayanagar, Bangalore**. The system uses a pre-generated, statistically realistic dataset simulating product data for the three dominant quick-commerce platforms — **Blinkit**, **Zepto**, and **Swiggy Instamart** — processes it at scale using Apache Spark on Databricks, applies machine learning for demand forecasting and trending product classification, and surfaces actionable insights through an interactive Streamlit dashboard with Plotly-based visualizations and Databricks pipeline monitoring.

The system also includes a **Deepgram API-powered voice interface**, enabling users to speak natural language queries — e.g. "Show me the cheapest oil available right now" — which are transcribed and processed to fetch recommendations and render results on the dashboard with a spoken response.

> **Data Strategy:** Because Blinkit, Zepto, and Swiggy Instamart do not permit web scraping or expose public APIs, this system uses a high-fidelity Python-based data simulator (`dataset.py`) seeded for **Jayanagar, Bangalore** (`LOCATION = "Jayanagar, Bangalore"`). The simulator generates a complete pre-generated dataset at **15-minute intervals**, producing a simulated streaming architecture using continuously generated transactional records that flow through the analytics pipeline.

---

## 2. Product Overview

### 2.1 Problem Statement

Residents of Jayanagar, Bangalore face the following friction when ordering groceries and essentials via quick-commerce platforms:

- **Price opacity:** The same product can vary significantly in price across Blinkit, Zepto, and Swiggy Instamart, and prices change dynamically based on demand, time of day, and local supply.
- **Delivery uncertainty:** Estimated delivery times vary and are not easily comparable in a single view.
- **Fragmented discovery:** Users must open multiple apps and manually compare offerings — time-consuming and error-prone.
- **No demand visibility:** Neither consumers nor local retailers can anticipate stock shortages, price surges, or trending products specific to their neighbourhood.
- **High interaction friction:** Even comparison apps require multiple taps and form inputs; users want to simply ask a question and get an answer.

### 2.2 Solution

A unified analytics platform scoped to Jayanagar, Bangalore that:

- Generates a pre-built, statistically realistic dataset simulating transactional records for Blinkit, Zepto, and Swiggy Instamart at a **15-minute cadence** for the Jayanagar location
- Stores and processes data at scale on Amazon S3 + Apache Spark on Databricks
- Applies MLlib-based demand forecasting and trending product classification
- Visualizes pricing trends, delivery performance, and demand behavior through **Plotly charts on the Streamlit dashboard**
- Offers a Streamlit dashboard with **user-adjustable preference sliders** for personalised platform recommendations
- Delivers a **Deepgram API-powered voice interface** where a user speaks a query, the audio is transcribed, the system fetches a recommendation, and the result is rendered on screen with a spoken response

### 2.3 Scope

**In Scope:**
- Single-location pre-generated data simulation: **Jayanagar, Bangalore** only
- Simulated data generation for Blinkit, Zepto, and Swiggy Instamart via `dataset.py`
- Batch data processing on Databricks
- Product price, rating, delivery time, and availability comparison
- User preference-weighted platform recommendation (price / delivery time / product quality sliders)
- Demand forecasting and trending product classification using MLlib
- Databricks job monitoring for pipeline observability
- Deepgram API voice interface (STT → query → data fetch → scoring → dashboard render → TTS response)

**Out of Scope (V1):**
- Multi-location or city-wide coverage (Jayanagar only)
- Order placement or cart integration on any platform
- User account management or purchase history tracking
- Support for platforms beyond the three listed
- Mobile native application (iOS/Android)
- Real-time streaming from live platform APIs

---

## 3. Goals & Objectives

### 3.1 Business Goals

| Goal | Description |
|------|-------------|
| G1 | Provide a single unified view for cross-platform quick-commerce comparison in Jayanagar, Bangalore |
| G2 | Process and surface insights from a multi-million scalable dataset using distributed big data tools |
| G3 | Forecast demand with ≥ 80% accuracy (MAPE ≤ 20%) for products in Jayanagar |
| G4 | Classify trending products with weighted F1-score ≥ 0.80 |
| G5 | Enable voice-driven product search and comparison via Deepgram API |
| G6 | Allow users to customise platform recommendations via preference sliders for price, delivery time, and product quality |
| G7 | Provide pipeline observability through Databricks monitoring and structured Plotly analytics on the dashboard |

### 3.2 Product Objectives

- **Objective 1:** Reduce the time a user spends comparing platforms from ~10 minutes (manual, multi-app) to under 30 seconds.
- **Objective 2:** Surface demand forecasts 24–72 hours ahead to help Jayanagar users make proactive purchase decisions.
- **Objective 3:** Enable users to get a full platform comparison by speaking a single sentence — the system transcribes the query, fetches data, scores platforms, and returns a spoken result within 10 seconds.
- **Objective 4:** Enable users to personalise the recommendation engine in real time by adjusting their price, delivery time, and product quality preferences.

---

## 4. Target Users

### 4.1 Primary Users

**Jayanagar Residents (Everyday Consumers)**
- Age: 22–55, urban, digitally active
- Location: Jayanagar, Bangalore (all zones)
- Need: Quick, intelligent answers to "Where should I order from right now in Jayanagar?"
- Behaviour: Price-sensitive, delivery-speed-conscious; wants to speak a query and get an instant answer without navigating multiple apps

**Local Small Retailers & Kirana Store Owners in Jayanagar**
- Need: Understand platform pricing trends and demand patterns specific to Jayanagar to position inventory and margins
- Behaviour: Interested in category-level trends and daily price movements

### 4.2 Secondary Users

**Market Analysts & Researchers**
- Need: Jayanagar-specific aggregated trend data, demand forecasts, and platform performance benchmarks
- Behaviour: Power users who export data and run custom queries

---

## 5. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA GENERATION LAYER                          │
│    dataset.py Simulator — Location: Jayanagar, Bangalore            │
│    Pre-generated · 15-min cadence · Blinkit · Zepto · Swiggy        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ Parquet files (3 platform schemas)
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CLOUD STORAGE LAYER (AWS S3)                   │
│    Raw Zone (Parquet) → Processed Zone → Curated Zone               │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│               DATA PROCESSING LAYER (Apache Spark / Databricks)     │
│    Schema Normalisation · Aggregation · Preprocessing               │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                ┌───────────────┴────────────────┐
                ▼                                ▼
┌──────────────────────┐             ┌──────────────────────────┐
│  ML LAYER (MLlib)    │             │  RECOMMENDATION ENGINE   │
│  GBTRegressor        │             │  Preference-Weighted     │
│  (Demand Forecast)   │             │  Scoring & Ranking       │
│  RandomForest        │             └────────────┬─────────────┘
│  (Trend Classifier)  │                          │
└──────────┬───────────┘                          │
           └──────────────┬───────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│              PRESENTATION LAYER                                     │
│   Streamlit Dashboard (Streamlit Community Cloud)                   │
│   Preference Sliders · Comparison View · Forecast Charts            │
│                                                                     │
│   Databricks Monitoring (Pipeline Observability)                    │
│   Spark job metrics · Data freshness · Error tracking               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│           VOICE INTERACTION LAYER (Deepgram API)                    │
│  Browser Mic → Deepgram STT → Query Builder →                       │
│  S3 Data Fetch → Scoring Engine → Dashboard Render → TTS Response   │
│               (Full cycle < 10 seconds)                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Detailed Methodology & Process Flow

### 6.1 Data Generation Layer

#### 6.1.1 Overview & Rationale

Since Blinkit, Zepto, and Swiggy Instamart do not permit web scraping and do not offer public APIs, the system uses a purpose-built Python data simulator (`dataset.py`) as the sole data source. All data is generated for a **single fixed location: Jayanagar, Bangalore**, matching the hardcoded constant in the simulator:

```python
LOCATION = "Jayanagar, Bangalore"
```

The simulator generates statistically realistic product records for all three platforms at **15-minute intervals** across a full calendar year, producing a pre-generated dataset that is uploaded to S3 and processed as a simulated streaming architecture using continuously generated transactional records. The dataset is pre-generated and processed in a simulated streaming workflow

#### 6.1.2 Product Catalogue

The simulator covers 51 products across 11 categories, all scoped to typical Jayanagar consumer demand patterns:

| Category | Products | Pack Sizes |
|----------|----------|------------|
| Dairy | Milk, Curd, Buttermilk, Paneer, Butter, Cheese Slices | 2–4 options per product |
| Vegetables | Tomato, Onion, Potato, Carrot, Capsicum, Beans | Up to 4 pack sizes (250g–2kg) |
| Fruits | Banana, Apple, Mango, Grapes, Pomegranate | 2–3 pack sizes |
| Snacks | Lays, Parle-G, Maggi, Kurkure, Dark Fantasy | 2–3 pack sizes |
| Beverages | Coca Cola, Tropicana, Bisleri, Red Bull | 2–4 pack sizes |
| Staples | Basmati Rice, Toor Dal, Atta, Salt, Sunflower Oil, Mustard Oil | Up to 5 pack sizes |
| Personal Care | Dettol Handwash, H&S Shampoo, Face Wash, Dove Soap, Colgate | 2–3 pack sizes |
| Household | Vim Dishwash, Harpic, Lizol | 3 pack sizes |
| Baby Care | Pampers, Johnson Baby Oil, Baby Wipes, Cerelac | Fixed packs |
| Pet Food | Pedigree Adult, Whiskas Cat, Dog Treats | Fixed packs |
| Healthcare | Crocin 500mg, Dettol Sanitizer, Band Aid, ORS Sachet | Fixed packs |

#### 6.1.3 Simulation Mechanism

The simulator encodes multiple layers of realistic variation across the pre-generated dataset:

**Timestamp cadence — 15-minute intervals**

```python
start_dt   = datetime(2024, 1, 1, 6, 0, 0)
timestamps = [start_dt + timedelta(minutes=15*i) for i in range(365*96)]
# 365 days × 96 slots/day = 35,040 time slots per platform
```

**Time-of-day dynamics**

| Time Window | Demand Multiplier | Price Surge | Notes |
|-------------|------------------|-------------|-------|
| 8:00–10:00 AM | 1.35× | 1.01–1.05× | Morning grocery rush |
| 12:00–14:00 PM | 1.20× | None | Lunch-hour orders |
| 18:00–21:00 PM | 1.45× | 1.02–1.08× | Evening peak, highest surge |
| 22:00–6:00 AM | 1.60× | None | Late-night premium |
| Other hours | 1.00× | None | Baseline |

**Bangalore weather simulation**

Bangalore's seasonal weather profile is encoded per month, affecting delivery times and demand:

| Month | Condition | Rain Probability | Delivery Delay Factor |
|-------|-----------|-----------------|----------------------|
| January–February | Clear | 5% | 1.00× |
| March | Partly Cloudy | 10% | 1.02× |
| April–May | Hot | 15–20% | 1.05× |
| June–August | Rainy | 75–85% | 1.30–1.35× |
| September | Rainy | 60% | 1.20× |
| October | Partly Cloudy | 30% | 1.10× |
| November–December | Clear | 5–10% | 1.00–1.02× |

**Festival demand signals**

| Event | Date | Demand Multiplier | Price Multiplier | Duration |
|-------|------|------------------|-----------------|----------|
| New Year | January 1 | 1.8× | 1.15× | 3 days |
| Holi | March 8 | 1.6× | 1.10× | 2 days |
| IPL Season Start | March 25 | 1.4× | 1.05× | 60 days |
| Diwali | October 24 | 2.2× | 1.20× | 5 days |
| Christmas | December 25 | 1.5× | 1.12× | 3 days |

**Seasonal produce pricing** (Mango, Tomato, Onion) varies per month via dedicated multiplier tables, producing realistic neighbourhood-level price fluctuations.

#### 6.1.4 Platform-Specific Schemas & Differentiation

| Platform | Price Factor | Delivery Base | Avg Rating | Stock Rate | Discount Rate |
|----------|-------------|---------------|------------|------------|---------------|
| Blinkit | 1.05–1.15× | 10–18 mins | 4.3 | 92% | 15% |
| Zepto | 0.95–1.05× | 8–15 mins | 4.1 | 88% | 35% |
| Swiggy Instamart | 1.00–1.10× | 12–22 mins | 4.2 | 85% | 15% |

Each platform uses distinct column naming conventions (e.g. `cost` vs `selling_price` vs `offer_price`), requiring the normalisation pipeline to perform cross-platform field mapping.

#### 6.1.5 Memory-Efficient Batch Generation

The chunked batch writer (`generate_platform_batched`) processes the full year in 7-day chunks, writing part files and merging via PyArrow streaming writer. Peak RAM per chunk ≈ 150–200 MB, enabling generation of multi-million scalable records on an 8 GB laptop without hitting memory limits.

```bash
# S3 upload after generation
aws s3 cp output/raw/ s3://qcommerce-datalake/raw/jayanagar/ --recursive
```

---

### 6.2 Cloud Storage Layer

#### 6.2.1 Amazon S3 Bucket Structure

All data is stored under a `jayanagar/` prefix:

```
s3://qcommerce-datalake/
├── raw/
│   └── jayanagar/
│       ├── blinkit/
│       │   └── blinkit_raw.parquet
│       ├── zepto/
│       │   └── zepto_raw.parquet
│       └── swiggy_instamart/
│           └── swiggy_raw.parquet
│
├── processed/
│   └── jayanagar/
│       ├── unified_schema/
│       ├── price_analytics/
│       ├── delivery_analytics/
│       ├── revenue_analytics/
│       └── stock_analytics/
│
└── curated/
    └── jayanagar/
        ├── platform_comparison/
        ├── demand_forecasts/
        ├── trend_labels/
        └── recommendations/
```

#### 6.2.2 Storage Format

- **Raw zone:** Apache Parquet (PyArrow-native output from simulator) — three separate files, one per platform
- **Processed and curated zones:** Parquet, partitioned by relevant dimensions for efficient Spark reads

#### 6.2.3 Data Retention Policy

| Zone | Retention | Notes |
|------|-----------|-------|
| Raw | 30 days | Original platform-schema Parquet files |
| Processed | 180 days | Unified schema and analytics outputs |
| Curated | 365 days | Aggregated insights, forecasts, scores |

---

### 6.3 Data Processing Layer

#### 6.3.1 Apache Spark on Databricks

All processing uses Apache Spark (PySpark) on a Databricks Community Edition workspace connected to S3. Since the dataset is scoped to a single location and is pre-generated, a single-node or small cluster is sufficient for batch processing.

#### 6.3.2 Processing Flow

```
[3 raw Parquet files in S3 — blinkit / zepto / swiggy]
        │
        ▼
[PySpark Preprocessing Job]
  ├── Read each platform file separately
  ├── Rename columns to unified canonical schema
  ├── Parse delivery_time: "15 mins" STRING → 15 INTEGER (Swiggy)
  ├── Normalise stock columns:
  │     "available"/"out_of_stock" → boolean (Blinkit)
  │     "yes"/"no" → boolean (Zepto)
  │     boolean → boolean (Swiggy, passthrough)
  ├── Normalise weekend_flag: int 0/1 → boolean (Zepto)
  ├── Add platform column to each DataFrame
  ├── Union all 3 DataFrames into single unified DataFrame
  ├── Deduplication on (record_id, platform, snapshot_time)
  └── Write unified Parquet to S3 processed zone
        │
        ▼
[PySpark Analytics Jobs — one per analytics layer]
  ├── Price analytics (Section 6.4 inputs)
  ├── Delivery analytics
  ├── Revenue & demand analytics
  └── Stock & availability analytics
        │
        ▼
[ML Pipeline — Section 6.4]
        │
        ▼
[Scoring Engine — Section 6.5]
        │
        ▼
[Write curated Parquet to S3 serving zone]
```

#### 6.3.3 Key Aggregations

| Aggregation | Grain | Output |
|-------------|-------|--------|
| Average price by product × platform | Hourly | Price trend table |
| Min delivery time by platform | Current slot | Live comparison |
| Availability rate by product × platform | 24-hour window | Stock reliability score |
| Price volatility index | 7-day window | Price stability metric |
| Revenue by category × platform | Daily | Category revenue breakdown |

---

### 6.4 ML & Demand Forecasting Layer

#### 6.4.1 Features Available from Simulator

Because the simulator embeds context fields directly in every record, most features are already present and do not need to be re-derived:

| Feature Group | Fields |
|---------------|--------|
| Temporal | hour_of_day, day_of_week, month, is_weekend, is_peak_hour |
| Product | category, pack_size, unit, bulk_discount_pct, seasonal_factor |
| Pricing | MRP, discount_pct, effective_price, price_per_unit |
| Demand signals | units_ordered, revenue |
| Environmental | weather_condition, is_raining, delivery_time_minutes |
| Platform | platform name, stock rate, rating |

#### 6.4.2 Models

**Model 1 — Demand Forecasting (Regression)**
- Algorithm: `GBTRegressor` (Spark MLlib)
- Target: `units_ordered` (demand proxy)
- Horizon: 24h and 72h ahead
- Train/test split: Time-based — April + May for training, June for testing
- Metric: MAPE ≤ 20%
- Pipeline: `StringIndexer` → `VectorAssembler` → `GBTRegressor`

**Model 2 — Trending Product Classification**
- Algorithm: `RandomForestClassifier` (Spark MLlib)
- Target: trend label (`trending` | `stable` | `declining`)
  - Derived from `units_ordered` rolling averages — products with consistently increasing order_qty over a 7-day window are labelled `trending`
- Metric: Weighted F1-score ≥ 0.80
- Pipeline: `StringIndexer` → `VectorAssembler` → `RandomForestClassifier`

> **Note:** Price anomaly detection using Isolation Forest has been removed from scope. The pre-generated dataset does not contain organically anomalous pricing behavior, and adding synthetic anomalies would undermine the dataset's statistical integrity without adding meaningful analytical value.

#### 6.4.3 Model Training Pipeline

```
[Unified Parquet — S3 processed zone]
        │
        ▼
[Feature Engineering (PySpark)]
  ├── Cast booleans to integers for MLlib
  ├── StringIndexer (platform, category, product, weather)
  └── VectorAssembler (all numeric + encoded features)
        │
        ▼
[Time-based train/test split]
  ├── Train: months 4–5 (April–May)
  └── Test:  month 6  (June)
        │
        ▼
[GBTRegressor — demand forecasting]
[RandomForestClassifier — trend classification]
        │
        ▼
[Evaluation — MAPE / Weighted F1]
        │
        ▼
[Save model to S3 models/ zone]
        │
        ▼
[Batch inference — write predictions to curated zone]
```

---

### 6.5 Recommendation Engine

#### 6.5.1 Preference-Weighted Scoring

Users control platform scoring via three sliders. Weights are normalised to sum to 1.0:

```
w_price    = price_slider    / (price_slider + delivery_slider + quality_slider)
w_delivery = delivery_slider / (price_slider + delivery_slider + quality_slider)
w_quality  = quality_slider  / (price_slider + delivery_slider + quality_slider)

Platform Score = w_price    × Price Score
              + w_delivery  × Delivery Score
              + w_quality   × Quality Score

Price Score(p)    = 1 - (price(p) - min_price) / (max_price - min_price)
Delivery Score(p) = 1 - (delivery_time(p) - min_time) / (max_time - min_time)
Quality Score(p)  = 0.5 × (rating(p) / 5.0) + 0.5 × availability_rate_7d(p)
```

#### 6.5.2 Recommendation Output

- `recommended_platform`: Best platform under current weights
- `best_price_platform`: Lowest per-unit price
- `fastest_delivery_platform`: Shortest ETA
- `highest_quality_platform`: Best rating + availability
- `score_breakdown`: Per-dimension scores
- `demand_alert`: Stockout warning if product predicted out-of-stock within 12 hours

---

### 6.6 Visualization & Dashboard Layer

#### 6.6.1 Analytics & Monitoring

Analytics and monitoring are handled within the Streamlit dashboard using Plotly visualizations, with Databricks job monitoring for pipeline observability. No separate analytics stack is required.

**Plotly Charts on Streamlit:**

| Dashboard Section | Visualizations |
|-----------|---------------|
| Pricing Trends | Price by product × platform over time, surge hour heatmap, seasonal price index |
| Delivery Performance | Average delivery time by platform and hour, rain vs clear comparison, weekend vs weekday |
| Demand Behavior | Order quantity trends by category, top 10 revenue products, category revenue breakdown |
| Stock & Availability | Availability rate gauges per platform, stockout frequency by product, peak hour stockout rate |
| Platform Intelligence | Platform scorecards, weekly average price index, discount frequency comparison |

#### 6.6.2 Streamlit Application

Built with Streamlit and deployed on Streamlit Community Cloud. Connects to the curated S3 zone (`jayanagar/` prefix) via `boto3`. The location label "Jayanagar, Bangalore" is displayed as a fixed context badge throughout — there is no city selector since the system is single-location.

**Page 1 — Product Search & Comparison**

- Search bar with autocomplete from canonical product registry
- **Preference Sliders Panel** (sticky sidebar):
  - 🏷️ **Price** (0–100, default: 50)
  - 🚴 **Delivery Time** (0–100, default: 30)
  - ⭐ **Product Quality** (0–100, default: 20)
  - Live weight display: e.g. "Price 50% · Delivery 30% · Quality 20%"
- Side-by-side comparison table: Blinkit vs. Zepto vs. Swiggy Instamart
  - Columns: Price, Per-unit price, Delivery ETA, Availability, Rating, Weighted Composite Score
  - 🏆 Winner badge on the recommended platform column
- 7-day historical price trend chart (Plotly)
- Edge case: All sliders at 0 → equal weights (33/33/34)

**Page 2 — Category Trends**

- Category selector with Preference Sliders Panel
- Top 10 trending products (from MLlib trend classifier)
- Demand forecast chart (24h / 72h horizon from GBTRegressor)

**Page 3 — Demand Forecasting Dashboard**

- Product-level forecast timeline
- Predicted stockout alerts
- Historical vs. predicted demand overlay

**Page 4 — Voice Query Center**

- Primary entry point for the Deepgram voice interface
- Microphone button → transcription → recommendation → spoken result
- Transcription displayed for transparency
- Text input fallback always visible

---

### 6.7 Voice Interaction Layer

#### 6.7.1 Overview

The voice feature uses **Deepgram API** for speech-to-text transcription. A user speaks a query into the browser microphone, Deepgram transcribes it, a lightweight NLP parser extracts intent and product entities, the scoring engine fetches and ranks platforms, and the result is rendered on the dashboard with a spoken response via gTTS or pyttsx3.

This is a practical, college-scale implementation — lightweight, with no dependency on large agentic AI frameworks.

#### 6.7.2 Voice Pipeline

```
[User speaks into browser microphone]
        │
        ▼ (audio stream)
[Deepgram API — Speech-to-Text]
  └── Transcribes audio to text
      Deepgram Nova-2 model, pre-recorded or live audio
        │
        ▼ (transcript text)
[Python NLP Parser — keyword matching + regex]
  ├── Intent: compare | find_cheapest | check_delivery | trending | recommend
  ├── Product entity: matched against 51-product catalogue
  └── Pack size (if mentioned)
        │
        ▼ (parsed intent object)
[Data Fetch — reads curated Parquet from S3 via boto3]
        │
        ▼ (product data)
[Preference-Weighted Scoring Engine]
  └── Applies current slider weights
        │
        ▼ (ranked result)
[Response text generation — template-based]
  e.g. "For Milk 500ml in Jayanagar, Zepto is best — ₹31, 11-minute delivery."
        │
        ▼
[Dashboard re-renders with result]
        │
        ▼
[gTTS / pyttsx3 — Text-to-Speech]
  └── Spoken response played in browser
        │
        ▼
[Done — full cycle < 10 seconds]
```

#### 6.7.3 Deepgram Integration Details

| Component | Technology | Details |
|-----------|-----------|---------|
| Speech-to-Text | Deepgram Nova-2 API | REST API call with audio file or browser mic stream |
| Intent parsing | Python (regex + keyword matching) | Lightweight; matches against product catalogue |
| Response generation | Template strings | 1–2 sentence spoken summary |
| Text-to-Speech | gTTS or pyttsx3 | Converts recommendation text to audio |

```python
# Deepgram STT example
from deepgram import DeepgramClient

deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)
response = deepgram.listen.prerecorded.v("1").transcribe_file(
    {"buffer": audio_bytes, "mimetype": "audio/wav"},
    {"model": "nova-2", "language": "en-IN"}
)
transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
```

#### 6.7.4 Supported Voice Intents

| Intent | Example Query | System Response |
|--------|--------------|-----------------|
| `recommend` | "Where should I order milk from?" | Best platform for default pack size |
| `find_cheapest` | "Which is cheapest for 500g paneer?" | Cheapest platform for that product and pack |
| `check_delivery` | "Which platform is fastest right now?" | Fastest delivery platform overall |
| `trending` | "What's trending today?" | Top 5 trending products from ML classifier |
| `compare` | "Compare Blinkit and Zepto for chips" | Side-by-side for those two platforms only |

#### 6.7.5 Fallback Handling

- If Deepgram confidence < 0.75, prompt user to rephrase — example queries shown on screen
- Text input box always visible as fallback alongside the microphone button
- Voice interface degrades gracefully if browser does not support microphone access
- On Deepgram API failure, transcript field is left empty and text input is focused automatically

---

## 7. Functional Requirements

### 7.1 Data Generation

| ID | Requirement |
|----|-------------|
| FR-DG-01 | The simulator shall generate data for all 51 products across Blinkit, Zepto, and Swiggy Instamart at **15-minute intervals** |
| FR-DG-02 | All generated records shall carry `location = "Jayanagar, Bangalore"` |
| FR-DG-03 | Each record shall include price, delivery time, availability, rating, and all temporal and environmental context fields |
| FR-DG-04 | The simulator shall apply time-of-day, weather, seasonal, weekend, and festival multipliers |
| FR-DG-05 | The simulator shall produce platform-differentiated schemas (distinct column names and pricing profiles per platform) |
| FR-DG-06 | The simulator shall operate within 8 GB RAM using chunked batch writing (peak ≤ 200 MB per chunk) |
| FR-DG-07 | The dataset is pre-generated in full before pipeline execution — it is not dynamically produced at runtime |
| FR-DG-08 | Generated Parquet files shall be uploaded to the `jayanagar/` S3 prefix before processing begins |

### 7.2 Data Processing

| ID | Requirement |
|----|-------------|
| FR-DP-01 | PySpark shall normalise all three platform schemas into a single unified canonical schema |
| FR-DP-02 | The system shall parse Swiggy's string delivery time format ("15 mins") into an integer |
| FR-DP-03 | The system shall normalise stock availability columns across all three platforms to a boolean |
| FR-DP-04 | The system shall compute per-unit prices for all products across all pack sizes |
| FR-DP-05 | The unified DataFrame shall be deduplicated on (record_id, platform, snapshot_time) |

### 7.3 Machine Learning

| ID | Requirement |
|----|-------------|
| FR-ML-01 | The demand forecasting model (GBTRegressor) shall produce 24h and 72h horizon predictions |
| FR-ML-02 | The trending product classifier (RandomForestClassifier) shall label products as trending / stable / declining |
| FR-ML-03 | Train/test split shall be time-based: training on April–May, testing on June |
| FR-ML-04 | GBTRegressor MAPE shall be ≤ 20% on the test set |
| FR-ML-05 | RandomForestClassifier weighted F1-score shall be ≥ 0.80 on the test set |
| FR-ML-06 | Trained models shall be saved to S3 models/ zone for reuse |

### 7.4 Dashboard & Monitoring

| ID | Requirement |
|----|-------------|
| FR-DB-01 | The Streamlit dashboard shall display "Jayanagar, Bangalore" as a fixed location context |
| FR-DB-02 | The dashboard shall provide three preference sliders: Price, Delivery Time, and Product Quality |
| FR-DB-03 | Slider values shall be normalised to weights summing to 1.0 before scoring |
| FR-DB-04 | The comparison table and recommendation shall update in real time on every slider change |
| FR-DB-05 | All sliders at 0 shall fall back to equal weights (33/33/34) |
| FR-DB-06 | Plotly charts shall visualise pricing trends, delivery performance, and demand behavior from Spark-processed S3 outputs |
| FR-DB-07 | Databricks monitoring shall track Spark job durations and pipeline errors |

### 7.5 Voice Interface

| ID | Requirement |
|----|-------------|
| FR-VA-01 | The system shall use Deepgram API (Nova-2) for speech-to-text transcription |
| FR-VA-02 | All 5 voice intents defined in Section 6.7.4 shall be supported |
| FR-VA-03 | Voice query results shall use the current dashboard slider weights |
| FR-VA-04 | The full cycle from end of speech to spoken response shall complete in < 10 seconds |
| FR-VA-05 | The transcribed text shall be displayed on screen for transparency |
| FR-VA-06 | On Deepgram API failure, the text input box shall be focused for manual fallback |
| FR-VA-07 | gTTS or pyttsx3 shall convert the recommendation text to spoken audio |

---

## 8. Non-Functional Requirements

| Category | Requirement |
|----------|-------------|
| **Performance** | Dashboard page load (P95) ≤ 3 seconds for cached data |
| **Scalability** | Architecture supports multi-million scalable records; designed for easy extension |
| **Availability** | Dashboard uptime SLA: 99.5% monthly |
| **Data Freshness** | Data is pre-generated; dashboard reflects the latest processed S3 outputs |
| **Voice Latency** | Full voice cycle (speech end → spoken response): < 10 seconds |
| **Reliability** | Simulator failure on one platform shall not affect processing of others |
| **Maintainability** | All Spark jobs, ML pipelines, and simulator code shall be version-controlled in Git |
| **Observability** | Pipeline stages emit metrics to Databricks monitoring; Spark job durations and errors tracked |
| **Cost** | S3 storage cost shall not exceed ₹2,000/month for processed zone at Jayanagar data volumes |

---

## 9. Data Schema & Models

### 9.1 Unified Canonical Schema (Post-Normalisation)

| Field | Type | Source (Blinkit / Zepto / Swiggy) |
|-------|------|----------------------------------|
| `record_id` | UUID string | `record_id` / `txn_id` / `order_ref` |
| `snapshot_time` | timestamp | `snapshot_time` / `recorded_at` / `timestamp` |
| `location` | string | `"Jayanagar, Bangalore"` (all platforms, fixed) |
| `product_name` | string | `product_name` / `item_name` / `product` |
| `category` | string | `product_category` / `item_type` / `category` |
| `pack_size` | integer | `pack_size` / `quantity_ml_g` / `pack_qty` |
| `pack_unit` | string | `pack_unit` / `quantity_unit` / `unit` |
| `platform` | string | Injected: `blinkit` \| `zepto` \| `swiggy` |
| `mrp` | float | `cost` / `mrp` / `mrp` |
| `discount_pct` | float | `deal_discount_pct` / `discount_percent` / `discount_pct` |
| `effective_price` | float | `final_cost` / `selling_price` / `offer_price` |
| `price_per_unit` | float | `cost_per_unit` / `price_per_unit` / `per_unit_price` |
| `in_stock` | boolean | `stock_status=available` / `available=yes` / `in_stock` |
| `stock_remaining` | integer | `stock_remaining` / `stock_remaining` / `stock_remaining` |
| `units_ordered` | integer | `units_ordered` / `qty_ordered` / `num_ordered` |
| `revenue_inr` | float | `gross_revenue` / `revenue_inr` / `total_revenue` |
| `delivery_minutes` | integer | `mins_to_deliver` / `eta_mins` / `delivery_time` (parsed) |
| `rating` | float | `customer_rating` / `app_rating` / `rating` |
| `hour_of_day` | integer | `hour_of_day` / `order_hour` / `hour` |
| `is_weekend` | boolean | `is_weekend` / `weekend_flag` (cast) / `is_weekend` |
| `is_peak_hour` | boolean | `is_rush_hour` / `peak_slot` (cast) / `is_peak_hour` |
| `weather` | string | `weather_condition` / `weather` / `weather` |
| `is_raining` | boolean | `raining` / `is_raining` / `is_raining` |
| `seasonal_factor` | float | `seasonal_factor` / `price_seasonal_mult` / `seasonal_multiplier` |
| `bulk_discount_pct` | float | `bulk_saving_pct` / `bulk_discount_applied` / `bulk_discount` |
| `festival_demand_mult` | float | `festival_demand` / `festival_demand` / `festival_demand` — normalise from row builder name |

### 9.2 Demand Forecast Schema

```json
{
  "canonical_product_id": "string",
  "platform": "string",
  "location": "Jayanagar, Bangalore",
  "forecast_generated_at": "timestamp",
  "forecast_horizon_hours": "integer",
  "predicted_demand_index": "float",
  "confidence_lower": "float",
  "confidence_upper": "float",
  "trend_label": "string (trending | stable | declining)",
  "stockout_risk": "string (high | medium | low)",
  "model_version": "string"
}
```

### 9.3 Platform Comparison Schema (with User Weights)

```json
{
  "canonical_product_id": "string",
  "product_name": "string",
  "location": "Jayanagar, Bangalore",
  "comparison_timestamp": "timestamp",
  "user_weights": {
    "price": "float",
    "delivery": "float",
    "quality": "float"
  },
  "voice_triggered": "boolean",
  "platforms": [
    {
      "platform": "string",
      "price": "float",
      "price_per_unit": "float",
      "delivery_minutes": "integer",
      "is_available": "boolean",
      "rating": "float",
      "availability_rate_7d": "float",
      "price_score": "float",
      "delivery_score": "float",
      "quality_score": "float",
      "composite_score": "float"
    }
  ],
  "recommended_platform": "string",
  "best_price_platform": "string",
  "fastest_delivery_platform": "string",
  "highest_quality_platform": "string"
}
```

---

## 10. Technology Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| **Data Generation** | Python 3.11, pandas, numpy, pyarrow | `dataset.py` simulator; pre-generated; fixed to Jayanagar, Bangalore |
| **Storage** | Amazon S3, Parquet | Three-zone lake; all data under `jayanagar/` prefix |
| **Processing** | Apache Spark 3.5 (PySpark) on Databricks Community Edition | Batch processing; schema normalisation and analytics |
| **ML Framework** | Apache Spark MLlib | GBTRegressor + RandomForestClassifier |
| **Analytics & Monitoring** | Plotly (in Streamlit) | Pricing trends, delivery performance, demand behavior charts |
| **Voice STT** | Deepgram API (Nova-2) | Speech-to-text; en-IN language model |
| **Voice TTS** | gTTS / pyttsx3 | Text-to-speech for spoken recommendations |
| **Dashboard** | Streamlit 1.35+, Plotly 5.x | Streamlit Community Cloud |
| **Data Access** | boto3, pandas, pyarrow | In Streamlit app; reads curated S3 zone |
| **Monitoring** | Databricks monitoring | Spark job metrics and pipeline observability |
| **CI/CD** | GitHub Actions | Lint, test, deploy on push to main |
| **Secrets** | AWS Secrets Manager | All credentials including Deepgram API key |
| **Version Control** | Git / GitHub | Monorepo structure |

---

## 11. Platform Comparison Logic

### 11.1 Preference-Weighted Composite Score

```
Step 1: Read current slider values (price_s, delivery_s, quality_s)
        Normalise:
          w_price    = price_s    / (price_s + delivery_s + quality_s)
          w_delivery = delivery_s / (price_s + delivery_s + quality_s)
          w_quality  = quality_s  / (price_s + delivery_s + quality_s)

Step 2: Fetch Jayanagar data for product P from curated S3 zone

Step 3: Normalise each dimension to [0, 1] across all platforms
  Price Score(p)    = 1 - (price(p) - min_price) / (max_price - min_price)
  Delivery Score(p) = 1 - (delivery(p) - min_time) / (max_time - min_time)
  Quality Score(p)  = 0.5 × (rating(p) / 5.0) + 0.5 × availability_rate_7d(p)

Step 4: Weighted composite
  composite_score = w_price × Price Score
                  + w_delivery × Delivery Score
                  + w_quality × Quality Score

Step 5: Rank platforms descending by composite_score
Step 6: Surface recommended_platform (highest score)
```

### 11.2 Edge Cases

| Scenario | Handling |
|----------|----------|
| Product unavailable on a platform | Quality score = 0; platform ranked last |
| Only one platform has the product | Still ranked; labeled "Only available here" |
| All sliders at 0 | Equal weights (33/33/34) |
| Delivery time unusually high (late-night + rain) | Capped at 60 mins by simulator; treated as valid |
| Rating data missing | Platform-level average rating used |

---

## 12. API & Integration Specifications

### 12.1 External Integrations

| Integration | Purpose | Auth Method |
|-------------|---------|-------------|
| AWS S3 | Read/write Parquet files | IAM Role |
| Databricks REST API | Trigger on-demand Spark jobs | Personal Access Token (PAT) |
| Deepgram API | Speech-to-text transcription | API key (stored in AWS Secrets Manager) |


### 12.2 Deepgram API Call Pattern

```python
from deepgram import DeepgramClient

deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)

# Transcribe audio from browser microphone recording
response = deepgram.listen.prerecorded.v("1").transcribe_file(
    {"buffer": audio_bytes, "mimetype": "audio/wav"},
    {
        "model": "nova-2",
        "language": "en-IN",
        "smart_format": True,
    }
)
transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
confidence = response["results"]["channels"][0]["alternatives"][0]["confidence"]
```

### 12.3 Databricks Job API Trigger

```
POST https://<databricks-workspace>/api/2.1/jobs/run-now
Body: { "job_id": <spark_job_id> }
Headers: Authorization: Bearer <PAT>
```

---

## 13. Security & Compliance

### 13.1 Data Security

- All S3 buckets are private; access only via IAM roles
- S3 data encrypted at rest (SSE-S3, AES-256)
- All inter-service communication uses TLS 1.2+
- Deepgram API key stored in AWS Secrets Manager and injected at runtime
- Voice audio sent to Deepgram is not stored beyond the request lifecycle

### 13.2 Ethical & Legal Considerations

- The system uses a data simulator — **no web scraping or API access to any platform is performed**
- No user behavioral data is collected without explicit consent
- The system does not claim any affiliation with Blinkit, Zepto, or Swiggy Instamart
- Scoped to Jayanagar, Bangalore only — no inference or extrapolation to other locations

### 13.3 Access Control

| Role | Permissions |
|------|-------------|
| Pipeline Service Account | S3 read/write (all zones, jayanagar/ prefix), Databricks Jobs |
| Streamlit App Service Account | S3 read-only (curated/jayanagar/ only) |
| Data Engineer | Full Databricks workspace, S3 all zones |
| Dashboard User (Public) | Read-only Streamlit app; no direct S3 access |

---

## 14. Deployment Strategy

### 14.1 Environment Structure

| Environment | Purpose | Infrastructure |
|-------------|---------|----------------|
| `dev` | Development and unit testing | Single-node Databricks cluster, small S3 buckets |
| `staging` | Integration testing and QA | Multi-node cluster, full pipeline, private Streamlit app |
| `production` | Live system | Databricks, public Streamlit app |

### 14.2 CI/CD Pipeline

```
[Developer pushes to feature branch]
        │
        ▼
[GitHub Actions — CI]
  ├── Python linting (ruff) & type checking (mypy)
  ├── Unit tests for simulator, normalizers, scoring logic
  ├── Spark job unit tests (pyspark local mode)
  └── ML model smoke tests
        │
        ▼
[Merge to main → CD to Staging]
  ├── Deploy Spark job bundles to Databricks (staging)
  ├── Deploy Streamlit app to staging URL
  └── End-to-end pipeline validation
        │
        ▼
[Manual approval gate]
        │
        ▼
[CD to Production]
  ├── Deploy Databricks job bundles
  ├── Deploy Streamlit app (Streamlit Community Cloud)
  └── Smoke tests on live environment
```

---

## 15. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Deepgram STT accuracy below threshold on Bangalore-accented English | Medium | Medium | Use en-IN language model; provide text fallback always visible |
| Deepgram API key quota exhausted under heavy voice usage | Low | Medium | Rate-limit voice requests per session; cache repeated queries |
| Simulator produces patterns that confuse ML models | Low | Medium | Validate against known Jayanagar price ranges; tune multiplier parameters |
| Spark job latency on large dataset | Medium | Medium | Optimize Parquet partitioning; use Databricks autoscaling |
| ML model accuracy degrades on edge cases (festivals, extreme weather) | Medium | Medium | Simulator audit trail fields allow feature debugging |
| Streamlit Community Cloud rate limits under high traffic | Low | Medium | Aggressive caching; fallback to self-hosted if needed |

| S3 costs exceed budget | Low | Low | Lifecycle policies; Parquet columnar format minimises read costs |

---

## 16. Success Metrics & KPIs

### 16.1 System Performance KPIs

| KPI | Target | Measurement |
|-----|--------|-------------|
| Dataset scale | Multi-million scalable records across 3 platforms | Output file row counts |
| Cross-platform schema normalisation | All 3 schemas unified without data loss | Unit test validation |
| Demand forecast MAPE | ≤ 20% | Spark MLlib evaluation |
| Trending classifier F1-score | ≥ 0.80 | Spark MLlib evaluation |
| Dashboard P95 load time | ≤ 3 seconds | Streamlit analytics |
| Voice pipeline cycle time | < 10 seconds | Client-side timing logs |


### 16.2 Product / User KPIs

| KPI | Target (3 months post-launch) | Measurement |
|-----|-------------------------------|-------------|
| Monthly Active Users (Jayanagar) | 2,000 | Streamlit analytics |
| Average session duration | ≥ 4 minutes | Streamlit analytics |
| Voice command usage rate | ≥ 40% of sessions | Custom event logging |
| Slider interaction rate | ≥ 60% of sessions | Custom event logging |
| Product searches per session | ≥ 2.5 | Custom event logging |
| Return user rate (7-day) | ≥ 30% | Streamlit analytics |

---

## 17. Glossary

| Term | Definition |
|------|-----------| 
| **Quick-commerce** | Grocery and essentials delivery with a promise of 10–30 minute delivery times |
| **Blinkit** | Quick-commerce platform by Zomato, operating in Jayanagar, Bangalore |
| **Zepto** | Quick-commerce startup operating in Jayanagar, Bangalore |
| **Swiggy Instamart** | Quick-commerce vertical of Swiggy food delivery, operating in Jayanagar, Bangalore |
| **dataset.py** | The Python data simulator that pre-generates realistic product data for all three platforms at 15-minute intervals, fixed to Jayanagar, Bangalore |
| **Deepgram API** | Cloud-based speech-to-text API used in this project for voice query transcription |
| **Voice Interface** | The browser-based voice input feature powered by Deepgram STT and gTTS/pyttsx3 TTS |

| **Preference Sliders** | Three interactive dashboard controls (Price, Delivery Time, Product Quality) controlling composite scoring weights |
| **Parquet** | Columnar binary file format optimized for analytical queries; natively produced by the simulator via PyArrow |
| **Databricks** | Cloud-based platform for running Apache Spark at scale |
| **MLlib** | Apache Spark's built-in distributed machine learning library |
| **GBTRegressor** | Gradient Boosted Trees Regressor — MLlib model used for demand forecasting |
| **RandomForestClassifier** | MLlib model used for trending product classification |
| **MAPE** | Mean Absolute Percentage Error — a measure of forecasting accuracy |
| **Canonical Product** | A unified product record mapping the same physical product across multiple platform-specific listings |
| **Composite Score** | A user-weighted combination of price, delivery, and quality scores used to rank platforms |
| **Seasonal Multiplier** | A per-month factor applied by the simulator to specific produce items to model realistic price variation |
| **Festival Demand Multiplier** | A demand and price factor applied by the simulator during major Indian festivals |
| **STT** | Speech-to-Text — converts spoken audio to written text (handled by Deepgram API) |
| **TTS** | Text-to-Speech — converts written text to spoken audio (handled by gTTS or pyttsx3) |
| **Pre-generated dataset** | A dataset fully created before pipeline execution, as opposed to dynamically produced at runtime |

---

*Document prepared for internal project planning and development reference.*
*All platform names (Blinkit, Zepto, Swiggy Instamart) are trademarks of their respective owners.*
*This system is an independent analytics tool with no affiliation with any named platform.*
*Location scope: Jayanagar, Bangalore only.*
