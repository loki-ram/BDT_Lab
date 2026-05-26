"""
streamlit_app.py — QuickCommerce Platform Recommender Dashboard (EC2-Optimised)

Key changes vs original:
  • Validates parquet files before reading (catches 0-byte / missing files)
  • Streams parquet in small batches with hard row caps — never loads full 740 MB
  • Catalog built from a 200 k-row scan instead of the full file
  • Categorical dtypes throughout (saves ~60 % RAM vs object columns)
  • Chart samples capped at 50 k rows
  • All failures are graceful — the app keeps running even if S3 files are absent
"""

import os, sys, json, glob
from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── S3 env vars must be set BEFORE any boto/s3fs import ──────────────────────
os.environ.setdefault("S3_CONNECT_TIMEOUT", "30")
os.environ.setdefault("S3_READ_TIMEOUT",    "120")
os.environ.setdefault("AWS_DEFAULT_REGION", "eu-north-1")

try:
    import pyarrow.parquet as pq
    import pyarrow.fs      as pafs
    import pyarrow.dataset as pad   # for exclude_invalid_files on Spark folders
except ImportError:
    pq   = None
    pafs = None
    pad  = None

# ── Debug helpers ─────────────────────────────────────────────────────────────
DEBUG = True

def dbg(msg: str):
    if DEBUG:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] DEBUG: {msg}", file=sys.stderr)

dbg("🚀 App init")

# ── Environment detection ─────────────────────────────────────────────────────
IS_DATABRICKS      = "DATABRICKS_RUNTIME_VERSION" in os.environ
IS_STREAMLIT_CLOUD = (
    os.environ.get("STREAMLIT_RUNTIME_VERSION") is not None
    or os.environ.get("STREAMLIT_SERVER_HEADLESS") == "true"
)

# Load AWS creds from st.secrets if not already in env
if not os.environ.get("AWS_ACCESS_KEY_ID"):
    try:
        os.environ["AWS_ACCESS_KEY_ID"]     = st.secrets.get("AWS_ACCESS_KEY_ID", "")
        os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets.get("AWS_SECRET_ACCESS_KEY", "")
        dbg("✅ AWS creds loaded from secrets")
    except Exception as e:
        dbg(f"⚠️  Secrets load failed: {e}")

# ── S3 paths ──────────────────────────────────────────────────────────────────
S3_BASE          = "s3://qcommerce-bdt-cct/parquets"
UNIFIED_PARQUET  = f"{S3_BASE}/unified.parquet"
DEMAND_FORECASTS = f"{S3_BASE}/demand_forecasts.parquet"
TREND_LABELS     = f"{S3_BASE}/trend_labels.parquet"

PLATFORM_COLORS = {"blinkit": "#F8C100", "zepto": "#7B2FF7", "swiggy": "#FC8019"}
PLATFORM_ICONS  = {"blinkit": "🟡",      "zepto": "🟣",      "swiggy": "🟠"}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuickCommerce Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS (unchanged from original) ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.main-title{font-size:2.2rem;font-weight:800;background:linear-gradient(135deg,#667eea,#764ba2);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px}
.sub-title{font-size:1.05rem;color:#94a3b8;margin-bottom:20px}
.rank-card{background:linear-gradient(135deg,#1e1e2e,#2a2a3e);border-radius:16px;padding:24px;
  margin-bottom:16px;border-left:5px solid;position:relative;overflow:hidden;
  box-shadow:0 4px 24px rgba(0,0,0,.15)}
.rank-badge{display:inline-flex;align-items:center;justify-content:center;width:44px;height:44px;
  border-radius:12px;font-size:1.3rem;font-weight:800;color:#fff;margin-bottom:8px}
.rank-1{background:linear-gradient(135deg,#f59e0b,#d97706)}
.rank-2{background:linear-gradient(135deg,#94a3b8,#64748b)}
.rank-3{background:linear-gradient(135deg,#b45309,#92400e)}
.platform-name{font-size:1.4rem;font-weight:700;color:#f1f5f9}
.score-big{font-size:2.4rem;font-weight:800;color:#a78bfa}
.score-label{font-size:.8rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px}
.metric-row{display:flex;gap:16px;margin-top:12px;flex-wrap:wrap}
.metric-pill{background:rgba(255,255,255,.06);border-radius:10px;padding:8px 14px;
  flex:1;min-width:100px;text-align:center}
.metric-pill .val{font-size:1.1rem;font-weight:700;color:#e2e8f0}
.metric-pill .lbl{font-size:.7rem;color:#94a3b8;text-transform:uppercase}
.insight-box{background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:12px;
  padding:16px;margin-top:12px;border:1px solid rgba(167,139,250,.2)}
.insight-box .title{font-weight:700;color:#a78bfa;margin-bottom:6px}
.insight-box .text{color:#cbd5e1;font-size:.9rem;line-height:1.5}
div[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f0f1a,#1a1a2e)}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# LOW-LEVEL PARQUET UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _s3fs():
    """Reusable S3FileSystem — created once per process."""
    if pafs is None:
        return None
    region = os.environ.get("AWS_DEFAULT_REGION") or None
    return pafs.S3FileSystem(region=region)


def _pq_path(path: str):
    """Return (filesystem_or_None, bare_path)."""
    if path.startswith("s3://"):
        return _s3fs(), path.replace("s3://", "", 1)
    return None, path


def _file_is_valid(path: str) -> bool:
    """
    Return True if the path points to a readable parquet dataset —
    either a single .parquet file or a Spark-partitioned folder like
    unified.parquet/ containing part-NNNNN-tid-<uuid>.c000.snappy.parquet files.

    Uses pyarrow.dataset with exclude_invalid_files so _SUCCESS and
    _committed_ Spark metadata files are ignored automatically.
    """
    if pad is None or pafs is None:
        return False
    try:
        fs, bare = _pq_path(path)
        ds = pad.dataset(bare, filesystem=fs, format="parquet",
                         exclude_invalid_files=True)
        # get_fragments() is lazy — it just lists matching files
        frags = list(ds.get_fragments())
        return len(frags) > 0
    except Exception as e:
        dbg(f"_file_is_valid({path}): {type(e).__name__}: {e}")
        return False


@st.cache_data(ttl=3600, max_entries=20)
def _stream_parquet(
    path: str,
    columns: Tuple[str, ...],
    max_rows: int,
    filter_categories: Optional[Tuple[str, ...]] = None,
    filter_products: Optional[Tuple[str, ...]] = None,
    batch_size: int = 65_536,
    seed: int = 42,
) -> Optional[pd.DataFrame]:
    """
    Stream a Parquet file in batches and return AT MOST `max_rows` rows.

    Strategy
    --------
    * With NO filters  → scan up to `scan_budget` rows then stop; subsample each
      batch so memory stays bounded even before concat.
    * With filters     → scan more of the file but still cap the output.

    This means the app never holds the full 740 MB file in RAM.
    """
    if not _file_is_valid(path):
        dbg(f"⛔ Skipping {path}: file missing or 0 bytes")
        return None
    if pq is None:
        dbg("pyarrow not available")
        return None

    cats_set  = set(filter_categories or ())
    prods_set = set(filter_products  or ())
    rng       = np.random.default_rng(seed)

    # How many source rows to scan before giving up
    scan_budget = (
        max(800_000, max_rows * 20) if not (cats_set or prods_set)
        else max(3_000_000, max_rows * 60)
    )

    chunks, collected, scanned = [], 0, 0
    try:
        fs, bare = _pq_path(path)
        # ParquetDataset handles both single files AND partitioned directories.
        # We pass the dataset to iter_batches via a Scanner for memory efficiency.
        # pyarrow.dataset handles Spark folders with _SUCCESS / metadata files cleanly
        dataset = pad.dataset(bare, filesystem=fs, format="parquet",
                              exclude_invalid_files=True)
        scanner = dataset.scanner(columns=list(columns), batch_size=batch_size)
        for batch in scanner.to_batches():
            scanned += batch.num_rows
            # RecordBatch.to_pandas does not accept use_threads
            dfb = batch.to_pandas(strings_to_categorical=True)

            if cats_set  and "category"     in dfb.columns: dfb = dfb[dfb["category"    ].isin(cats_set )]
            if prods_set and "product_name" in dfb.columns: dfb = dfb[dfb["product_name"].isin(prods_set)]
            if dfb.empty:
                if scanned >= scan_budget: break
                continue

            remaining = max_rows - collected
            if remaining <= 0: break

            # cap per-batch contribution to spread sampling across the file
            per_batch_cap = max(2_000, max_rows // 50) if not (cats_set or prods_set) else remaining
            take = min(len(dfb), per_batch_cap, remaining)
            if take < len(dfb):
                dfb = dfb.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1)), ignore_index=True)

            chunks.append(dfb)
            collected += len(dfb)
            if collected >= max_rows or scanned >= scan_budget:
                break

        if not chunks:
            return pd.DataFrame(columns=list(columns))
        df = pd.concat(chunks, ignore_index=True)
        for col in ("snapshot_time", "date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        dbg(f"✅ {path.split('/')[-1]}: {len(df):,} rows (scanned {scanned:,})")
        return df
    except Exception as e:
        dbg(f"❌ _stream_parquet({path}): {type(e).__name__}: {str(e)[:200]}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# CATALOG (sidebar lists)  — built from a tiny scan
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def build_catalog(scan_rows: int = 200_000) -> Optional[dict]:
    """
    Build category / product lists by scanning only `scan_rows` source rows.
    Returns None if the file is missing/empty, so the caller can show a helpful
    error instead of crashing.
    """
    if not _file_is_valid(UNIFIED_PARQUET):
        dbg("⛔ build_catalog: unified.parquet missing or 0 bytes")
        return None

    cols = ("category", "product_name", "units_ordered")
    categories: set           = set()
    product_counts: Counter   = Counter()
    per_cat: dict             = defaultdict(Counter)
    scanned = 0

    try:
        fs, bare = _pq_path(UNIFIED_PARQUET)
        dataset = pad.dataset(bare, filesystem=fs, format="parquet",
                              exclude_invalid_files=True)
        scanner = dataset.scanner(columns=list(cols), batch_size=65_536)
        for batch in scanner.to_batches():
            scanned += batch.num_rows
            dfb = batch.to_pandas(strings_to_categorical=True)
            dfb = dfb.dropna(subset=["category", "product_name"])
            if dfb.empty:
                if scanned >= scan_rows: break
                continue

            categories.update(dfb["category"].unique().tolist())

            if pd.api.types.is_numeric_dtype(dfb.get("units_ordered", pd.Series(dtype=float))):
                for (cat, prod), v in dfb.groupby(["category", "product_name"])["units_ordered"].sum().items():
                    product_counts[prod]   += float(v)
                    per_cat[cat][prod]     += float(v)
            else:
                for prod, v in dfb["product_name"].value_counts().items():
                    product_counts[prod] += int(v)
                for cat, sub in dfb.groupby("category"):
                    for prod, v in sub["product_name"].value_counts().items():
                        per_cat[cat][prod] += int(v)

            if scanned >= scan_rows:
                break

        return {
            "categories":           sorted(c for c in categories if c),
            "top_products":         [p for p, _ in product_counts.most_common(5_000)],
            "products_by_category": {cat: [p for p, _ in cnt.most_common(1_500)]
                                     for cat, cnt in per_cat.items()},
        }
    except Exception as e:
        dbg(f"❌ build_catalog: {type(e).__name__}: {str(e)[:200]}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# TYPED LOADERS  (thin wrappers around _stream_parquet)
# ═════════════════════════════════════════════════════════════════════════════

UNIFIED_COLS = (
    "platform", "category", "product_name",
    "delivery_minutes", "effective_price", "rating",
    "units_ordered", "stock_remaining", "in_stock", "snapshot_time",
)
DEMAND_COLS  = ("platform", "category", "product_name", "predicted_demand", "units_ordered")
TREND_COLS   = ("platform", "category", "product_name", "date",
                "daily_demand", "rolling_avg_7d", "trend_label", "predicted_label_idx")


def load_unified(cats, prods, max_rows):
    return _stream_parquet(UNIFIED_PARQUET, UNIFIED_COLS, max_rows,
                           filter_categories=cats, filter_products=prods)

def load_demand(cats, prods, max_rows=80_000):
    return _stream_parquet(DEMAND_FORECASTS, DEMAND_COLS, max_rows,
                           filter_categories=cats, filter_products=prods)

def load_trends(cats, prods, max_rows=80_000):
    return _stream_parquet(TREND_LABELS, TREND_COLS, max_rows,
                           filter_categories=cats, filter_products=prods)


# ═════════════════════════════════════════════════════════════════════════════
# SCORING
# ═════════════════════════════════════════════════════════════════════════════

def compute_scores(df, w_delivery, w_price, w_quality):
    agg = df.groupby("platform").agg(
        avg_delivery  = ("delivery_minutes",  "mean"),
        avg_price     = ("effective_price",    "mean"),
        avg_rating    = ("rating",             "mean"),
        product_count = ("product_name",      "nunique"),
        total_orders  = ("units_ordered",      "sum"),
        avg_stock     = ("stock_remaining",    "mean"),
        in_stock_pct  = ("in_stock",           "mean"),
    ).reset_index()

    def norm(s, invert=False):
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series([50.0] * len(s), index=s.index)
        n = (s - mn) / (mx - mn) * 100
        return 100 - n if invert else n

    agg["delivery_score"] = norm(agg["avg_delivery"], invert=True)
    agg["price_score"]    = norm(agg["avg_price"],    invert=True)
    agg["quality_score"]  = norm(agg["avg_rating"],   invert=False)

    tw = max(w_delivery + w_price + w_quality, 1)
    agg["overall_score"] = (
        agg["delivery_score"] * w_delivery
        + agg["price_score"]  * w_price
        + agg["quality_score"] * w_quality
    ) / tw

    agg["in_stock_pct"] = agg["in_stock_pct"] * 100
    agg = agg.sort_values("overall_score", ascending=False).reset_index(drop=True)
    agg["rank"] = range(1, len(agg) + 1)
    return agg


# ═════════════════════════════════════════════════════════════════════════════
# DEMO DATA  — shown when S3 is unavailable
# ═════════════════════════════════════════════════════════════════════════════

def _demo_unified() -> pd.DataFrame:
    """Small synthetic dataset so the UI is never completely broken."""
    rng = np.random.default_rng(0)
    platforms   = ["blinkit", "zepto", "swiggy"]
    categories  = ["Fruits & Vegs", "Dairy", "Snacks", "Beverages", "Personal Care"]
    products    = ["Product A", "Product B", "Product C", "Product D", "Product E"]
    n = 3_000
    return pd.DataFrame({
        "platform":         rng.choice(platforms, n),
        "category":         rng.choice(categories, n),
        "product_name":     rng.choice(products, n),
        "delivery_minutes": rng.uniform(8, 45, n),
        "effective_price":  rng.uniform(20, 500, n),
        "rating":           rng.uniform(3.0, 5.0, n),
        "units_ordered":    rng.integers(1, 200, n).astype(float),
        "stock_remaining":  rng.integers(0, 500, n).astype(float),
        "in_stock":         rng.choice([0, 1], n, p=[0.1, 0.9]).astype(float),
        "snapshot_time":    pd.date_range("2024-01-01", periods=n, freq="1min"),
    })

def _demo_catalog() -> dict:
    df = _demo_unified()
    return {
        "categories":           sorted(df["category"].unique().tolist()),
        "top_products":         df["product_name"].unique().tolist(),
        "products_by_category": {cat: df[df["category"] == cat]["product_name"].unique().tolist()
                                 for cat in df["category"].unique()},
    }


# ═════════════════════════════════════════════════════════════════════════════
# APP  — sidebar
# ═════════════════════════════════════════════════════════════════════════════

dbg("Building catalog…")
with st.spinner("⏳ Connecting to S3 and building catalog…"):
    catalog = build_catalog()

USING_DEMO = False
if catalog is None:
    st.warning(
        "⚠️  **Live data unavailable** — `unified.parquet` is missing or empty on S3.  "
        "Showing a **demo dataset** so you can explore the UI.  \n\n"
        "To fix: upload a valid parquet file to `s3://qcommerce-bdt-cct/parquets/unified.parquet`."
    )
    catalog = _demo_catalog()
    USING_DEMO = True
    dbg("Using demo data")

# ---------- sidebar controls ----------
st.sidebar.markdown("### ⚙️ Performance")
default_rows  = int(os.environ.get("UNIFIED_SAMPLE_ROWS", "150000"))
sample_rows   = st.sidebar.slider(
    "Max rows to sample",
    min_value=50_000, max_value=400_000,
    value=max(50_000, min(400_000, default_rows)),
    step=50_000,
    help="Cap on rows read from the large Parquet file. Lower = faster & less RAM.",
)

st.sidebar.markdown("### 📂 Category")
all_cats          = catalog.get("categories", [])
selected_cats     = st.sidebar.multiselect("Select Categories", options=all_cats, default=[])

st.sidebar.markdown("### 🛒 Products")
product_query = st.sidebar.text_input("Search Products", value="",
                                      help="Type to filter the product list.")
if selected_cats:
    cands = []
    for cat in selected_cats:
        cands.extend(catalog.get("products_by_category", {}).get(cat, []))
    cands = cands or catalog.get("top_products", [])
else:
    cands = catalog.get("top_products", [])

cands = list(dict.fromkeys(p for p in cands if p))
if product_query.strip():
    q     = product_query.strip().lower()
    cands = [p for p in cands if q in str(p).lower()]
if len(cands) > 2_000:
    st.sidebar.caption(f"Showing first 2,000 of {len(cands):,} products.")
    cands = cands[:2_000]

selected_prods = st.sidebar.multiselect("Select Products", options=cands, default=[])

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚖️ Your Preferences")
st.sidebar.caption("Sliders affect rankings AND the ML analytics below.")
w_delivery = st.sidebar.slider("🚚 Delivery Speed",        0, 10, 5)
w_price    = st.sidebar.slider("💰 Low Price",              0, 10, 5)
w_quality  = st.sidebar.slider("⭐ Product Quality/Rating", 0, 10, 5)

st.sidebar.markdown("---")
tw = max(w_delivery + w_price + w_quality, 1)
st.sidebar.markdown(f"""
| Factor | Weight | Share |
|---|---|---|
| 🚚 Delivery | **{w_delivery}** | {w_delivery/tw*100:.0f}% |
| 💰 Price    | **{w_price}**    | {w_price/tw*100:.0f}%    |
| ⭐ Quality  | **{w_quality}**  | {w_quality/tw*100:.0f}%  |
""")


# ═════════════════════════════════════════════════════════════════════════════
# LOAD MAIN DATA
# ═════════════════════════════════════════════════════════════════════════════

cats_t  = tuple(selected_cats)  or None
prods_t = tuple(selected_prods) or None

if USING_DEMO:
    unified_df = _demo_unified()
else:
    with st.spinner("📥 Loading data sample from S3…"):
        unified_df = load_unified(cats_t, prods_t, max_rows=int(sample_rows))

if unified_df is None or unified_df.empty:
    st.error("No data loaded. Try broadening your filters or increasing the sample size.")
    st.stop()

# apply sidebar filters (may already be pre-filtered by the loader, but be safe)
filtered = unified_df.copy()
if selected_cats:  filtered = filtered[filtered["category"    ].isin(selected_cats )]
if selected_prods: filtered = filtered[filtered["product_name"].isin(selected_prods)]
if filtered.empty:
    st.warning("No rows match your current filters. Please broaden the selection.")
    st.stop()

# cap chart sample to avoid browser rendering lag
CHART_CAP   = 50_000
chart_df    = filtered.sample(n=CHART_CAP, random_state=42) if len(filtered) > CHART_CAP else filtered
dbg(f"filtered={len(filtered):,}  chart_df={len(chart_df):,}")


# ═════════════════════════════════════════════════════════════════════════════
# HEADER + TOP METRICS
# ═════════════════════════════════════════════════════════════════════════════

if USING_DEMO:
    st.info("🧪 **Demo mode** — displaying synthetic data. Connect S3 to see live results.")

st.markdown('<div class="main-title">🛒 QuickCommerce Platform Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Pick your products · set your priorities · find the best platform</div>', unsafe_allow_html=True)

scores = compute_scores(filtered, w_delivery, w_price, w_quality)

c1, c2, c3, c4 = st.columns(4)
c1.metric("📦 Products Matched",  f"{filtered['product_name'].nunique():,}")
c2.metric("📄 Records (sampled)", f"{len(filtered):,}")
c3.metric("🏆 Top Platform",
          f"{PLATFORM_ICONS.get(scores.iloc[0]['platform'],'')} {scores.iloc[0]['platform'].title()}")
c4.metric("🎯 Top Score",        f"{scores.iloc[0]['overall_score']:.1f} / 100")

st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
# RANKING CARDS
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("### 🏅 Platform Rankings")
for _, row in scores.iterrows():
    rank  = int(row["rank"])
    plat  = row["platform"]
    color = PLATFORM_COLORS.get(plat, "#6366f1")
    st.markdown(f"""
    <div class="rank-card" style="border-left-color:{color};">
      <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
        <div><div class="rank-badge rank-{rank}">#{rank}</div></div>
        <div style="flex:1;min-width:150px;">
          <div class="platform-name">{PLATFORM_ICONS.get(plat,'')} {plat.title()}</div>
        </div>
        <div style="text-align:right;">
          <div class="score-label">Overall Score</div>
          <div class="score-big">{row['overall_score']:.1f}</div>
        </div>
      </div>
      <div class="metric-row">
        <div class="metric-pill"><div class="val">{row['avg_delivery']:.1f} min</div><div class="lbl">Avg Delivery</div></div>
        <div class="metric-pill"><div class="val">₹{row['avg_price']:.1f}</div><div class="lbl">Avg Price</div></div>
        <div class="metric-pill"><div class="val">{row['avg_rating']:.2f} ⭐</div><div class="lbl">Avg Rating</div></div>
        <div class="metric-pill"><div class="val">{row['in_stock_pct']:.1f}%</div><div class="lbl">In Stock</div></div>
        <div class="metric-pill"><div class="val">{int(row['total_orders']):,}</div><div class="lbl">Total Orders</div></div>
      </div>
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SCORE BREAKDOWN CHARTS
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 📊 Score Breakdown")

col_l, col_r = st.columns(2)
with col_l:
    fig = go.Figure()
    for _, row in scores.iterrows():
        plat = row["platform"]
        fig.add_trace(go.Scatterpolar(
            r=[row["delivery_score"], row["price_score"], row["quality_score"]],
            theta=["Delivery Speed", "Low Price", "Quality / Rating"],
            fill="toself", name=plat.title(),
            line_color=PLATFORM_COLORS.get(plat, "#6366f1"), opacity=0.75,
        ))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(visible=True, range=[0,100], showticklabels=False)),
        title="Platform Strength Radar", template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=400, margin=dict(t=50, b=30), legend=dict(orientation="h", y=-0.1),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    bar_rows = []
    for _, row in scores.iterrows():
        p = row["platform"].title()
        bar_rows += [
            {"Platform": p, "Factor": "🚚 Delivery", "Score": row["delivery_score"]},
            {"Platform": p, "Factor": "💰 Price",    "Score": row["price_score"]},
            {"Platform": p, "Factor": "⭐ Quality",  "Score": row["quality_score"]},
        ]
    fig2 = px.bar(pd.DataFrame(bar_rows), x="Platform", y="Score", color="Factor",
                  barmode="group", title="Individual Factor Scores",
                  color_discrete_sequence=["#38bdf8","#34d399","#f472b6"],
                  template="plotly_dark")
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       height=400, margin=dict(t=50,b=30), legend=dict(orientation="h",y=-0.15))
    st.plotly_chart(fig2, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY-LEVEL COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 🔍 Category-Level Platform Comparison")

cat_plat = chart_df.groupby(["category","platform"]).agg(
    avg_price    = ("effective_price",   "mean"),
    avg_delivery = ("delivery_minutes",  "mean"),
    avg_rating   = ("rating",            "mean"),
).reset_index()

col_a, col_b = st.columns(2)
with col_a:
    f3 = px.bar(cat_plat, x="category", y="avg_price", color="platform", barmode="group",
                title="Avg Price by Category & Platform",
                color_discrete_map=PLATFORM_COLORS, template="plotly_dark")
    f3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     xaxis_tickangle=-45, height=420, margin=dict(t=50,b=80))
    st.plotly_chart(f3, use_container_width=True)

with col_b:
    f4 = px.bar(cat_plat, x="category", y="avg_delivery", color="platform", barmode="group",
                title="Avg Delivery Time by Category & Platform",
                color_discrete_map=PLATFORM_COLORS, template="plotly_dark")
    f4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     xaxis_tickangle=-45, height=420, margin=dict(t=50,b=80))
    st.plotly_chart(f4, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PRODUCT-LEVEL TABLE
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 📋 Product-Level Platform Metrics")

prod_plat = (chart_df
    .groupby(["product_name","category","platform"])
    .agg(avg_price=("effective_price","mean"), avg_delivery=("delivery_minutes","mean"),
         avg_rating=("rating","mean"), in_stock_pct=("in_stock","mean"))
    .reset_index()
)
prod_plat["in_stock_pct"]  = (prod_plat["in_stock_pct"] * 100).round(1)
prod_plat["avg_price"]     = prod_plat["avg_price"].round(2)
prod_plat["avg_delivery"]  = prod_plat["avg_delivery"].round(1)
prod_plat["avg_rating"]    = prod_plat["avg_rating"].round(2)

st.dataframe(
    prod_plat.rename(columns={
        "product_name": "Product", "category": "Category", "platform": "Platform",
        "avg_price": "Avg Price (₹)", "avg_delivery": "Delivery (min)",
        "avg_rating": "Rating ⭐", "in_stock_pct": "In Stock %",
    }),
    use_container_width=True, height=400,
)


# ═════════════════════════════════════════════════════════════════════════════
# INSIGHT BOX
# ═════════════════════════════════════════════════════════════════════════════

best = scores.iloc[0]
label = "products" if selected_prods else "categories"
st.markdown(f"""
<div class="insight-box">
  <div class="title">💡 Recommendation Insight</div>
  <div class="text">
    Based on your weights (Delivery <b>{w_delivery}</b> · Price <b>{w_price}</b> ·
    Quality <b>{w_quality}</b>), <b>{best['platform'].title()}</b> is your best pick
    with a score of <b>{best['overall_score']:.1f}/100</b>.<br>
    Delivers in ~<b>{best['avg_delivery']:.0f} min</b> · avg price
    <b>₹{best['avg_price']:.0f}</b> · rating <b>{best['avg_rating']:.2f}⭐</b>
    across your selected {label}.
  </div>
</div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# DEMAND FORECASTING  (GBT)
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 🤖 Demand Forecasting — GBT Model")

demand_df = None if USING_DEMO else load_demand(cats_t, prods_t)

if demand_df is not None and not demand_df.empty:
    if selected_cats:  demand_df = demand_df[demand_df["category"    ].isin(selected_cats )]
    if selected_prods: demand_df = demand_df[demand_df["product_name"].isin(selected_prods)]
    if len(demand_df) > 50_000:
        demand_df = demand_df.sample(50_000, random_state=42)

    plat_score_map = dict(zip(scores["platform"], scores["overall_score"] / 100))
    demand_df = demand_df.copy()
    demand_df["pref_mul"]        = demand_df["platform"].map(plat_score_map).fillna(0.5)
    demand_df["adj_predicted"]   = demand_df["predicted_demand"] * demand_df["pref_mul"]

    st.markdown("""
    <div class="insight-box">
      <div class="title">🌲 GBT Regressor — Model Info</div>
      <div class="text">
        <b>Type:</b> Gradient-Boosted Trees (Spark MLlib) &nbsp;|&nbsp;
        <b>Trees:</b> 15 &nbsp;|&nbsp; <b>Max Depth:</b> 4 &nbsp;|&nbsp; <b>Features:</b> 6<br>
        <b>Columns:</b> platform_idx, category_idx, weather_idx, is_raining_int, in_stock_int, stock_remaining
      </div>
    </div>""", unsafe_allow_html=True)

    dagg = demand_df.groupby("platform").agg(
        avg_actual    = ("units_ordered",   "mean"),
        avg_adj_pred  = ("adj_predicted",   "mean"),
    ).reset_index()

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        fd1 = go.Figure()
        fd1.add_trace(go.Bar(x=dagg["platform"].str.title(), y=dagg["avg_actual"],
                             name="Actual Demand", marker_color="#38bdf8"))
        fd1.add_trace(go.Bar(x=dagg["platform"].str.title(), y=dagg["avg_adj_pred"],
                             name="Adjusted Prediction", marker_color="#a78bfa"))
        fd1.update_layout(title="Avg Demand: Actual vs Predicted", barmode="group",
                          template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=400,
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fd1, use_container_width=True)

    with col_d2:
        top_prods = (demand_df.groupby("product_name")
                    .agg(avg_adj_pred=("adj_predicted","mean"), avg_actual=("units_ordered","mean"))
                    .reset_index().nlargest(15,"avg_adj_pred"))
        fd2 = go.Figure()
        fd2.add_trace(go.Bar(y=top_prods["product_name"], x=top_prods["avg_actual"],
                             name="Actual", orientation="h", marker_color="#38bdf8"))
        fd2.add_trace(go.Bar(y=top_prods["product_name"], x=top_prods["avg_adj_pred"],
                             name="Adjusted Predicted", orientation="h", marker_color="#a78bfa"))
        fd2.update_layout(title="Top 15 Products by Weighted Demand", barmode="group",
                          template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=400,
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fd2, use_container_width=True)

    pivot = (demand_df.groupby(["category","platform"])["adj_predicted"].mean()
             .reset_index()
             .pivot(index="category", columns="platform", values="adj_predicted")
             .fillna(0))
    fheat = px.imshow(pivot.values, x=pivot.columns.str.title(), y=pivot.index,
                      color_continuous_scale="Viridis", aspect="auto",
                      title="Adjusted Demand Heatmap (Category × Platform)",
                      labels={"color":"Adj Demand"})
    fheat.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)", height=380)
    st.plotly_chart(fheat, use_container_width=True)
else:
    st.info("Demand forecast data not available. Run `ml_pipeline.py` to generate `demand_forecasts.parquet`.")


# ═════════════════════════════════════════════════════════════════════════════
# TREND ANALYSIS  (RF)
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 📈 Trend Analysis — Random Forest Classifier")

trend_df = None if USING_DEMO else load_trends(cats_t, prods_t)

TREND_MAP = {0: "stable", 1: "declining", 2: "trending"}
COLOR_MAP  = {"stable": "#34d399", "trending": "#f472b6", "declining": "#fb923c"}

if trend_df is not None and not trend_df.empty:
    if selected_cats:  trend_df = trend_df[trend_df["category"    ].isin(selected_cats )]
    if selected_prods: trend_df = trend_df[trend_df["product_name"].isin(selected_prods)]
    if len(trend_df) > 50_000:
        trend_df = trend_df.sample(50_000, random_state=42)

    plat_score_map = dict(zip(scores["platform"], scores["overall_score"] / 100))
    trend_df = trend_df.copy()
    trend_df["pref_mul"]        = trend_df["platform"].map(plat_score_map).fillna(0.5)
    trend_df["adj_daily"]       = trend_df["daily_demand"]    * trend_df["pref_mul"]
    trend_df["adj_rolling"]     = trend_df["rolling_avg_7d"]  * trend_df["pref_mul"]
    trend_df["predicted_trend"] = trend_df["predicted_label_idx"].map(TREND_MAP).fillna("unknown")

    st.markdown("""
    <div class="insight-box">
      <div class="title">🌳 Random Forest Classifier — Model Info</div>
      <div class="text">
        <b>Type:</b> Random Forest (Spark MLlib) &nbsp;|&nbsp;
        <b>Trees:</b> 30 &nbsp;|&nbsp; <b>Max Depth:</b> 6 &nbsp;|&nbsp;
        <b>Classes:</b> 3 (stable · declining · trending)<br>
        <b>Columns:</b> platform_idx, category_idx, month, daily_demand, daily_avg_price, rolling_avg_7d
      </div>
    </div>""", unsafe_allow_html=True)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        tc = trend_df["trend_label"].value_counts().reset_index()
        tc.columns = ["Trend","Count"]
        ft1 = px.pie(tc, values="Count", names="Trend", title="Actual Trend Distribution",
                     color="Trend", color_discrete_map=COLOR_MAP)
        ft1.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=380)
        st.plotly_chart(ft1, use_container_width=True)

    with col_t2:
        pc = trend_df["predicted_trend"].value_counts().reset_index()
        pc.columns = ["Trend","Count"]
        ft2 = px.pie(pc, values="Count", names="Trend", title="Predicted Trend Distribution",
                     color="Trend", color_discrete_map=COLOR_MAP)
        ft2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=380)
        st.plotly_chart(ft2, use_container_width=True)

    st.markdown("#### 🔥 Top Trending Products (Preference-Weighted)")
    trending = (trend_df[trend_df["trend_label"] == "trending"]
                .groupby(["product_name","category","platform"])
                .agg(avg_daily=("adj_daily","mean"), rolling_7d=("adj_rolling","mean"))
                .reset_index()
                .sort_values("avg_daily", ascending=False))
    if not trending.empty:
        trending["avg_daily"]  = trending["avg_daily"].round(1)
        trending["rolling_7d"] = trending["rolling_7d"].round(1)
        st.dataframe(
            trending.rename(columns={
                "product_name": "Product", "category": "Category", "platform": "Platform",
                "avg_daily": "Adj Daily Demand", "rolling_7d": "Adj 7-Day Rolling",
            }),
            use_container_width=True, height=300,
        )
    else:
        st.info("No trending products in the selected data.")
else:
    st.info("Trend label data not available. Run `ml_pipeline.py` to generate `trend_labels.parquet`.")


# ═════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(f"""
<div style='text-align:center;color:#64748b;font-size:.85rem;'>
  QuickCommerce Platform Recommender &bull;
  Refreshed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} &bull;
  Data: Blinkit · Zepto · Swiggy Instamart &bull;
  Models: GBT Demand Forecaster + RF Trend Classifier
  {"&bull; <b>⚠️ DEMO MODE</b>" if USING_DEMO else ""}
</div>""", unsafe_allow_html=True)

dbg("✅ App rendered successfully")