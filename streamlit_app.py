"""
streamlit_app.py — QuickCommerce Platform Recommender Dashboard

Combines the original dashboard structure (local / Databricks / S3 paths,
full sidebar, all charts) with EC2-safe optimisations:
  • pyarrow.dataset reads Spark-partitioned folders (part-*.snappy.parquet)
  • Streaming with hard row-cap so 740 MB never lands in RAM
  • Categorical dtypes cast to proper types before any arithmetic
  • Chart samples capped at 50 k rows
  • Graceful fallback to demo data when S3 is unavailable
  • _resolve_s3_path probes both <name>.parquet/ and <name>/ folder variants
"""

import os, sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── S3 timeouts BEFORE any AWS import ────────────────────────────────────────
os.environ.setdefault("S3_CONNECT_TIMEOUT",  "30")
os.environ.setdefault("S3_READ_TIMEOUT",     "120")
os.environ.setdefault("AWS_DEFAULT_REGION",  "eu-north-1")

try:
    import pyarrow.parquet as pq
    import pyarrow.fs      as pafs
    import pyarrow.dataset as pad
except ImportError:
    pq = pafs = pad = None

# ── Debug ─────────────────────────────────────────────────────────────────────
DEBUG = True
def dbg(msg: str):
    if DEBUG:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: {msg}", file=sys.stderr)

dbg("🚀 App init")

# ── Environment detection ─────────────────────────────────────────────────────
IS_DATABRICKS      = "DATABRICKS_RUNTIME_VERSION" in os.environ
IS_STREAMLIT_CLOUD = (
    os.environ.get("STREAMLIT_RUNTIME_VERSION") is not None
    or os.environ.get("STREAMLIT_SERVER_HEADLESS") == "true"
)

# ── AWS credentials ───────────────────────────────────────────────────────────
if IS_STREAMLIT_CLOUD:
    try:
        os.environ["AWS_ACCESS_KEY_ID"]     = st.secrets["AWS_ACCESS_KEY_ID"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["AWS_SECRET_ACCESS_KEY"]
        os.environ["AWS_DEFAULT_REGION"]    = "eu-north-1"
    except Exception:
        pass
elif not os.environ.get("AWS_ACCESS_KEY_ID"):
    try:
        os.environ["AWS_ACCESS_KEY_ID"]     = st.secrets.get("AWS_ACCESS_KEY_ID", "")
        os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets.get("AWS_SECRET_ACCESS_KEY", "")
        dbg("✅ AWS creds loaded from secrets")
    except Exception as e:
        dbg(f"⚠️  Secrets load failed: {e}")

# ── S3 base ───────────────────────────────────────────────────────────────────
S3_BASE = "s3://qcommerce-bdt-cct/parquets"

# ── Static paths for Databricks ───────────────────────────────────────────────
if IS_DATABRICKS:
    PROCESSED_DIR      = "/Volumes/workspace/default/data/processed"
    CURATED_DIR        = "/Volumes/workspace/default/data/curated"
    UNIFIED_PARQUET    = os.path.join(PROCESSED_DIR, "unified.parquet")
    PRICE_ANALYTICS    = os.path.join(PROCESSED_DIR, "price_analytics.parquet")
    DELIVERY_ANALYTICS = os.path.join(PROCESSED_DIR, "delivery_analytics.parquet")
    REVENUE_ANALYTICS  = os.path.join(PROCESSED_DIR, "revenue_analytics.parquet")
    STOCK_ANALYTICS    = os.path.join(PROCESSED_DIR, "stock_analytics.parquet")
    DEMAND_FORECASTS   = os.path.join(CURATED_DIR,   "demand_forecasts.parquet")
    TREND_LABELS       = os.path.join(CURATED_DIR,   "trend_labels.parquet")
else:
    # EC2 / local / Streamlit Cloud — all read from S3
    # Static paths that are known to resolve:
    UNIFIED_PARQUET    = f"{S3_BASE}/unified.parquet"
    PRICE_ANALYTICS    = f"{S3_BASE}/price_analytics.parquet"
    DELIVERY_ANALYTICS = f"{S3_BASE}/delivery_analytics.parquet"
    REVENUE_ANALYTICS  = f"{S3_BASE}/revenue_analytics.parquet"
    STOCK_ANALYTICS    = f"{S3_BASE}/stock_analytics.parquet"
    # These two are resolved at runtime via _resolve_s3_path (set to None here,
    # populated after pyarrow is confirmed available — see "Resolve paths" block)
    DEMAND_FORECASTS   = None
    TREND_LABELS       = None

PLATFORM_COLORS = {"blinkit": "#F8C100", "zepto": "#7B2FF7", "swiggy": "#FC8019"}
PLATFORM_ICONS  = {"blinkit": "🟡",      "zepto": "🟣",      "swiggy": "🟠"}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuickCommerce Platform Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.main-title{font-size:2.2rem;font-weight:800;
  background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px}
.sub-title{font-size:1.05rem;color:#94a3b8;margin-bottom:20px}
.rank-card{background:linear-gradient(135deg,#1e1e2e,#2a2a3e);border-radius:16px;
  padding:24px;margin-bottom:16px;border-left:5px solid;position:relative;
  overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,.15)}
.rank-card::before{content:'';position:absolute;top:0;right:0;width:100px;
  height:100px;border-radius:50%;filter:blur(40px);opacity:.3}
.rank-badge{display:inline-flex;align-items:center;justify-content:center;
  width:44px;height:44px;border-radius:12px;font-size:1.3rem;font-weight:800;
  color:#fff;margin-bottom:8px}
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
.sidebar-section{background:rgba(255,255,255,.03);border-radius:12px;padding:16px;
  margin-bottom:16px;border:1px solid rgba(255,255,255,.06)}
.sidebar-heading{font-size:.85rem;font-weight:700;color:#a78bfa;
  text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px}
.insight-box{background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:12px;
  padding:16px;margin-top:12px;border:1px solid rgba(167,139,250,.2)}
.insight-box .title{font-weight:700;color:#a78bfa;margin-bottom:6px}
.insight-box .text{color:#cbd5e1;font-size:.9rem;line-height:1.5}
div[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f0f1a,#1a1a2e)}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PARQUET UTILITIES  (handles Spark-partitioned S3 folders)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _s3fs():
    if pafs is None:
        return None
    return pafs.S3FileSystem(region=os.environ.get("AWS_DEFAULT_REGION") or None)

def _pq_path(path: str):
    if path.startswith("s3://"):
        return _s3fs(), path.replace("s3://", "", 1)
    return None, path

def _path_is_valid(path: str) -> bool:
    """
    Returns True for:
    • a non-empty local/S3 .parquet file
    • a Spark-partitioned directory containing ≥1 *.parquet part-file
    Uses pad.dataset(..., exclude_invalid_files=True) so _SUCCESS / metadata
    files are silently skipped.
    """
    if pad is None or not path:
        return False
    try:
        fs, bare = _pq_path(path)
        ds    = pad.dataset(bare, filesystem=fs, format="parquet",
                            exclude_invalid_files=True)
        frags = list(ds.get_fragments())
        return len(frags) > 0
    except Exception as e:
        dbg(f"_path_is_valid({path}): {type(e).__name__}: {e}")
        return False


def _resolve_s3_path(base: str, name: str) -> str:
    """
    Probe both <name>.parquet (Spark default folder name) and bare <name>.
    Returns whichever has valid part-files, or the .parquet variant as
    a safe default so error messages still point at a meaningful path.

    This makes trend_labels and demand_forecasts resilient to whether the
    Spark job wrote the folder as  trend_labels.parquet/  or  trend_labels/.
    """
    with_ext = f"{base}/{name}.parquet"
    without  = f"{base}/{name}"
    if _path_is_valid(with_ext):
        dbg(f"✅ resolved {name} → {with_ext}")
        return with_ext
    if _path_is_valid(without):
        dbg(f"✅ resolved {name} → {without}")
        return without
    dbg(f"⚠️  neither path valid for '{name}', defaulting to {with_ext}")
    return with_ext


def _cast_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix dtypes after reading with strings_to_categorical=True.

    pyarrow's strings_to_categorical converts ALL string columns to Categorical.
    That's great for memory, BUT Categorical columns cannot be multiplied,
    added, or used in arithmetic — causing the TypeError seen in production.

    Strategy:
      • Keep string-like categoricals AS Categorical (good for memory/groupby)
      • Cast numeric columns that were accidentally categorised back to float64
      • Cast columns used in arithmetic (predicted_demand, daily_demand, etc.)
        explicitly to float64
    """
    NUMERIC_COLS = {
        "delivery_minutes", "effective_price", "rating",
        "units_ordered", "stock_remaining", "in_stock",
        "predicted_demand", "daily_demand", "rolling_avg_7d",
        "predicted_label_idx",
    }
    for col in df.columns:
        if col not in df.columns:
            continue
        if col in NUMERIC_COLS:
            if hasattr(df[col], "cat"):
                df[col] = df[col].astype(str).replace("nan", np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif hasattr(df[col], "cat"):
            df[col] = df[col].astype("category")
    return df


@st.cache_data(ttl=3600, max_entries=20)
def _stream_parquet(
    path: str,
    columns: Tuple[str, ...],
    max_rows: int,
    filter_categories: Optional[Tuple[str, ...]] = None,
    filter_products:   Optional[Tuple[str, ...]] = None,
    batch_size: int = 65_536,
    seed: int = 42,
) -> Optional[pd.DataFrame]:
    """
    Stream a Spark-partitioned Parquet dataset from S3 in small batches.
    Never holds more than `max_rows` rows in RAM at once.
    """
    if not _path_is_valid(path):
        dbg(f"⛔ {path}: missing / no valid part-files")
        return None
    if pad is None:
        return None

    cats_set  = set(filter_categories or ())
    prods_set = set(filter_products   or ())
    rng       = np.random.default_rng(seed)

    scan_budget = (
        max(800_000,   max_rows * 20) if not (cats_set or prods_set)
        else max(3_000_000, max_rows * 60)
    )

    chunks, collected, scanned = [], 0, 0
    try:
        fs, bare = _pq_path(path)
        dataset  = pad.dataset(bare, filesystem=fs, format="parquet",
                               exclude_invalid_files=True)
        scanner  = dataset.scanner(columns=list(columns), batch_size=batch_size)

        for batch in scanner.to_batches():
            scanned += batch.num_rows
            dfb = _cast_df(batch.to_pandas(strings_to_categorical=True))

            if cats_set  and "category"     in dfb.columns:
                dfb = dfb[dfb["category"].astype(str).isin(cats_set)]
            if prods_set and "product_name" in dfb.columns:
                dfb = dfb[dfb["product_name"].astype(str).isin(prods_set)]
            if dfb.empty:
                if scanned >= scan_budget: break
                continue

            remaining     = max_rows - collected
            if remaining <= 0: break

            per_batch_cap = max(2_000, max_rows // 50) if not (cats_set or prods_set) else remaining
            take          = min(len(dfb), per_batch_cap, remaining)
            if take < len(dfb):
                dfb = dfb.sample(n=take,
                                 random_state=int(rng.integers(0, 2**31 - 1)),
                                 ignore_index=True)

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
        dbg(f"❌ _stream_parquet({path}): {type(e).__name__}: {str(e)[:300]}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# RESOLVE DYNAMIC S3 PATHS  (done once after pyarrow is available)
# ═════════════════════════════════════════════════════════════════════════════

if not IS_DATABRICKS:
    DEMAND_FORECASTS = _resolve_s3_path(S3_BASE, "demand_forecasts")
    TREND_LABELS     = _resolve_s3_path(S3_BASE, "trend_labels")
    dbg(f"DEMAND_FORECASTS → {DEMAND_FORECASTS}")
    dbg(f"TREND_LABELS     → {TREND_LABELS}")


# ═════════════════════════════════════════════════════════════════════════════
# CATALOG  (lightweight sidebar lists — scans only 200 k rows)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def build_catalog(scan_rows: int = 200_000) -> Optional[dict]:
    if not _path_is_valid(UNIFIED_PARQUET):
        dbg("⛔ build_catalog: unified.parquet not found / empty")
        return None

    cols = ("category", "product_name", "units_ordered")
    categories: set = set()
    product_counts  = Counter()
    per_cat: dict   = defaultdict(Counter)
    scanned = 0

    try:
        fs, bare = _pq_path(UNIFIED_PARQUET)
        dataset  = pad.dataset(bare, filesystem=fs, format="parquet",
                               exclude_invalid_files=True)
        scanner  = dataset.scanner(columns=list(cols), batch_size=65_536)

        for batch in scanner.to_batches():
            scanned += batch.num_rows
            dfb = _cast_df(batch.to_pandas(strings_to_categorical=True))
            dfb = dfb.dropna(subset=["category", "product_name"])
            dfb["category"]     = dfb["category"    ].astype(str)
            dfb["product_name"] = dfb["product_name"].astype(str)
            if dfb.empty:
                if scanned >= scan_rows: break
                continue

            categories.update(dfb["category"].unique().tolist())

            if pd.api.types.is_numeric_dtype(dfb["units_ordered"]):
                for (cat, prod), v in (dfb.groupby(["category", "product_name"])
                                         ["units_ordered"].sum().items()):
                    product_counts[prod]  += float(v)
                    per_cat[cat][prod]    += float(v)
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
            "products_by_category": {
                cat: [p for p, _ in cnt.most_common(1_500)]
                for cat, cnt in per_cat.items()
            },
        }
    except Exception as e:
        dbg(f"❌ build_catalog: {type(e).__name__}: {str(e)[:300]}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# TYPED LOADERS
# ═════════════════════════════════════════════════════════════════════════════

UNIFIED_COLS = (
    "platform", "category", "product_name",
    "delivery_minutes", "effective_price", "rating",
    "units_ordered", "stock_remaining", "in_stock", "snapshot_time",
)
DEMAND_COLS = ("platform", "category", "product_name",
               "predicted_demand", "units_ordered")
TREND_COLS  = ("platform", "category", "product_name", "date",
               "daily_demand", "rolling_avg_7d",
               "trend_label", "predicted_label_idx")


def load_unified(cats, prods, max_rows):
    return _stream_parquet(UNIFIED_PARQUET, UNIFIED_COLS, max_rows,
                           filter_categories=cats, filter_products=prods)

def load_demand_forecasts(cats=None, prods=None, max_rows=80_000):
    return _stream_parquet(DEMAND_FORECASTS, DEMAND_COLS, max_rows,
                           filter_categories=cats, filter_products=prods)

def load_trend_labels(cats=None, prods=None, max_rows=80_000):
    return _stream_parquet(TREND_LABELS, TREND_COLS, max_rows,
                           filter_categories=cats, filter_products=prods)


# ═════════════════════════════════════════════════════════════════════════════
# DEMO DATA  (shown when S3 is unreachable)
# ═════════════════════════════════════════════════════════════════════════════

def _demo_unified() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    platforms  = ["blinkit", "zepto", "swiggy"]
    categories = ["Fruits & Vegs", "Dairy", "Snacks", "Beverages", "Personal Care"]
    products   = ["Product A", "Product B", "Product C", "Product D", "Product E"]
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
        "in_stock":         rng.choice([0.0, 1.0], n, p=[0.1, 0.9]),
        "snapshot_time":    pd.date_range("2024-01-01", periods=n, freq="1min"),
    })

def _demo_catalog(df) -> dict:
    return {
        "categories":           sorted(df["category"].unique().tolist()),
        "top_products":         df["product_name"].unique().tolist(),
        "products_by_category": {
            cat: df[df["category"] == cat]["product_name"].unique().tolist()
            for cat in df["category"].unique()
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# SCORING
# ═════════════════════════════════════════════════════════════════════════════

def compute_platform_scores(df, w_delivery, w_price, w_quality):
    agg = df.groupby("platform").agg(
        avg_delivery  = ("delivery_minutes",  "mean"),
        avg_price     = ("effective_price",    "mean"),
        avg_rating    = ("rating",             "mean"),
        product_count = ("product_name",      "nunique"),
        total_orders  = ("units_ordered",      "sum"),
        avg_stock     = ("stock_remaining",    "mean"),
        in_stock_pct  = ("in_stock",           "mean"),
    ).reset_index()

    agg["platform"] = agg["platform"].astype(str)

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
# SIDEBAR — catalog + controls
# ═════════════════════════════════════════════════════════════════════════════

dbg("Building catalog…")
with st.spinner("⏳ Connecting to S3 and building catalog…"):
    catalog = build_catalog()

USING_DEMO = catalog is None
if USING_DEMO:
    _demo_df = _demo_unified()
    catalog  = _demo_catalog(_demo_df)
    st.warning(
        "⚠️ **Live data unavailable** — `unified.parquet` is missing or unreachable on S3.  "
        "Showing **demo data** so you can explore the UI.  \n\n"
        "To fix: ensure valid part-files exist under "
        "`s3://qcommerce-bdt-cct/parquets/unified.parquet/`."
    )

# Performance slider (EC2 memory control)
st.sidebar.markdown('<div class="sidebar-heading">⚙️ Performance</div>', unsafe_allow_html=True)
default_rows = int(os.environ.get("UNIFIED_SAMPLE_ROWS", "150000"))
sample_rows  = st.sidebar.slider(
    "Max rows to sample", 50_000, 400_000,
    value=max(50_000, min(400_000, default_rows)), step=50_000,
    help="Cap on rows read from S3. Lower = faster & less RAM on EC2.",
)

st.sidebar.markdown('<div class="sidebar-heading">📂 Category Selection</div>', unsafe_allow_html=True)
all_categories    = catalog.get("categories", [])
selected_categories = st.sidebar.multiselect(
    "Select Categories", options=all_categories, default=[],
    help="Filter by product category. Leave empty to include all.",
)

st.sidebar.markdown('<div class="sidebar-heading">🛒 Product Selection</div>', unsafe_allow_html=True)
product_query = st.sidebar.text_input("Search Products", value="",
                                      help="Type to filter the product list.")

if selected_categories:
    cands = []
    for cat in selected_categories:
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

selected_products = st.sidebar.multiselect(
    "Select Products", options=cands, default=[],
    help="Leave empty to use all products in the selected categories.",
)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-heading">⚖️ Your Preferences</div>', unsafe_allow_html=True)
st.sidebar.caption("Drag the sliders to tell us what matters most to you. "
                   "All charts — including ML analytics — update based on these weights.")

w_delivery = st.sidebar.slider("🚚  Delivery Speed",        0, 10, 5)
w_price    = st.sidebar.slider("💰  Low Price",              0, 10, 5)
w_quality  = st.sidebar.slider("⭐  Product Quality / Rating", 0, 10, 5)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-heading">ℹ️ Weights Breakdown</div>', unsafe_allow_html=True)
total_w = max(w_delivery + w_price + w_quality, 1)
st.sidebar.markdown(f"""
| Factor | Weight | Share |
|--------|--------|-------|
| 🚚 Delivery | **{w_delivery}** | {w_delivery/total_w*100:.0f}% |
| 💰 Price    | **{w_price}**    | {w_price/total_w*100:.0f}%    |
| ⭐ Quality  | **{w_quality}**  | {w_quality/total_w*100:.0f}%  |
""")


# ═════════════════════════════════════════════════════════════════════════════
# LOAD MAIN DATA
# ═════════════════════════════════════════════════════════════════════════════

cats_t  = tuple(selected_categories) or None
prods_t = tuple(selected_products)   or None

if USING_DEMO:
    unified_df = _demo_df
else:
    with st.spinner("📥 Loading data sample from S3…"):
        unified_df = load_unified(cats_t, prods_t, max_rows=int(sample_rows))

if unified_df is None or unified_df.empty:
    st.error("No data loaded. Try broadening your filters or increasing the sample size.")
    st.stop()

unified_df["platform"] = unified_df["platform"].astype(str)

filtered = unified_df.copy()
if selected_categories:
    filtered = filtered[filtered["category"].astype(str).isin(selected_categories)]
if selected_products:
    filtered = filtered[filtered["product_name"].astype(str).isin(selected_products)]

if filtered.empty:
    st.warning("No rows match the current filters. Please broaden the selection.")
    st.stop()

CHART_CAP = 50_000
chart_df  = filtered.sample(n=CHART_CAP, random_state=42) if len(filtered) > CHART_CAP else filtered
dbg(f"filtered={len(filtered):,}  chart_df={len(chart_df):,}")


# ═════════════════════════════════════════════════════════════════════════════
# HEADER + TOP METRICS
# ═════════════════════════════════════════════════════════════════════════════

if USING_DEMO:
    st.info("🧪 **Demo mode** — displaying synthetic data. Connect S3 to see live results.")

st.markdown('<div class="main-title">🛒 QuickCommerce Platform Recommender</div>', unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Pick your products, set your priorities — we'll rank the best platform for you</div>", unsafe_allow_html=True)

scores = compute_platform_scores(filtered, w_delivery, w_price, w_quality)

c1, c2, c3, c4 = st.columns(4)
c1.metric("📦 Products Matched",  f"{filtered['product_name'].nunique()}")
c2.metric("📄 Records Analysed",  f"{len(filtered):,}")
c3.metric("🏆 Top Platform",
          f"{PLATFORM_ICONS.get(scores.iloc[0]['platform'],'')} {scores.iloc[0]['platform'].title()}")
c4.metric("🎯 Top Score",        f"{scores.iloc[0]['overall_score']:.1f} / 100")

st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
# PLATFORM RANKING CARDS
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("### 🏅 Platform Rankings")

for _, row in scores.iterrows():
    rank  = int(row["rank"])
    plat  = str(row["platform"])
    color = PLATFORM_COLORS.get(plat, "#6366f1")

    st.markdown(f"""
    <div class="rank-card" style="border-left-color: {color};">
        <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
            <div>
                <div class="rank-badge rank-{rank}">#{rank}</div>
            </div>
            <div style="flex:1; min-width:150px;">
                <div class="platform-name">{PLATFORM_ICONS.get(plat,'')} {plat.title()}</div>
            </div>
            <div style="text-align:right;">
                <div class="score-label">Overall Score</div>
                <div class="score-big">{row['overall_score']:.1f}</div>
            </div>
        </div>
        <div class="metric-row">
            <div class="metric-pill">
                <div class="val">{row['avg_delivery']:.1f} min</div>
                <div class="lbl">Avg Delivery</div>
            </div>
            <div class="metric-pill">
                <div class="val">₹{row['avg_price']:.1f}</div>
                <div class="lbl">Avg Price</div>
            </div>
            <div class="metric-pill">
                <div class="val">{row['avg_rating']:.2f} ⭐</div>
                <div class="lbl">Avg Rating</div>
            </div>
            <div class="metric-pill">
                <div class="val">{row['in_stock_pct']:.1f}%</div>
                <div class="lbl">In Stock</div>
            </div>
            <div class="metric-pill">
                <div class="val">{int(row['total_orders']):,}</div>
                <div class="lbl">Total Orders</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SCORE BREAKDOWN CHARTS
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 📊 Score Breakdown")

col_left, col_right = st.columns(2)

with col_left:
    fig = go.Figure()
    for _, row in scores.iterrows():
        plat = str(row["platform"])
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
        height=400, margin=dict(t=50,b=30), legend=dict(orientation="h",y=-0.1),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    bar_data = []
    for _, row in scores.iterrows():
        p = str(row["platform"]).title()
        bar_data += [
            {"Platform": p, "Factor": "🚚 Delivery", "Score": row["delivery_score"]},
            {"Platform": p, "Factor": "💰 Price",    "Score": row["price_score"]},
            {"Platform": p, "Factor": "⭐ Quality",  "Score": row["quality_score"]},
        ]
    fig2 = px.bar(pd.DataFrame(bar_data), x="Platform", y="Score", color="Factor",
                  barmode="group", title="Individual Factor Scores",
                  color_discrete_sequence=["#38bdf8","#34d399","#f472b6"],
                  template="plotly_dark")
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       height=400, margin=dict(t=50,b=30),
                       legend=dict(orientation="h",y=-0.15))
    st.plotly_chart(fig2, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY-LEVEL COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 🔍 Category-Level Platform Comparison")

cat_platform = chart_df.groupby(["category", "platform"]).agg(
    avg_price    = ("effective_price",   "mean"),
    avg_delivery = ("delivery_minutes",  "mean"),
    avg_rating   = ("rating",            "mean"),
).reset_index()
cat_platform["platform"] = cat_platform["platform"].astype(str)
cat_platform["category"] = cat_platform["category"].astype(str)

col_a, col_b = st.columns(2)
with col_a:
    fig3 = px.bar(cat_platform, x="category", y="avg_price", color="platform",
                  barmode="group", title="Average Price by Category & Platform",
                  color_discrete_map=PLATFORM_COLORS, template="plotly_dark")
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       xaxis_tickangle=-45, height=420, margin=dict(t=50,b=80))
    st.plotly_chart(fig3, use_container_width=True)

with col_b:
    fig4 = px.bar(cat_platform, x="category", y="avg_delivery", color="platform",
                  barmode="group", title="Average Delivery Time by Category & Platform",
                  color_discrete_map=PLATFORM_COLORS, template="plotly_dark")
    fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       xaxis_tickangle=-45, height=420, margin=dict(t=50,b=80))
    st.plotly_chart(fig4, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PRODUCT-LEVEL TABLE
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 📋 Product-Level Platform Metrics")

product_plat = chart_df.groupby(["product_name", "category", "platform"]).agg(
    avg_price    = ("effective_price",  "mean"),
    avg_delivery = ("delivery_minutes", "mean"),
    avg_rating   = ("rating",           "mean"),
    in_stock_pct = ("in_stock",         "mean"),
).reset_index()
product_plat["in_stock_pct"]  = (product_plat["in_stock_pct"] * 100).round(1)
product_plat["avg_price"]     = product_plat["avg_price"].round(2)
product_plat["avg_delivery"]  = product_plat["avg_delivery"].round(1)
product_plat["avg_rating"]    = product_plat["avg_rating"].round(2)

st.dataframe(
    product_plat.rename(columns={
        "product_name": "Product",      "category":     "Category",
        "platform":     "Platform",     "avg_price":    "Avg Price (₹)",
        "avg_delivery": "Delivery (min)", "avg_rating": "Rating ⭐",
        "in_stock_pct": "In Stock %",
    }),
    use_container_width=True, height=400,
)


# ═════════════════════════════════════════════════════════════════════════════
# SMART INSIGHT
# ═════════════════════════════════════════════════════════════════════════════

best = scores.iloc[0]
st.markdown(f"""
<div class="insight-box">
    <div class="title">💡 Recommendation Insight</div>
    <div class="text">
        Based on your preference weights (Delivery: <b>{w_delivery}</b>, Price: <b>{w_price}</b>,
        Quality: <b>{w_quality}</b>), <b>{str(best['platform']).title()}</b> is your best pick
        with an overall score of <b>{best['overall_score']:.1f}/100</b>.<br>
        It delivers in ~<b>{best['avg_delivery']:.0f} min</b> on average at
        <b>₹{best['avg_price']:.0f}</b> avg price with a <b>{best['avg_rating']:.2f}⭐</b> rating
        across your selected {'products' if selected_products else 'categories'}.
    </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# DEMAND FORECASTING  (GBT Model)
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 🤖 Demand Forecasting — GBT Model")

demand_df = None if USING_DEMO else load_demand_forecasts(cats_t, prods_t)

if demand_df is not None and not demand_df.empty:
    demand_df["platform"] = demand_df["platform"].astype(str)

    if selected_categories:
        demand_filtered = demand_df[demand_df["category"].astype(str)
                                    .isin(selected_categories)].copy()
    else:
        demand_filtered = demand_df.copy()
    if selected_products:
        demand_filtered = demand_filtered[demand_filtered["product_name"].astype(str)
                                          .isin(selected_products)]

    if len(demand_filtered) > 50_000:
        demand_filtered = demand_filtered.sample(50_000, random_state=42)

    platform_scores_map = dict(zip(scores["platform"], scores["overall_score"] / 100))
    demand_filtered = demand_filtered.copy()
    demand_filtered["preference_multiplier"] = (
        demand_filtered["platform"].map(platform_scores_map).fillna(0.5)
    )
    demand_filtered["adj_predicted_demand"] = (
        demand_filtered["predicted_demand"].astype(float)
        * demand_filtered["preference_multiplier"].astype(float)
    )

    if not demand_filtered.empty:
        n_trees      = 15
        max_depth    = 4
        feature_list = ["platform_idx","category_idx","weather_idx",
                        "is_raining_int","in_stock_int","stock_remaining"]

        st.markdown(f"""
        <div class="insight-box">
            <div class="title">🌲 Model Architecture — GBT Regressor</div>
            <div class="text">
                <b>Type:</b> Gradient-Boosted Trees (Spark MLlib) &nbsp;|&nbsp;
                <b>Trees:</b> {n_trees} &nbsp;|&nbsp;
                <b>Max Depth:</b> {max_depth} &nbsp;|&nbsp;
                <b>Features:</b> {len(feature_list)} &nbsp;|&nbsp;
                <b>Target:</b> units_ordered<br>
                <b>Feature columns:</b> {', '.join(feature_list)}
            </div>
        </div>
        """, unsafe_allow_html=True)

        demand_agg = demand_filtered.groupby("platform").agg(
            avg_actual        = ("units_ordered",       "mean"),
            avg_predicted     = ("predicted_demand",    "mean"),
            avg_adj_predicted = ("adj_predicted_demand","mean"),
            total_actual      = ("units_ordered",       "sum"),
            total_predicted   = ("predicted_demand",    "sum"),
        ).reset_index()

        col_d1, col_d2 = st.columns(2)

        with col_d1:
            fig_d1 = go.Figure()
            fig_d1.add_trace(go.Bar(
                x=demand_agg["platform"].str.title(), y=demand_agg["avg_actual"],
                name="Actual Demand", marker_color="#38bdf8",
            ))
            fig_d1.add_trace(go.Bar(
                x=demand_agg["platform"].str.title(), y=demand_agg["avg_adj_predicted"],
                name="Slider-Adjusted Prediction", marker_color="#a78bfa",
            ))
            fig_d1.update_layout(
                title="Avg Demand: Actual vs Predicted", barmode="group",
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=400,
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_d1, use_container_width=True)

        with col_d2:
            prod_demand = (
                demand_filtered.groupby("product_name").agg(
                    avg_predicted = ("adj_predicted_demand", "mean"),
                    avg_actual    = ("units_ordered",         "mean"),
                ).reset_index().nlargest(15, "avg_predicted")
            )
            prod_demand["product_name"] = prod_demand["product_name"].astype(str)
            fig_d2 = go.Figure()
            fig_d2.add_trace(go.Bar(
                y=prod_demand["product_name"], x=prod_demand["avg_actual"],
                name="Actual", orientation="h", marker_color="#38bdf8",
            ))
            fig_d2.add_trace(go.Bar(
                y=prod_demand["product_name"], x=prod_demand["avg_predicted"],
                name="Adjusted Predicted", orientation="h", marker_color="#a78bfa",
            ))
            fig_d2.update_layout(
                title="Top Products by Preference-Weighted Demand", barmode="group",
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=400,
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_d2, use_container_width=True)

        cat_plat_demand = (
            demand_filtered.groupby(["category", "platform"])
            ["adj_predicted_demand"].mean().reset_index()
        )
        cat_plat_demand["category"] = cat_plat_demand["category"].astype(str)
        cat_plat_demand["platform"] = cat_plat_demand["platform"].astype(str)
        pivot = (cat_plat_demand
                 .pivot(index="category", columns="platform",
                        values="adj_predicted_demand")
                 .fillna(0))
        fig_heat = px.imshow(
            pivot.values, x=pivot.columns.str.title(), y=pivot.index,
            color_continuous_scale="Viridis", aspect="auto",
            title="Slider-Adjusted Demand Heatmap (Category × Platform)",
            labels={"color": "Adj Demand"},
        )
        fig_heat.update_layout(template="plotly_dark",
                               paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", height=380)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No demand forecast data for selected products.")
else:
    if USING_DEMO:
        st.warning("Demand forecast data not available in demo mode.")
    else:
        st.warning(
            f"Demand forecast data not found at `{DEMAND_FORECASTS}`. "
            "Run `ml_pipeline.py` to generate it."
        )


# ═════════════════════════════════════════════════════════════════════════════
# TREND ANALYSIS  (RF Model)
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 📈 Trend Analysis — Random Forest Classifier")

# ── Load using the same _stream_parquet path machinery as unified ─────────────
trend_df = None if USING_DEMO else load_trend_labels(cats_t, prods_t)

if trend_df is not None and not trend_df.empty:
    trend_df["platform"] = trend_df["platform"].astype(str)

    if selected_categories:
        trend_filtered = trend_df[trend_df["category"].astype(str)
                                  .isin(selected_categories)].copy()
    else:
        trend_filtered = trend_df.copy()
    if selected_products:
        trend_filtered = trend_filtered[trend_filtered["product_name"].astype(str)
                                        .isin(selected_products)]

    if len(trend_filtered) > 50_000:
        trend_filtered = trend_filtered.sample(50_000, random_state=42)

    platform_scores_map = dict(zip(scores["platform"], scores["overall_score"] / 100))
    trend_filtered = trend_filtered.copy()
    trend_filtered["preference_multiplier"] = (
        trend_filtered["platform"].map(platform_scores_map).fillna(0.5)
    )
    trend_filtered["adj_daily_demand"] = (
        trend_filtered["daily_demand"].astype(float)
        * trend_filtered["preference_multiplier"].astype(float)
    )
    trend_filtered["adj_rolling_7d"] = (
        trend_filtered["rolling_avg_7d"].astype(float)
        * trend_filtered["preference_multiplier"].astype(float)
    )

    if not trend_filtered.empty:
        rf_trees    = 30
        rf_depth    = 6
        rf_classes  = 3
        rf_features = ["platform_idx","category_idx","month",
                       "daily_demand","daily_avg_price","rolling_avg_7d"]
        trend_label_map = {0: "stable", 1: "declining", 2: "trending"}

        st.markdown(f"""
        <div class="insight-box">
            <div class="title">🌳 Model Architecture — Random Forest Classifier</div>
            <div class="text">
                <b>Type:</b> Random Forest (Spark MLlib) &nbsp;|&nbsp;
                <b>Trees:</b> {rf_trees} &nbsp;|&nbsp;
                <b>Max Depth:</b> {rf_depth} &nbsp;|&nbsp;
                <b>Classes:</b> {rf_classes} (stable, declining, trending) &nbsp;|&nbsp;
                <b>Features:</b> {len(rf_features)}<br>
                <b>Feature columns:</b> {', '.join(rf_features)}
            </div>
        </div>
        """, unsafe_allow_html=True)

        trend_filtered["predicted_trend"] = (
            trend_filtered["predicted_label_idx"]
            .astype(float).round().astype("Int64")
            .map(trend_label_map).fillna("unknown")
        )

        color_map = {"stable": "#34d399", "trending": "#f472b6", "declining": "#fb923c"}
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            trend_counts = (trend_filtered["trend_label"].astype(str)
                            .value_counts().reset_index())
            trend_counts.columns = ["Trend", "Count"]
            fig_t1 = px.pie(trend_counts, values="Count", names="Trend",
                            title="Actual Trend Distribution",
                            color="Trend", color_discrete_map=color_map)
            fig_t1.update_layout(template="plotly_dark",
                                 paper_bgcolor="rgba(0,0,0,0)", height=380)
            st.plotly_chart(fig_t1, use_container_width=True)

        with col_t2:
            pred_counts = (trend_filtered["predicted_trend"]
                           .value_counts().reset_index())
            pred_counts.columns = ["Trend", "Count"]
            fig_t2 = px.pie(pred_counts, values="Count", names="Trend",
                            title="Predicted Trend Distribution",
                            color="Trend", color_discrete_map=color_map)
            fig_t2.update_layout(template="plotly_dark",
                                 paper_bgcolor="rgba(0,0,0,0)", height=380)
            st.plotly_chart(fig_t2, use_container_width=True)

        st.markdown("#### 🔥 Top Trending Products (Weighted by Preferences)")
        trending_prods = (
            trend_filtered[trend_filtered["trend_label"].astype(str) == "trending"]
            .groupby(["product_name", "category", "platform"])
            .agg(avg_daily_demand=("adj_daily_demand","mean"),
                 rolling_7d      =("adj_rolling_7d",  "mean"))
            .reset_index()
            .sort_values("avg_daily_demand", ascending=False)
        )
        if not trending_prods.empty:
            trending_prods["avg_daily_demand"] = trending_prods["avg_daily_demand"].round(1)
            trending_prods["rolling_7d"]       = trending_prods["rolling_7d"].round(1)
            st.dataframe(
                trending_prods.rename(columns={
                    "product_name":    "Product",
                    "category":        "Category",
                    "platform":        "Platform",
                    "avg_daily_demand":"Adj Daily Demand",
                    "rolling_7d":      "Adj 7-Day Rolling",
                }),
                use_container_width=True, height=300,
            )
        else:
            st.info("No trending products found in the selected data.")
    else:
        st.info("No trend data for selected products.")
else:
    if USING_DEMO:
        st.warning("Trend analysis not available in demo mode.")
    else:
        st.warning(
            f"Trend labels data not found at `{TREND_LABELS}`. "
            "Run `ml_pipeline.py` to generate it, or check that the S3 folder "
            "`trend_labels.parquet/` (or `trend_labels/`) exists under "
            f"`{S3_BASE}/`."
        )


# ═════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(f"""
<div style='text-align:center; color:#64748b; font-size:0.85rem;'>
    QuickCommerce Platform Recommender &bull;
    Last refreshed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    &bull; Data: Blinkit · Zepto · Swiggy Instamart
    &bull; Models: GBT Demand Forecaster + RF Trend Classifier
    {"&bull; <b>⚠️ DEMO MODE</b>" if USING_DEMO else ""}
</div>
""", unsafe_allow_html=True)

dbg("✅ App rendered successfully")