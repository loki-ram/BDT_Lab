"""
streamlit_app.py — QuickCommerce Platform Recommender Dashboard

Lets users pick product categories, adjust preference sliders
(delivery time, price, product quality), and ranks platforms accordingly.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime
import os
import json
import glob

# ─────────────────────────────────────────────
# ENVIRONMENT & PATHS
# ─────────────────────────────────────────────
IS_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ

# Safely detect Streamlit Cloud (check env var only at module level)
IS_STREAMLIT_CLOUD = os.environ.get("STREAMLIT_RUNTIME_VERSION") is not None

if IS_STREAMLIT_CLOUD:
    # Read Secrets internally provided to Streamlit Cloud
    try:
        os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["AWS_ACCESS_KEY_ID"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["AWS_SECRET_ACCESS_KEY"]
        os.environ["AWS_DEFAULT_REGION"] = "eu-north-1"
    except Exception:
        pass
    
    # User's specified S3 path containing the uploaded parquets
    S3_BASE = "s3://qcommerce-bdt-cct/parquets"
    BASE_DIR = S3_BASE  # Provides a dummy BASE_DIR for non-existent models to safely fail
    UNIFIED_PARQUET = f"{S3_BASE}/unified.parquet"
    PRICE_ANALYTICS = f"{S3_BASE}/price_analytics.parquet"
    DELIVERY_ANALYTICS = f"{S3_BASE}/delivery_analytics.parquet"
    REVENUE_ANALYTICS = f"{S3_BASE}/revenue_analytics.parquet"
    STOCK_ANALYTICS = f"{S3_BASE}/stock_analytics.parquet"
    DEMAND_FORECASTS = f"{S3_BASE}/demand_forecasts.parquet"
    TREND_LABELS = f"{S3_BASE}/trend_labels.parquet"

PLATFORM_COLORS = {"blinkit": "#F8C100", "zepto": "#7B2FF7", "swiggy": "#FC8019"}
PLATFORM_ICONS = {"blinkit": "🟡", "zepto": "🟣", "swiggy": "🟠"}

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="QuickCommerce Platform Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-title {
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.sub-title { font-size: 1.05rem; color: #94a3b8; margin-bottom: 20px; }

.rank-card {
    background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
    border-radius: 16px; padding: 24px; margin-bottom: 16px;
    border-left: 5px solid; position: relative; overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.15);
}
.rank-card::before {
    content: ''; position: absolute; top: 0; right: 0;
    width: 100px; height: 100px; border-radius: 50%;
    filter: blur(40px); opacity: 0.3;
}
.rank-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 44px; height: 44px; border-radius: 12px;
    font-size: 1.3rem; font-weight: 800; color: #fff;
    margin-bottom: 8px;
}
.rank-1 { background: linear-gradient(135deg, #f59e0b, #d97706); }
.rank-2 { background: linear-gradient(135deg, #94a3b8, #64748b); }
.rank-3 { background: linear-gradient(135deg, #b45309, #92400e); }

.platform-name { font-size: 1.4rem; font-weight: 700; color: #f1f5f9; }
.score-big { font-size: 2.4rem; font-weight: 800; color: #a78bfa; }
.score-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }

.metric-row { display: flex; gap: 16px; margin-top: 12px; flex-wrap: wrap; }
.metric-pill {
    background: rgba(255,255,255,0.06); border-radius: 10px;
    padding: 8px 14px; flex: 1; min-width: 100px; text-align: center;
}
.metric-pill .val { font-size: 1.1rem; font-weight: 700; color: #e2e8f0; }
.metric-pill .lbl { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; }

.sidebar-section {
    background: rgba(255,255,255,0.03); border-radius: 12px;
    padding: 16px; margin-bottom: 16px;
    border: 1px solid rgba(255,255,255,0.06);
}
.sidebar-heading {
    font-size: 0.85rem; font-weight: 700; color: #a78bfa;
    text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 10px;
}

.insight-box {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 12px; padding: 16px; margin-top: 12px;
    border: 1px solid rgba(167,139,250,0.2);
}
.insight-box .title { font-weight: 700; color: #a78bfa; margin-bottom: 6px; }
.insight-box .text { color: #cbd5e1; font-size: 0.9rem; line-height: 1.5; }

div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_unified():
    try:
        df = pd.read_parquet(UNIFIED_PARQUET)
        df["snapshot_time"] = pd.to_datetime(df["snapshot_time"])
        return df
    except Exception as e:
        st.error(f"❌ Could not load unified data: {e}")
        return None

@st.cache_data(ttl=3600)
def load_demand_forecasts():
    try:
        return pd.read_parquet(DEMAND_FORECASTS)
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def load_trend_labels():
    try:
        df = pd.read_parquet(TREND_LABELS)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        return None

# ─────────────────────────────────────────────
# SCORING LOGIC
# ─────────────────────────────────────────────
def compute_platform_scores(df, w_delivery, w_price, w_quality):
    """
    For each platform compute normalised scores for delivery, price and quality,
    then combine them using the user-supplied weights.
    Lower delivery time is better, lower price is better, higher rating is better.
    """
    agg = df.groupby("platform").agg(
        avg_delivery=("delivery_minutes", "mean"),
        avg_price=("effective_price", "mean"),
        avg_rating=("rating", "mean"),
        product_count=("product_name", "nunique"),
        total_orders=("units_ordered", "sum"),
        avg_stock=("stock_remaining", "mean"),
        in_stock_pct=("in_stock", "mean"),
    ).reset_index()

    # Normalise 0-100 (invert delivery & price so lower = better score)
    def norm(s, invert=False):
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series([50.0] * len(s))
        n = (s - mn) / (mx - mn) * 100
        return 100 - n if invert else n

    agg["delivery_score"] = norm(agg["avg_delivery"], invert=True)
    agg["price_score"] = norm(agg["avg_price"], invert=True)
    agg["quality_score"] = norm(agg["avg_rating"], invert=False)

    total_w = w_delivery + w_price + w_quality
    if total_w == 0:
        total_w = 1

    agg["overall_score"] = (
        agg["delivery_score"] * w_delivery
        + agg["price_score"] * w_price
        + agg["quality_score"] * w_quality
    ) / total_w

    agg["in_stock_pct"] = agg["in_stock_pct"] * 100
    agg = agg.sort_values("overall_score", ascending=False).reset_index(drop=True)
    agg["rank"] = range(1, len(agg) + 1)
    return agg


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
unified_df = load_unified()

if unified_df is None:
    st.stop()

# ---------- SIDEBAR ----------
st.sidebar.markdown('<div class="sidebar-heading">📂 Category Selection</div>', unsafe_allow_html=True)

all_categories = sorted(unified_df["category"].unique())
selected_categories = st.sidebar.multiselect(
    "Select Categories",
    options=all_categories,
    default=[],
    help="Filter by product category. Leave empty to include all categories.",
)

st.sidebar.markdown('<div class="sidebar-heading">🛒 Product Selection</div>', unsafe_allow_html=True)

# Filter product list based on selected categories
if selected_categories:
    category_products = sorted(unified_df[unified_df["category"].isin(selected_categories)]["product_name"].unique())
else:
    category_products = sorted(unified_df["product_name"].unique())

selected_products = st.sidebar.multiselect(
    "Select Products",
    options=category_products,
    default=[],
    help="Choose one or more products to compare platforms. Leave empty to use all products in selected categories.",
)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-heading">⚖️ Your Preferences</div>', unsafe_allow_html=True)
st.sidebar.caption("Drag the sliders to tell us what matters most to you. All charts — including ML analytics — update based on these weights.")

w_delivery = st.sidebar.slider("🚚  Delivery Speed", 0, 10, 5, help="How important is fast delivery?")
w_price = st.sidebar.slider("💰  Low Price", 0, 10, 5, help="How important is a lower price?")
w_quality = st.sidebar.slider("⭐  Product Quality / Rating", 0, 10, 5, help="How important are product ratings?")

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-heading">ℹ️ Weights Breakdown</div>', unsafe_allow_html=True)
total_w = w_delivery + w_price + w_quality or 1
st.sidebar.markdown(f"""
| Factor | Weight | Share |
|--------|--------|-------|
| 🚚 Delivery | **{w_delivery}** | {w_delivery/total_w*100:.0f}% |
| 💰 Price | **{w_price}** | {w_price/total_w*100:.0f}% |
| ⭐ Quality | **{w_quality}** | {w_quality/total_w*100:.0f}% |
""")

# ---------- FILTER DATA ----------
filtered = unified_df.copy()
if selected_categories:
    filtered = filtered[filtered["category"].isin(selected_categories)]
if selected_products:
    filtered = filtered[filtered["product_name"].isin(selected_products)]

if filtered.empty:
    st.warning("No data matches the selected filters. Please adjust your selection.")
    st.stop()

# ---------- HEADER ----------
st.markdown('<div class="main-title">🛒 QuickCommerce Platform Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Pick your products, set your priorities — we\'ll rank the best platform for you</div>', unsafe_allow_html=True)

# ---------- COMPUTE SCORES ----------
scores = compute_platform_scores(filtered, w_delivery, w_price, w_quality)

# ---------- TOP METRICS ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("📦 Products Matched", f"{filtered['product_name'].nunique()}")
c2.metric("📄 Records Analysed", f"{len(filtered):,}")
c3.metric("🏆 Top Platform", f"{PLATFORM_ICONS.get(scores.iloc[0]['platform'],'')} {scores.iloc[0]['platform'].title()}")
c4.metric("🎯 Top Score", f"{scores.iloc[0]['overall_score']:.1f} / 100")

st.markdown("---")

# ─────────────────────────────────────────────
# PLATFORM RANKING CARDS
# ─────────────────────────────────────────────
st.markdown("### 🏅 Platform Rankings")

for _, row in scores.iterrows():
    rank = int(row["rank"])
    plat = row["platform"]
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

# ─────────────────────────────────────────────
# SCORE BREAKDOWN CHARTS
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Score Breakdown")

col_left, col_right = st.columns(2)

with col_left:
    # Radar chart
    cats_radar = ["Delivery Speed", "Low Price", "Quality / Rating"]
    fig = go.Figure()
    for _, row in scores.iterrows():
        plat = row["platform"]
        fig.add_trace(go.Scatterpolar(
            r=[row["delivery_score"], row["price_score"], row["quality_score"]],
            theta=cats_radar,
            fill="toself",
            name=plat.title(),
            line_color=PLATFORM_COLORS.get(plat, "#6366f1"),
            opacity=0.75,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False),
        ),
        title="Platform Strength Radar",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin=dict(t=50, b=30),
        legend=dict(orientation="h", y=-0.1),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    # Grouped bar
    bar_data = []
    for _, row in scores.iterrows():
        plat = row["platform"].title()
        bar_data.append({"Platform": plat, "Factor": "🚚 Delivery", "Score": row["delivery_score"]})
        bar_data.append({"Platform": plat, "Factor": "💰 Price", "Score": row["price_score"]})
        bar_data.append({"Platform": plat, "Factor": "⭐ Quality", "Score": row["quality_score"]})
    bar_df = pd.DataFrame(bar_data)
    fig2 = px.bar(
        bar_df, x="Platform", y="Score", color="Factor",
        barmode="group", title="Individual Factor Scores",
        color_discrete_sequence=["#38bdf8", "#34d399", "#f472b6"],
        template="plotly_dark",
    )
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=400, margin=dict(t=50, b=30),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────
# DETAILED CATEGORY-LEVEL COMPARISON
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔍 Category-Level Platform Comparison")

cat_platform = filtered.groupby(["category", "platform"]).agg(
    avg_price=("effective_price", "mean"),
    avg_delivery=("delivery_minutes", "mean"),
    avg_rating=("rating", "mean"),
).reset_index()

col_a, col_b = st.columns(2)

with col_a:
    fig3 = px.bar(
        cat_platform, x="category", y="avg_price", color="platform",
        barmode="group", title="Average Price by Category & Platform",
        color_discrete_map=PLATFORM_COLORS, template="plotly_dark",
    )
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       xaxis_tickangle=-45, height=420, margin=dict(t=50, b=80))
    st.plotly_chart(fig3, use_container_width=True)

with col_b:
    fig4 = px.bar(
        cat_platform, x="category", y="avg_delivery", color="platform",
        barmode="group", title="Average Delivery Time by Category & Platform",
        color_discrete_map=PLATFORM_COLORS, template="plotly_dark",
    )
    fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       xaxis_tickangle=-45, height=420, margin=dict(t=50, b=80))
    st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────────────────────
# PRODUCT-LEVEL TABLE
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📋 Product-Level Platform Metrics")

product_plat = filtered.groupby(["product_name", "category", "platform"]).agg(
    avg_price=("effective_price", "mean"),
    avg_delivery=("delivery_minutes", "mean"),
    avg_rating=("rating", "mean"),
    in_stock_pct=("in_stock", "mean"),
).reset_index()
product_plat["in_stock_pct"] = (product_plat["in_stock_pct"] * 100).round(1)
product_plat["avg_price"] = product_plat["avg_price"].round(2)
product_plat["avg_delivery"] = product_plat["avg_delivery"].round(1)
product_plat["avg_rating"] = product_plat["avg_rating"].round(2)

st.dataframe(
    product_plat.rename(columns={
        "product_name": "Product",
        "category": "Category",
        "platform": "Platform",
        "avg_price": "Avg Price (₹)",
        "avg_delivery": "Delivery (min)",
        "avg_rating": "Rating ⭐",
        "in_stock_pct": "In Stock %",
    }),
    use_container_width=True,
    height=400,
)

# ─────────────────────────────────────────────
# SMART INSIGHT
# ─────────────────────────────────────────────
best = scores.iloc[0]
st.markdown(f"""
<div class="insight-box">
    <div class="title">💡 Recommendation Insight</div>
    <div class="text">
        Based on your preference weights (Delivery: <b>{w_delivery}</b>, Price: <b>{w_price}</b>,
        Quality: <b>{w_quality}</b>), <b>{best['platform'].title()}</b> is your best pick
        with an overall score of <b>{best['overall_score']:.1f}/100</b>.<br>
        It delivers in ~<b>{best['avg_delivery']:.0f} min</b> on average at
        <b>₹{best['avg_price']:.0f}</b> avg price with a <b>{best['avg_rating']:.2f}⭐</b> rating
        across your selected {'products' if selected_products else 'categories'}.
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DEMAND FORECASTING (GBT Model)
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🤖 Demand Forecasting — GBT Model")

demand_df = load_demand_forecasts()
if demand_df is not None:
    # Filter by selected categories and products
    if selected_categories:
        demand_filtered = demand_df[demand_df["category"].isin(selected_categories)].copy()
    else:
        demand_filtered = demand_df.copy()

    if selected_products:
        demand_filtered = demand_filtered[demand_filtered["product_name"].isin(selected_products)]

    # Dynamic Weighting: Adjust predictions based on user's slider preferences (via platform scores)
    platform_scores_map = dict(zip(scores["platform"], scores["overall_score"] / 100))
    demand_filtered["preference_multiplier"] = demand_filtered["platform"].map(platform_scores_map).fillna(0.5)
    demand_filtered["adj_predicted_demand"] = demand_filtered["predicted_demand"] * demand_filtered["preference_multiplier"]

    if not demand_filtered.empty:
        # Hardcoded Model Info for GBT Regressor
        n_trees = 15
        n_features = 6
        max_depth = 4
        feature_list = ["platform_idx", "category_idx", "weather_idx", "is_raining_int", "in_stock_int", "stock_remaining"]

        st.markdown(f"""
        <div class="insight-box">
            <div class="title">🌲 Model Architecture — GBT Regressor</div>
            <div class="text">
                <b>Type:</b> Gradient-Boosted Trees (Spark MLlib) &nbsp;|&nbsp;
                <b>Trees:</b> {n_trees} &nbsp;|&nbsp;
                <b>Max Depth:</b> {max_depth} &nbsp;|&nbsp;
                <b>Features:</b> {n_features} &nbsp;|&nbsp;
                <b>Target:</b> units_ordered<br>
                <b>Feature columns:</b> {', '.join(feature_list)}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Actual vs Predicted by platform
        demand_agg = demand_filtered.groupby("platform").agg(
            avg_actual=("units_ordered", "mean"),
            avg_predicted=("predicted_demand", "mean"),
            avg_adj_predicted=("adj_predicted_demand", "mean"),
            total_actual=("units_ordered", "sum"),
            total_predicted=("predicted_demand", "sum"),
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
            # Demand by product (top 15)
            prod_demand = demand_filtered.groupby("product_name").agg(
                avg_predicted=("adj_predicted_demand", "mean"),
                avg_actual=("units_ordered", "mean"),
            ).reset_index().nlargest(15, "avg_predicted")

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

        # Demand by category heatmap
        cat_plat_demand = demand_filtered.groupby(["category", "platform"])["adj_predicted_demand"].mean().reset_index()
        pivot = cat_plat_demand.pivot(index="category", columns="platform", values="adj_predicted_demand").fillna(0)
        fig_heat = px.imshow(
            pivot.values, x=pivot.columns.str.title(), y=pivot.index,
            color_continuous_scale="Viridis", aspect="auto",
            title="Slider-Adjusted Demand Heatmap (Category × Platform)",
            labels={"color": "Adj Demand"},
        )
        fig_heat.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=380,
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No demand forecast data for selected products.")
else:
    st.warning("Demand forecast data not found. Run `ml_pipeline.py` to generate it.")

# ─────────────────────────────────────────────
# TREND ANALYSIS (RF Model)
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Trend Analysis — Random Forest Classifier")

trend_df = load_trend_labels()
if trend_df is not None:
    # Filter by selected categories and products
    if selected_categories:
        trend_filtered = trend_df[trend_df["category"].isin(selected_categories)].copy()
    else:
        trend_filtered = trend_df.copy()

    if selected_products:
        trend_filtered = trend_filtered[trend_filtered["product_name"].isin(selected_products)]

    # Dynamic Weighting: Adjust trend strength based on slider preferences
    platform_scores_map = dict(zip(scores["platform"], scores["overall_score"] / 100))
    trend_filtered["preference_multiplier"] = trend_filtered["platform"].map(platform_scores_map).fillna(0.5)
    trend_filtered["adj_daily_demand"] = trend_filtered["daily_demand"] * trend_filtered["preference_multiplier"]
    trend_filtered["adj_rolling_7d"] = trend_filtered["rolling_avg_7d"] * trend_filtered["preference_multiplier"]

    if not trend_filtered.empty:
        # Hardcoded Model Info for Random Forest Classifier
        rf_trees = 30
        rf_classes = 3
        rf_depth = 6
        rf_features = ["platform_idx", "category_idx", "month", "daily_demand", "daily_avg_price", "rolling_avg_7d"]
        
        # Label mapping from first stage (StringIndexer on trend_label)
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

        # Map predicted index to label
        trend_filtered = trend_filtered.copy()
        trend_filtered["predicted_trend"] = trend_filtered["predicted_label_idx"].map(trend_label_map).fillna("unknown")

        col_t1, col_t2 = st.columns(2)

        with col_t1:
            # Actual trend distribution
            trend_counts = trend_filtered["trend_label"].value_counts().reset_index()
            trend_counts.columns = ["Trend", "Count"]
            color_map = {"stable": "#34d399", "trending": "#f472b6", "declining": "#fb923c"}
            fig_t1 = px.pie(
                trend_counts, values="Count", names="Trend",
                title="Actual Trend Distribution",
                color="Trend", color_discrete_map=color_map,
            )
            fig_t1.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                height=380,
            )
            st.plotly_chart(fig_t1, use_container_width=True)

        with col_t2:
            # Predicted trend distribution
            pred_counts = trend_filtered["predicted_trend"].value_counts().reset_index()
            pred_counts.columns = ["Trend", "Count"]
            fig_t2 = px.pie(
                pred_counts, values="Count", names="Trend",
                title="Predicted Trend Distribution",
                color="Trend", color_discrete_map=color_map,
            )
            fig_t2.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                height=380,
            )
            st.plotly_chart(fig_t2, use_container_width=True)



        # Trending products table
        st.markdown("#### 🔥 Top Trending Products (Weighted by Preferences)")
        trending_prods = (
            trend_filtered[trend_filtered["trend_label"] == "trending"]
            .groupby(["product_name", "category", "platform"])
            .agg(avg_daily_demand=("adj_daily_demand", "mean"), rolling_7d=("adj_rolling_7d", "mean"))
            .reset_index()
            .sort_values("avg_daily_demand", ascending=False)
        )
        if not trending_prods.empty:
            trending_prods["avg_daily_demand"] = trending_prods["avg_daily_demand"].round(1)
            trending_prods["rolling_7d"] = trending_prods["rolling_7d"].round(1)
            st.dataframe(
                trending_prods.rename(columns={
                    "product_name": "Product", "category": "Category",
                    "platform": "Platform", "avg_daily_demand": "Adj Daily Demand",
                    "rolling_7d": "Adj 7-Day Rolling",
                }),
                use_container_width=True, height=300,
            )
        else:
            st.info("No trending products found in the selected data.")
    else:
        st.info("No trend data for selected products.")
else:
    st.warning("Trend labels data not found. Run `ml_pipeline.py` to generate it.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='text-align:center; color:#64748b; font-size:0.85rem;'>
    QuickCommerce Platform Recommender &bull; Last refreshed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    &bull; Data: Blinkit · Zepto · Swiggy Instamart
    &bull; Models: GBT Demand Forecaster + RF Trend Classifier
</div>
""", unsafe_allow_html=True)
