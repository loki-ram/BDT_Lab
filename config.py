"""
config.py — Shared constants, paths, and product registry for the QuickCommerce Analytics system.
Location Scope: Jayanagar, Bangalore (fixed, single-location).
"""

import os

# ─────────────────────────────────────────────
# LOCATION
# ─────────────────────────────────────────────
LOCATION = "Jayanagar, Bangalore"

# ─────────────────────────────────────────────
# FILE PATHS (local development)
# In production, replace with S3 paths via boto3
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "output", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "output", "processed")
CURATED_DIR = os.path.join(BASE_DIR, "output", "curated")
MODELS_DIR = os.path.join(BASE_DIR, "output", "models")

# Raw platform files
BLINKIT_RAW = os.path.join(RAW_DIR, "blinkit_raw.parquet")
ZEPTO_RAW = os.path.join(RAW_DIR, "zepto_raw.parquet")
SWIGGY_RAW = os.path.join(RAW_DIR, "swiggy_raw.parquet")

# Processed outputs
UNIFIED_PARQUET = os.path.join(PROCESSED_DIR, "unified.parquet")
PRICE_ANALYTICS = os.path.join(PROCESSED_DIR, "price_analytics.parquet")
DELIVERY_ANALYTICS = os.path.join(PROCESSED_DIR, "delivery_analytics.parquet")
REVENUE_ANALYTICS = os.path.join(PROCESSED_DIR, "revenue_analytics.parquet")
STOCK_ANALYTICS = os.path.join(PROCESSED_DIR, "stock_analytics.parquet")

# Curated outputs
DEMAND_FORECASTS = os.path.join(CURATED_DIR, "demand_forecasts.parquet")
TREND_LABELS = os.path.join(CURATED_DIR, "trend_labels.parquet")
PLATFORM_COMPARISON = os.path.join(CURATED_DIR, "platform_comparison.parquet")

# ─────────────────────────────────────────────
# PRODUCT REGISTRY (canonical names)
# ─────────────────────────────────────────────
PRODUCT_CATALOGUE = [
    # (product_name, category)
    ("Milk", "Dairy"), ("Curd", "Dairy"), ("Buttermilk", "Dairy"),
    ("Paneer", "Dairy"), ("Butter", "Dairy"), ("Cheese Slices", "Dairy"),
    ("Tomato", "Vegetables"), ("Onion", "Vegetables"), ("Potato", "Vegetables"),
    ("Carrot", "Vegetables"), ("Capsicum", "Vegetables"), ("Beans", "Vegetables"),
    ("Banana", "Fruits"), ("Apple", "Fruits"), ("Mango", "Fruits"),
    ("Grapes", "Fruits"), ("Pomegranate", "Fruits"),
    ("Lays Chips", "Snacks"), ("Parle-G", "Snacks"), ("Maggi Noodles", "Snacks"),
    ("Kurkure", "Snacks"), ("Dark Fantasy", "Snacks"),
    ("Coca Cola", "Beverages"), ("Tropicana", "Beverages"),
    ("Bisleri Water", "Beverages"), ("Red Bull", "Beverages"),
    ("Basmati Rice", "Staples"), ("Toor Dal", "Staples"),
    ("Whole Wheat Atta", "Staples"), ("Salt", "Staples"),
    ("Sunflower Oil", "Staples"), ("Mustard Oil", "Staples"),
    ("Dettol Handwash", "Personal Care"), ("Shampoo H&S", "Personal Care"),
    ("Face Wash", "Personal Care"), ("Dove Soap", "Personal Care"),
    ("Colgate", "Personal Care"),
    ("Vim Dishwash", "Household"), ("Harpic", "Household"), ("Lizol", "Household"),
    ("Pampers S", "Baby Care"), ("Johnson Baby Oil", "Baby Care"),
    ("Baby Wipes", "Baby Care"), ("Cerelac", "Baby Care"),
    ("Pedigree Adult", "Pet Food"), ("Whiskas Cat", "Pet Food"),
    ("Dog Treats", "Pet Food"),
    ("Crocin 500mg", "Healthcare"), ("Dettol Sanitizer", "Healthcare"),
    ("Band Aid", "Healthcare"), ("ORS Sachet", "Healthcare"),
]

PRODUCT_NAMES = [p[0] for p in PRODUCT_CATALOGUE]
CATEGORIES = sorted(set(p[1] for p in PRODUCT_CATALOGUE))

# ─────────────────────────────────────────────
# PLATFORM CONFIG
# ─────────────────────────────────────────────
PLATFORMS = ["blinkit", "zepto", "swiggy"]

PLATFORM_DISPLAY_NAMES = {
    "blinkit": "Blinkit",
    "zepto": "Zepto",
    "swiggy": "Swiggy Instamart",
}

PLATFORM_COLORS = {
    "blinkit": "#F8C100",   # Blinkit yellow
    "zepto": "#7B2FF7",     # Zepto purple
    "swiggy": "#FC8019",    # Swiggy orange
}

# ─────────────────────────────────────────────
# COLUMN MAPPING — platform raw → unified canonical schema
# ─────────────────────────────────────────────
BLINKIT_COLUMN_MAP = {
    "record_id": "record_id",
    "snapshot_time": "snapshot_time",
    "product_name": "product_name",
    "product_category": "category",
    "pack_size": "pack_size",
    "pack_unit": "pack_unit",
    "city_zone": "location",
    "cost": "mrp",
    "deal_discount_pct": "discount_pct",
    "final_cost": "effective_price",
    "cost_per_unit": "price_per_unit",
    "stock_status": "in_stock",         # needs transformation: "available" → True
    "stock_remaining": "stock_remaining",
    "units_ordered": "units_ordered",
    "gross_revenue": "revenue_inr",
    "mins_to_deliver": "delivery_minutes",
    "customer_rating": "rating",
    "hour_of_day": "hour_of_day",
    "is_weekend": "is_weekend",
    "is_rush_hour": "is_peak_hour",
    "month_num": "month",
    "weather_condition": "weather",
    "raining": "is_raining",
    "seasonal_factor": "seasonal_factor",
    "bulk_saving_pct": "bulk_discount_pct",
    "festival_demand": "festival_demand_mult",
}

ZEPTO_COLUMN_MAP = {
    "txn_id": "record_id",
    "recorded_at": "snapshot_time",
    "item_name": "product_name",
    "item_type": "category",
    "quantity_ml_g": "pack_size",
    "quantity_unit": "pack_unit",
    "delivery_zone": "location",
    "mrp": "mrp",
    "discount_percent": "discount_pct",
    "selling_price": "effective_price",
    "price_per_unit": "price_per_unit",
    "available": "in_stock",            # needs transformation: "yes" → True
    "stock_remaining": "stock_remaining",
    "qty_ordered": "units_ordered",
    "revenue_inr": "revenue_inr",
    "eta_mins": "delivery_minutes",
    "app_rating": "rating",
    "order_hour": "hour_of_day",
    "weekend_flag": "is_weekend",       # needs transformation: 1 → True
    "peak_slot": "is_peak_hour",        # needs transformation: 1 → True
    "month": "month",
    "weather": "weather",
    "is_raining": "is_raining",
    "price_seasonal_mult": "seasonal_factor",
    "bulk_discount_applied": "bulk_discount_pct",
    "festival_demand": "festival_demand_mult",
}

SWIGGY_COLUMN_MAP = {
    "order_ref": "record_id",
    "timestamp": "snapshot_time",
    "product": "product_name",
    "category": "category",
    "pack_qty": "pack_size",
    "unit": "pack_unit",
    "location": "location",
    "mrp": "mrp",
    "discount_pct": "discount_pct",
    "offer_price": "effective_price",
    "per_unit_price": "price_per_unit",
    "in_stock": "in_stock",             # boolean passthrough
    "stock_remaining": "stock_remaining",
    "num_ordered": "units_ordered",
    "total_revenue": "revenue_inr",
    "delivery_time": "delivery_minutes", # needs transformation: "15 mins" → 15
    "rating": "rating",
    "hour": "hour_of_day",
    "is_weekend": "is_weekend",
    "is_peak_hour": "is_peak_hour",
    "month": "month",
    "weather": "weather",
    "is_raining": "is_raining",
    "seasonal_multiplier": "seasonal_factor",
    "bulk_discount": "bulk_discount_pct",
    "festival_demand": "festival_demand_mult",
}

# ─────────────────────────────────────────────
# API KEYS (set via environment variables)
# ─────────────────────────────────────────────
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")

# ─────────────────────────────────────────────
# DEFAULT SLIDER VALUES
# ─────────────────────────────────────────────
DEFAULT_PRICE_WEIGHT = 50
DEFAULT_DELIVERY_WEIGHT = 30
DEFAULT_QUALITY_WEIGHT = 20
