import pandas as pd
import numpy as np
import random
import uuid
import os
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# SHARED CONFIG
# ─────────────────────────────────────────────

LOCATION = "Jayanagar, Bangalore"

PRODUCTS = [
    # ── Dairy (liquid) ── base unit: ml
    ("Milk",            "Dairy",         62,   1000, "ml",  [250, 500, 1000]),
    ("Curd",            "Dairy",         90,   1000, "ml",  [200, 400, 500, 1000]),
    ("Buttermilk",      "Dairy",         30,   500,  "ml",  [200, 500, 1000]),

    # ── Dairy (solid) ── base unit: g
    ("Paneer",          "Dairy",         90,   200,  "g",   [100, 200, 500]),
    ("Butter",          "Dairy",         55,   100,  "g",   [100, 200, 500]),
    ("Cheese Slices",   "Dairy",        120,   200,  "g",   [100, 200, 400]),

    # ── Vegetables ── base unit: g
    ("Tomato",          "Vegetables",    40,   1000, "g",   [250, 500, 1000, 2000]),
    ("Onion",           "Vegetables",    35,   1000, "g",   [250, 500, 1000, 2000]),
    ("Potato",          "Vegetables",    30,   1000, "g",   [250, 500, 1000, 2000]),
    ("Carrot",          "Vegetables",    50,   1000, "g",   [250, 500, 1000]),
    ("Capsicum",        "Vegetables",    80,   500,  "g",   [250, 500, 1000]),
    ("Beans",           "Vegetables",    60,   500,  "g",   [250, 500, 1000]),

    # ── Fruits ── base unit: g
    ("Banana",          "Fruits",        60,   1000, "g",   [500, 1000]),
    ("Apple",           "Fruits",       200,   1000, "g",   [500, 1000, 2000]),
    ("Mango",           "Fruits",       120,   1000, "g",   [500, 1000, 2000]),
    ("Grapes",          "Fruits",        80,   500,  "g",   [250, 500, 1000]),
    ("Pomegranate",     "Fruits",       180,   1000, "g",   [500, 1000]),

    # ── Snacks ── base unit: g
    ("Lays Chips",      "Snacks",        40,   100,  "g",   [50, 100, 200]),
    ("Parle-G",         "Snacks",        10,   100,  "g",   [100, 200, 500]),
    ("Maggi Noodles",   "Snacks",        14,   70,   "g",   [70, 140, 280]),
    ("Kurkure",         "Snacks",        20,   90,   "g",   [90, 180]),
    ("Dark Fantasy",    "Snacks",        35,   75,   "g",   [75, 150, 300]),

    # ── Beverages ── base unit: ml
    ("Coca Cola",       "Beverages",     60,   750,  "ml",  [250, 500, 750, 2000]),
    ("Tropicana",       "Beverages",    120,   1000, "ml",  [200, 500, 1000]),
    ("Bisleri Water",   "Beverages",     20,   1000, "ml",  [500, 1000, 2000]),
    ("Red Bull",        "Beverages",    125,   250,  "ml",  [250, 500]),

    # ── Staples (solid) ── base unit: g
    ("Basmati Rice",    "Staples",      130,   1000, "g",   [500, 1000, 2000, 5000]),
    ("Toor Dal",        "Staples",      150,   1000, "g",   [500, 1000, 2000]),
    ("Whole Wheat Atta","Staples",       50,   1000, "g",   [1000, 2000, 5000]),
    ("Salt",            "Staples",       22,   1000, "g",   [500, 1000, 2000]),

    # ── Staples (liquid) ── base unit: ml
    ("Sunflower Oil",   "Staples",      145,   1000, "ml",  [500, 1000, 2000, 5000]),
    ("Mustard Oil",     "Staples",      180,   1000, "ml",  [500, 1000, 2000]),

    # ── Personal Care (liquid) ── base unit: ml
    ("Dettol Handwash", "Personal Care",115,   200,  "ml",  [100, 200, 500]),
    ("Shampoo H&S",     "Personal Care",175,   180,  "ml",  [90, 180, 340]),
    ("Face Wash",       "Personal Care",180,   100,  "ml",  [50, 100, 200]),

    # ── Personal Care (solid) ── base unit: g
    ("Dove Soap",       "Personal Care", 55,   100,  "g",   [75, 100, 125]),
    ("Colgate",         "Personal Care",110,   200,  "g",   [100, 200, 500]),

    # ── Household ── base unit: ml
    ("Vim Dishwash",    "Household",     85,   500,  "ml",  [250, 500, 1000]),
    ("Harpic",          "Household",    115,   500,  "ml",  [200, 500, 1000]),
    ("Lizol",           "Household",    145,   500,  "ml",  [200, 500, 1000]),

    # ── Fixed-pack: Baby Care ──
    ("Pampers S",       "Baby Care",    399,   20,   "pcs", [20]),
    ("Johnson Baby Oil","Baby Care",    175,   100,  "ml",  [100]),
    ("Baby Wipes",      "Baby Care",    199,   72,   "pcs", [72]),
    ("Cerelac",         "Baby Care",    210,   200,  "g",   [200]),

    # ── Fixed-pack: Pet Food ──
    ("Pedigree Adult",  "Pet Food",     155,   400,  "g",   [400]),
    ("Whiskas Cat",     "Pet Food",      45,   85,   "g",   [85]),
    ("Dog Treats",      "Pet Food",     120,   100,  "g",   [100]),

    # ── Fixed-pack: Healthcare ──
    ("Crocin 500mg",    "Healthcare",    28,   15,   "tabs",[15]),
    ("Dettol Sanitizer","Healthcare",    65,   50,   "ml",  [50]),
    ("Band Aid",        "Healthcare",    55,   10,   "pcs", [10]),
    ("ORS Sachet",      "Healthcare",    35,   5,    "pcs", [5]),
]


SEASONAL_MULTIPLIERS = {
    "Mango":  {1:1.8,2:1.6,3:1.2,4:0.7,5:0.6,6:0.65,7:1.0,8:1.3,9:1.5,10:1.6,11:1.7,12:1.9},
    "Tomato": {1:1.4,2:1.1,3:0.9,4:0.8,5:0.9,6:1.0, 7:1.1,8:1.0,9:1.0,10:1.1,11:1.4,12:1.5},
    "Onion":  {1:1.0,2:0.9,3:0.9,4:0.8,5:0.9,6:1.0, 7:1.1,8:1.2,9:1.5,10:1.4,11:1.3,12:1.1},
}

WEATHER_BY_MONTH = {
    1:("Clear",0.05,1.00), 2:("Clear",0.05,1.00), 3:("Partly Cloudy",0.10,1.02),
    4:("Hot",0.15,1.05),   5:("Hot",0.20,1.05),   6:("Rainy",0.75,1.30),
    7:("Rainy",0.85,1.35), 8:("Rainy",0.80,1.30), 9:("Rainy",0.60,1.20),
    10:("Partly Cloudy",0.30,1.10), 11:("Clear",0.10,1.02), 12:("Clear",0.05,1.00),
}

FESTIVAL_DATES = {
    # (month, day): (festival_name, demand_multiplier, price_multiplier, duration_days)
    (1, 1): ("New Year", 1.8, 1.15, 3),
    (3, 8): ("Holi", 1.6, 1.10, 2),
    (3, 25): ("IPL Season Start", 1.4, 1.05, 60),  # IPL runs ~60 days
    (10, 24): ("Diwali", 2.2, 1.20, 5),
    (12, 25): ("Christmas", 1.5, 1.12, 3),
}

DEMAND_PROFILE = {
    # category     : (order_qty_options, weights)

    "Dairy"        : ([1,2,3],   [0.50,0.35,0.15]),
    "Vegetables"   : ([1,2,3,4], [0.40,0.35,0.15,0.10]),
    "Fruits"       : ([1,2],     [0.65,0.35]),
    "Snacks"       : ([1,2,3,5], [0.45,0.30,0.15,0.10]),
    "Beverages"    : ([1,2,4,6], [0.40,0.35,0.15,0.10]),
    "Staples"      : ([1,2],     [0.70,0.30]),
    "Personal Care": ([1,2],     [0.75,0.25]),
    "Household"    : ([1,2],     [0.80,0.20]),
    "Baby Care"    : ([1,2],     [0.85,0.15]),
    "Pet Food"     : ([1,2],     [0.80,0.20]),
    "Healthcare"   : ([1],       [1.00]),
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_time_multiplier(hour):
    if 8 <= hour <= 10:           return 1.35
    elif 12 <= hour <= 14:        return 1.20
    elif 18 <= hour <= 21:        return 1.45
    elif hour >= 22 or hour <= 6: return 1.60
    else:                         return 1.00

def get_price_surge(hour):
    if 18 <= hour <= 21:  return random.uniform(1.02, 1.08)
    elif 8 <= hour <= 10: return random.uniform(1.01, 1.05)
    else:                 return 1.0

def get_festival_multiplier(timestamp):
    """Returns (demand_mult, price_mult) for festival dates."""
    month, day = timestamp.month, timestamp.day
    
    for (fest_month, fest_day), (name, demand_mult, price_mult, duration) in FESTIVAL_DATES.items():
        # Check if current date is within festival window
        festival_start = datetime(timestamp.year, fest_month, fest_day)
        festival_end = festival_start + timedelta(days=duration)
        if festival_start <= timestamp < festival_end:
            return demand_mult, price_mult
    
    return 1.0, 1.0

def get_order_qty(category, is_weekend, hour):
    options, weights = DEMAND_PROFILE.get(category, ([1],[1.0]))
    if is_weekend and 8 <= hour <= 12 and category in ("Dairy","Vegetables","Fruits","Staples"):
        weights = [w*(1.3 if i > 0 else 1.0) for i,w in enumerate(weights)]
    total   = sum(weights)
    weights = [w/total for w in weights]
    return random.choices(options, weights=weights, k=1)[0]

def compute_base_values(product_info, timestamp, price_factor_range,
                        delivery_base_range, rating_mean, stock_rate, discount_rate):
    name, category, base_price, base_qty, base_unit, available_qtys = product_info
    hour       = timestamp.hour
    month      = timestamp.month
    is_weekend = timestamp.weekday() >= 5

    weather_label, rain_prob, weather_delay = WEATHER_BY_MONTH[month]
    is_raining    = random.random() < rain_prob
    seasonal_mult = SEASONAL_MULTIPLIERS.get(name, {}).get(month, 1.0)

    results = []
    for chosen_qty in available_qtys:
        qty_ratio  = chosen_qty / base_qty
        unit_label = f"{chosen_qty}{base_unit}"

        max_qty       = max(available_qtys) if len(available_qtys) > 1 else base_qty
        denom         = max(max_qty - base_qty, 1)
        bulk_discount = 1.0 - (0.05*(chosen_qty-base_qty)/denom) if chosen_qty > base_qty else 1.0
        bulk_discount = max(0.90, bulk_discount)

        pf_low, pf_high = price_factor_range
        price_factor = random.uniform(pf_low, pf_high)
        surge        = get_price_surge(hour)
        weekend_mult = 1.04 if (is_weekend and category in ("Dairy","Staples") and 8<=hour<=11) else 1.0
        festival_demand_mult, festival_price_mult = get_festival_multiplier(timestamp)

        total_before = round(
            base_price * qty_ratio * price_factor * surge
            * seasonal_mult * weekend_mult * festival_price_mult * bulk_discount, 2
        )

        discount_chance = discount_rate
        if is_raining and category == "Beverages":
            discount_chance += 0.10
        discount_pct   = random.choice([5,10,15,20]) if random.random() < discount_chance else 0
        total_price    = round(total_before*(1-discount_pct/100), 2) if discount_pct else total_before
        price_per_unit = round(total_price/chosen_qty, 4)

        dl_low, dl_high = delivery_base_range
        base_delivery = random.randint(dl_low, dl_high)
        time_mult     = get_time_multiplier(hour)
        weekend_delay = 1.15 if is_weekend else 1.0
        rain_delay    = weather_delay if is_raining else 1.0
        delivery_time = int(base_delivery * time_mult * weekend_delay * rain_delay)
        delivery_time = max(7, min(delivery_time, 60))

        rating_adj = rating_mean - (0.15 if is_raining else 0)
        rating     = round(np.clip(np.random.normal(rating_adj, 0.3), 1.0, 5.0), 1)

        stock_penalty  = 0.07 if is_weekend else 0.0
        stock_penalty += 0.05 if get_time_multiplier(hour) > 1.2 else 0.0
        stock_penalty += 0.03 if chosen_qty > base_qty else 0.0
        
        # Calculate stock_remaining instead of just boolean
        base_stock = random.randint(50, 200)
        stock_remaining = int(base_stock * max(0.5, stock_rate - stock_penalty))
        in_stock = stock_remaining > 0

        base_order_qty = get_order_qty(category, is_weekend, hour) if in_stock else 0
        order_qty = min(int(base_order_qty * festival_demand_mult), stock_remaining)
        revenue   = round(total_price * order_qty, 2)

        results.append({
            "name": name, "category": category,
            "base_qty": base_qty, "base_unit": base_unit,
            "chosen_qty": chosen_qty, "unit_label": unit_label,
            "seasonal_mult": round(seasonal_mult, 2),
            "bulk_discount": round(1-bulk_discount, 3),
            "total_before": total_before,
            "discount_pct": discount_pct, "total_price": total_price,
            "price_per_unit": price_per_unit,
            "delivery_time": delivery_time, "rating": rating,
            "in_stock": in_stock, "stock_remaining": stock_remaining, "order_qty": order_qty, "revenue": revenue,
            "hour": hour, "month": month, "is_weekend": is_weekend,
            "is_peak_hour": get_time_multiplier(hour) > 1.1,
            "weather": weather_label, "is_raining": is_raining,
            "day_of_week": timestamp.strftime("%A"),
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "festival_demand_mult": round(festival_demand_mult, 2),
        })
    return results

# ─────────────────────────────────────────────
# ROW BUILDERS — one per platform schema
# ─────────────────────────────────────────────

def build_blinkit_row(r):
    return {
        "record_id":         str(uuid.uuid4()),
        "snapshot_time":     r["timestamp"],
        "product_name":      r["name"],
        "product_category":  r["category"],
        "pack_size":         r["chosen_qty"],
        "pack_unit":         r["base_unit"],
        "pack_label":        r["unit_label"],
        "city_zone":         LOCATION,
        "cost":              r["total_before"],
        "deal_discount_pct": r["discount_pct"],
        "final_cost":        r["total_price"],
        "cost_per_unit":     r["price_per_unit"],
        "stock_status":      "available" if r["in_stock"] else "out_of_stock",
        "stock_remaining":   r["stock_remaining"],
        "units_ordered":     r["order_qty"],
        "gross_revenue":     r["revenue"],
        "mins_to_deliver":   r["delivery_time"],
        "customer_rating":   r["rating"],
        "hour_of_day":       r["hour"],
        "weekday":           r["day_of_week"],
        "is_weekend":        r["is_weekend"],
        "is_rush_hour":      r["is_peak_hour"],
        "month_num":         r["month"],
        "weather_condition": r["weather"],
        "raining":           r["is_raining"],
        "seasonal_factor":   r["seasonal_mult"],
        "bulk_saving_pct":   r["bulk_discount"],
        "festival_demand":   r["festival_demand_mult"],
    }

def build_zepto_row(r):
    return {
        "txn_id":                str(uuid.uuid4()),
        "recorded_at":           r["timestamp"],
        "item_name":             r["name"],
        "item_type":             r["category"],
        "quantity_ml_g":         r["chosen_qty"],
        "quantity_unit":         r["base_unit"],
        "qty_label":             r["unit_label"],
        "delivery_zone":         LOCATION,
        "mrp":                   r["total_before"],
        "discount_percent":      r["discount_pct"],
        "selling_price":         r["total_price"],
        "price_per_unit":        r["price_per_unit"],
        "available":             "yes" if r["in_stock"] else "no",
        "stock_remaining":       r["stock_remaining"],
        "qty_ordered":           r["order_qty"],
        "revenue_inr":           r["revenue"],
        "eta_mins":              r["delivery_time"],
        "app_rating":            r["rating"],
        "order_hour":            r["hour"],
        "day_name":              r["day_of_week"],
        "weekend_flag":          1 if r["is_weekend"] else 0,
        "peak_slot":             1 if r["is_peak_hour"] else 0,
        "month":                 r["month"],
        "weather":               r["weather"],
        "is_raining":            r["is_raining"],
        "price_seasonal_mult":   r["seasonal_mult"],
        "bulk_discount_applied": r["bulk_discount"],
        "festival_demand":       r["festival_demand_mult"],
    }

def build_swiggy_row(r):
    return {
        "order_ref":           str(uuid.uuid4()),
        "timestamp":           r["timestamp"],
        "product":             r["name"],
        "category":            r["category"],
        "pack_qty":            r["chosen_qty"],
        "unit":                r["base_unit"],
        "pack_description":    r["unit_label"],
        "location":            LOCATION,
        "mrp":                 r["total_before"],
        "discount_pct":        r["discount_pct"],
        "offer_price":         r["total_price"],
        "per_unit_price":      r["price_per_unit"],
        "in_stock":            r["in_stock"],
        "stock_remaining":     r["stock_remaining"],
        "num_ordered":         r["order_qty"],
        "total_revenue":       r["revenue"],
        "delivery_time":       f"{r['delivery_time']} mins",
        "rating":              r["rating"],
        "hour":                r["hour"],
        "day_of_week":         r["day_of_week"],
        "is_weekend":          r["is_weekend"],
        "is_peak_hour":        r["is_peak_hour"],
        "month":               r["month"],
        "weather":             r["weather"],
        "is_raining":          r["is_raining"],
        "seasonal_multiplier": r["seasonal_mult"],
        "bulk_discount":       r["bulk_discount"],
        "festival_demand":     r["festival_demand_mult"],
    }

# ─────────────────────────────────────────────
# CORE: BATCH WRITER
# Processes timestamps in chunks, writes each
# chunk as a Parquet part file, then clears RAM.
# Never holds the full dataset in memory at once.
# Peak RAM usage ≈ one chunk (~150-200MB for 7d)
# ─────────────────────────────────────────────

def generate_platform_batched(
    platform_name,
    timestamps,
    price_factor_range,
    delivery_base_range,
    rating_mean,
    stock_rate,
    discount_rate,
    row_builder,
    output_dir,
    days_per_chunk=7,       # lower to 3 if still OOM
):
    import pyarrow.parquet as pq
    import pyarrow as pa
    import shutil

    parts_dir = os.path.join(output_dir, f"{platform_name}_parts")
    os.makedirs(parts_dir, exist_ok=True)

    # 96 slots/day for 15-min intervals, 48 for 30-min
    slots_per_chunk = days_per_chunk * 96
    chunks = [timestamps[i:i+slots_per_chunk]
              for i in range(0, len(timestamps), slots_per_chunk)]

    print(f"\nGenerating {platform_name}: "
          f"{len(chunks)} chunks × {days_per_chunk} days each")

    total_records = 0
    part_files    = []

    for chunk_idx, chunk_ts in enumerate(chunks):
        records = []
        for ts in chunk_ts:
            for product in PRODUCTS:
                rows = compute_base_values(
                    product, ts,
                    price_factor_range, delivery_base_range,
                    rating_mean, stock_rate, discount_rate
                )
                for r in rows:
                    records.append(row_builder(r))

        df_chunk       = pd.DataFrame(records)
        total_records += len(df_chunk)

        part_path = os.path.join(parts_dir, f"part_{chunk_idx:04d}.parquet")
        df_chunk.to_parquet(part_path, index=False, engine="pyarrow")
        part_files.append(part_path)

        # Free memory immediately — critical for 8GB RAM
        del records, df_chunk

        print(f"  [{platform_name}] chunk {chunk_idx+1}/{len(chunks)} "
              f"| records so far: {total_records:,}")

    # ── Stream-merge all part files into one final Parquet ──
    # Uses PyArrow writer — reads one part at a time, never all at once
    print(f"  [{platform_name}] Merging {len(part_files)} parts...")
    final_path = os.path.join(output_dir, f"{platform_name}_raw.parquet")
    writer = None
    try:
        for pf in part_files:
            table = pq.read_table(pf)
            if writer is None:
                writer = pq.ParquetWriter(final_path, table.schema)
            writer.write_table(table)
            del table
    finally:
        if writer:
            writer.close()

    # ── Sample CSV (500 rows for quick inspection) ──
    sample = pq.read_table(final_path).to_pandas().sample(500, random_state=42)
    sample.to_csv(os.path.join(output_dir, f"{platform_name}_sample.csv"), index=False)
    del sample

    # ── Cleanup part files ──
    shutil.rmtree(parts_dir)

    size_mb = os.path.getsize(final_path) / 1e6
    print(f"  [{platform_name}] Done — {total_records:,} records | {size_mb:.1f} MB")
    return total_records


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR = "output/raw"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Full year at 15-min intervals → ~17-20M records across 3 platforms
    # To reduce size: change to timedelta(minutes=30) and range(365*96)
    start_dt   = datetime(2024, 1, 1, 6, 0, 0)
    timestamps = [start_dt + timedelta(minutes=15*i) for i in range(365*96)]

    print(f"Timestamps : {len(timestamps):,}  (365 days × 96 slots/day)")
    print(f"Expected   : ~{len(timestamps) * len(PRODUCTS) * 3:,} total records across 3 platforms")
    print(f"Chunk size : 7 days = ~{7*96*len(PRODUCTS):,} records per chunk per platform\n")

    generate_platform_batched(
        "blinkit", timestamps,
        price_factor_range=(1.05, 1.15), delivery_base_range=(10, 18),
        rating_mean=4.3, stock_rate=0.92, discount_rate=0.15,
        row_builder=build_blinkit_row, output_dir=OUTPUT_DIR, days_per_chunk=7,
    )

    generate_platform_batched(
        "zepto", timestamps,
        price_factor_range=(0.95, 1.05), delivery_base_range=(8, 15),
        rating_mean=4.1, stock_rate=0.88, discount_rate=0.35,
        row_builder=build_zepto_row, output_dir=OUTPUT_DIR, days_per_chunk=7,
    )

    generate_platform_batched(
        "swiggy", timestamps,
        price_factor_range=(1.00, 1.10), delivery_base_range=(12, 22),
        rating_mean=4.2, stock_rate=0.85, discount_rate=0.15,
        row_builder=build_swiggy_row, output_dir=OUTPUT_DIR, days_per_chunk=7,
    )

    print("\n── Final Output Files ──")
    for fname in ["blinkit_raw.parquet", "zepto_raw.parquet", "swiggy_raw.parquet"]:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            print(f"  {fname}: {os.path.getsize(path)/1e6:.1f} MB")

    print("\nDone! Upload to S3:")
    print("  aws s3 cp output/raw/ s3://your-bucket/raw/ --recursive")