"""
preprocessing.py — PySpark schema normalisation and analytics pipeline.

Reads raw Parquet files (Blinkit, Zepto, Swiggy Instamart),
normalises schemas into a unified canonical format,
and produces analytics aggregation tables.

Can run on Databricks (with S3 paths) or locally (with local paths).
"""

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType,
    IntegerType, BooleanType, TimestampType
)

# Add parent dir to path for config import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    BLINKIT_RAW, ZEPTO_RAW, SWIGGY_RAW,
    BLINKIT_COLUMN_MAP, ZEPTO_COLUMN_MAP, SWIGGY_COLUMN_MAP,
    PROCESSED_DIR, UNIFIED_PARQUET,
    PRICE_ANALYTICS, DELIVERY_ANALYTICS,
    REVENUE_ANALYTICS, STOCK_ANALYTICS,
    LOCATION
)


def create_spark_session():
    """Create a Spark session for local or Databricks execution."""
    return (
        SparkSession.builder
        .appName("QuickCommerce_Preprocessing")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.parquet.enableVectorizedReader", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def rename_columns(df, column_map):
    """Rename DataFrame columns using the provided mapping."""
    for old_name, new_name in column_map.items():
        if old_name in df.columns:
            df = df.withColumnRenamed(old_name, new_name)
    return df


def normalise_blinkit(spark, path):
    """Read and normalise Blinkit raw data."""
    print(f"  Reading Blinkit from: {path}")
    df = spark.read.parquet(path)
    print(f"  Blinkit raw records: {df.count():,}")

    # Rename columns to canonical schema
    df = rename_columns(df, BLINKIT_COLUMN_MAP)

    # Add platform column
    df = df.withColumn("platform", F.lit("blinkit"))

    # Normalise stock_status: "available" → True, "out_of_stock" → False
    df = df.withColumn(
        "in_stock",
        F.when(F.col("in_stock") == "available", True).otherwise(False)
    )

    # Parse snapshot_time to timestamp
    df = df.withColumn("snapshot_time", F.to_timestamp("snapshot_time"))

    # Ensure boolean columns
    df = df.withColumn("is_weekend", F.col("is_weekend").cast(BooleanType()))
    df = df.withColumn("is_peak_hour", F.col("is_peak_hour").cast(BooleanType()))
    df = df.withColumn("is_raining", F.col("is_raining").cast(BooleanType()))

    return df


def normalise_zepto(spark, path):
    """Read and normalise Zepto raw data."""
    print(f"  Reading Zepto from: {path}")
    df = spark.read.parquet(path)
    print(f"  Zepto raw records: {df.count():,}")

    # Rename columns to canonical schema
    df = rename_columns(df, ZEPTO_COLUMN_MAP)

    # Add platform column
    df = df.withColumn("platform", F.lit("zepto"))

    # Normalise stock availability: "yes" → True, "no" → False
    df = df.withColumn(
        "in_stock",
        F.when(F.col("in_stock") == "yes", True).otherwise(False)
    )

    # Normalise weekend_flag: int 0/1 → boolean
    df = df.withColumn("is_weekend", F.col("is_weekend").cast(BooleanType()))

    # Normalise peak_slot: int 0/1 → boolean
    df = df.withColumn("is_peak_hour", F.col("is_peak_hour").cast(BooleanType()))

    # Parse snapshot_time to timestamp
    df = df.withColumn("snapshot_time", F.to_timestamp("snapshot_time"))

    # Ensure boolean for is_raining
    df = df.withColumn("is_raining", F.col("is_raining").cast(BooleanType()))

    return df


def normalise_swiggy(spark, path):
    """Read and normalise Swiggy Instamart raw data."""
    print(f"  Reading Swiggy from: {path}")
    df = spark.read.parquet(path)
    print(f"  Swiggy raw records: {df.count():,}")

    # Rename columns to canonical schema
    df = rename_columns(df, SWIGGY_COLUMN_MAP)

    # Add platform column
    df = df.withColumn("platform", F.lit("swiggy"))

    # Parse delivery_time: "15 mins" STRING → 15 INTEGER
    df = df.withColumn(
        "delivery_minutes",
        F.regexp_extract("delivery_minutes", r"(\d+)", 1).cast(IntegerType())
    )

    # Parse snapshot_time to timestamp
    df = df.withColumn("snapshot_time", F.to_timestamp("snapshot_time"))

    # Ensure boolean columns
    df = df.withColumn("in_stock", F.col("in_stock").cast(BooleanType()))
    df = df.withColumn("is_weekend", F.col("is_weekend").cast(BooleanType()))
    df = df.withColumn("is_peak_hour", F.col("is_peak_hour").cast(BooleanType()))
    df = df.withColumn("is_raining", F.col("is_raining").cast(BooleanType()))

    return df


# ─── Canonical column list (order matters for union) ───
CANONICAL_COLUMNS = [
    "record_id", "snapshot_time", "location", "product_name", "category",
    "pack_size", "pack_unit", "platform", "mrp", "discount_pct",
    "effective_price", "price_per_unit", "in_stock", "stock_remaining",
    "units_ordered", "revenue_inr", "delivery_minutes", "rating",
    "hour_of_day", "is_weekend", "is_peak_hour", "weather", "is_raining",
    "seasonal_factor", "bulk_discount_pct", "festival_demand_mult", "month",
]


def select_canonical(df):
    """Select only the canonical columns in the correct order."""
    available = set(df.columns)
    cols = []
    for col in CANONICAL_COLUMNS:
        if col in available:
            cols.append(col)
        else:
            # Add missing columns with null
            df = df.withColumn(col, F.lit(None))
            cols.append(col)
    return df.select(cols)


def unify_and_deduplicate(blinkit_df, zepto_df, swiggy_df):
    """Union all three DataFrames and deduplicate."""
    print("\n  Selecting canonical columns and union...")

    blinkit_norm = select_canonical(blinkit_df)
    zepto_norm = select_canonical(zepto_df)
    swiggy_norm = select_canonical(swiggy_df)

    unified = blinkit_norm.unionByName(zepto_norm).unionByName(swiggy_norm)
    total_before = unified.count()
    print(f"  Total records after union: {total_before:,}")

    # Deduplicate on (record_id, platform, snapshot_time)
    unified = unified.dropDuplicates(["record_id", "platform", "snapshot_time"])
    total_after = unified.count()
    print(f"  After deduplication: {total_after:,} (removed {total_before - total_after:,} dupes)")

    return unified


def compute_price_analytics(unified_df):
    """Compute price analytics: avg price by product × platform hourly."""
    print("  Computing price analytics...")
    return (
        unified_df
        .withColumn("date", F.to_date("snapshot_time"))
        .withColumn("hour", F.col("hour_of_day"))
        .groupBy("product_name", "category", "platform", "date", "hour", "pack_size", "pack_unit")
        .agg(
            F.avg("effective_price").alias("avg_price"),
            F.min("effective_price").alias("min_price"),
            F.max("effective_price").alias("max_price"),
            F.avg("price_per_unit").alias("avg_price_per_unit"),
            F.avg("discount_pct").alias("avg_discount_pct"),
            F.stddev("effective_price").alias("price_stddev"),
            F.count("*").alias("record_count"),
        )
    )


def compute_delivery_analytics(unified_df):
    """Compute delivery analytics: avg delivery time by platform and hour."""
    print("  Computing delivery analytics...")
    return (
        unified_df
        .withColumn("date", F.to_date("snapshot_time"))
        .groupBy("platform", "date", "hour_of_day", "is_weekend", "weather", "is_raining")
        .agg(
            F.avg("delivery_minutes").alias("avg_delivery_mins"),
            F.min("delivery_minutes").alias("min_delivery_mins"),
            F.max("delivery_minutes").alias("max_delivery_mins"),
            F.count("*").alias("record_count"),
        )
    )


def compute_revenue_analytics(unified_df):
    """Compute revenue and demand analytics by category × platform daily."""
    print("  Computing revenue analytics...")
    return (
        unified_df
        .withColumn("date", F.to_date("snapshot_time"))
        .groupBy("product_name", "category", "platform", "date", "pack_size")
        .agg(
            F.sum("revenue_inr").alias("total_revenue"),
            F.sum("units_ordered").alias("total_units_ordered"),
            F.avg("units_ordered").alias("avg_units_ordered"),
            F.count("*").alias("record_count"),
        )
    )


def compute_stock_analytics(unified_df):
    """Compute stock and availability analytics by product × platform."""
    print("  Computing stock analytics...")
    return (
        unified_df
        .withColumn("date", F.to_date("snapshot_time"))
        .groupBy("product_name", "category", "platform", "date", "hour_of_day", "is_peak_hour")
        .agg(
            F.avg(F.col("in_stock").cast("int")).alias("availability_rate"),
            F.sum(F.when(~F.col("in_stock"), 1).otherwise(0)).alias("stockout_count"),
            F.avg("stock_remaining").alias("avg_stock_remaining"),
            F.count("*").alias("record_count"),
        )
    )


def main():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("QuickCommerce Analytics — Preprocessing Pipeline")
    print(f"Location: {LOCATION}")
    print("=" * 60)

    # Create output directories
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Create Spark session
    spark = create_spark_session()

    try:
        # ── Step 1: Read and normalise each platform ──
        print("\n[Step 1] Reading and normalising platform data...")
        blinkit_df = normalise_blinkit(spark, BLINKIT_RAW)
        zepto_df = normalise_zepto(spark, ZEPTO_RAW)
        swiggy_df = normalise_swiggy(spark, SWIGGY_RAW)

        # ── Step 2: Union and deduplicate ──
        print("\n[Step 2] Unifying and deduplicating...")
        unified_df = unify_and_deduplicate(blinkit_df, zepto_df, swiggy_df)

        # Cache unified for multiple downstream operations
        unified_df.cache()

        # ── Step 3: Write unified parquet ──
        print(f"\n[Step 3] Writing unified Parquet → {UNIFIED_PARQUET}")
        unified_df.write.mode("overwrite").parquet(UNIFIED_PARQUET)
        print(f"  ✓ Unified Parquet written")

        # ── Step 4: Compute analytics aggregations ──
        print("\n[Step 4] Computing analytics aggregations...")

        price_df = compute_price_analytics(unified_df)
        price_df.write.mode("overwrite").parquet(PRICE_ANALYTICS)
        print(f"  ✓ Price analytics → {PRICE_ANALYTICS}")

        delivery_df = compute_delivery_analytics(unified_df)
        delivery_df.write.mode("overwrite").parquet(DELIVERY_ANALYTICS)
        print(f"  ✓ Delivery analytics → {DELIVERY_ANALYTICS}")

        revenue_df = compute_revenue_analytics(unified_df)
        revenue_df.write.mode("overwrite").parquet(REVENUE_ANALYTICS)
        print(f"  ✓ Revenue analytics → {REVENUE_ANALYTICS}")

        stock_df = compute_stock_analytics(unified_df)
        stock_df.write.mode("overwrite").parquet(STOCK_ANALYTICS)
        print(f"  ✓ Stock analytics → {STOCK_ANALYTICS}")

        # ── Summary ──
        print("\n" + "=" * 60)
        print("Preprocessing complete!")
        print(f"  Unified records: {unified_df.count():,}")
        print(f"  Platforms: {[r.platform for r in unified_df.select('platform').distinct().collect()]}")
        print(f"  Products: {unified_df.select('product_name').distinct().count()}")
        print(f"  Date range: {unified_df.agg(F.min('snapshot_time')).collect()[0][0]} → "
              f"{unified_df.agg(F.max('snapshot_time')).collect()[0][0]}")
        print("=" * 60)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
