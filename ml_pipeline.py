"""
ml_pipeline.py — Machine Learning pipeline for QuickCommerce Analytics.

Model 1: GBTRegressor — Demand Forecasting (units_ordered prediction)
Model 2: RandomForestClassifier — Trending Product Classification

Uses Spark MLlib. Runs on Databricks or locally.
Train: April–May | Test: June (time-based split per PRD).
"""

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    RegressionEvaluator,
    MulticlassClassificationEvaluator
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    UNIFIED_PARQUET, CURATED_DIR, MODELS_DIR,
    DEMAND_FORECASTS, TREND_LABELS, LOCATION
)


def create_spark_session():
    """Create a Spark session for ML pipeline."""
    return (
        SparkSession.builder
        .appName("QuickCommerce_ML_Pipeline")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def load_unified_data(spark):
    """Load the unified Parquet and prepare for ML."""
    print("  Loading unified Parquet...")
    df = spark.read.parquet(UNIFIED_PARQUET)
    print(f"  Total records: {df.count():,}")

    # Ensure numeric types
    df = (
        df
        .withColumn("units_ordered", F.col("units_ordered").cast(IntegerType()))
        .withColumn("effective_price", F.col("effective_price").cast(DoubleType()))
        .withColumn("mrp", F.col("mrp").cast(DoubleType()))
        .withColumn("price_per_unit", F.col("price_per_unit").cast(DoubleType()))
        .withColumn("delivery_minutes", F.col("delivery_minutes").cast(IntegerType()))
        .withColumn("rating", F.col("rating").cast(DoubleType()))
        .withColumn("discount_pct", F.col("discount_pct").cast(DoubleType()))
        .withColumn("seasonal_factor", F.col("seasonal_factor").cast(DoubleType()))
        .withColumn("bulk_discount_pct", F.col("bulk_discount_pct").cast(DoubleType()))
        .withColumn("festival_demand_mult", F.col("festival_demand_mult").cast(DoubleType()))
        .withColumn("hour_of_day", F.col("hour_of_day").cast(IntegerType()))
        .withColumn("month", F.col("month").cast(IntegerType()))
        .withColumn("pack_size", F.col("pack_size").cast(IntegerType()))
        .withColumn("stock_remaining", F.col("stock_remaining").cast(IntegerType()))
        # Cast booleans to integers for MLlib
        .withColumn("is_weekend_int", F.col("is_weekend").cast(IntegerType()))
        .withColumn("is_peak_hour_int", F.col("is_peak_hour").cast(IntegerType()))
        .withColumn("is_raining_int", F.col("is_raining").cast(IntegerType()))
        .withColumn("in_stock_int", F.col("in_stock").cast(IntegerType()))
        # Extract day_of_week as integer
        .withColumn("day_of_week", F.dayofweek("snapshot_time"))
    )

    # Drop nulls in critical columns
    df = df.na.drop(subset=["units_ordered", "effective_price", "hour_of_day"])

    return df


def compute_trend_labels(df):
    """
    Derive trend labels from units_ordered rolling averages.
    Products with increasing 7-day rolling avg → 'trending'
    Products with decreasing 7-day rolling avg → 'declining'
    Otherwise → 'stable'
    """
    from pyspark.sql.window import Window

    print("  Computing trend labels from rolling averages...")

    # Daily aggregation per product × platform
    daily = (
        df
        .withColumn("date", F.to_date("snapshot_time"))
        .groupBy("product_name", "category", "platform", "date", "month")
        .agg(
            F.sum("units_ordered").alias("daily_demand"),
            F.avg("effective_price").alias("daily_avg_price"),
        )
    )

    # 7-day rolling average
    window_7d = (
        Window
        .partitionBy("product_name", "platform")
        .orderBy("date")
        .rowsBetween(-6, 0)
    )

    window_prev_7d = (
        Window
        .partitionBy("product_name", "platform")
        .orderBy("date")
        .rowsBetween(-13, -7)
    )

    daily = (
        daily
        .withColumn("rolling_avg_7d", F.avg("daily_demand").over(window_7d))
        .withColumn("rolling_avg_prev_7d", F.avg("daily_demand").over(window_prev_7d))
    )

    # Compute trend: compare current 7d avg vs previous 7d avg
    daily = daily.withColumn(
        "trend_label",
        F.when(
            F.col("rolling_avg_7d") > F.col("rolling_avg_prev_7d") * 1.10,
            "trending"
        ).when(
            F.col("rolling_avg_7d") < F.col("rolling_avg_prev_7d") * 0.90,
            "declining"
        ).otherwise("stable")
    )

    # Fill nulls (first 13 days won't have prev window)
    daily = daily.na.fill({"trend_label": "stable"})

    return daily


def build_demand_forecast_pipeline():
    """Build the GBTRegressor pipeline for demand forecasting."""
    # String indexers for categorical features
    platform_indexer = StringIndexer(
        inputCol="platform", outputCol="platform_idx", handleInvalid="keep"
    )
    category_indexer = StringIndexer(
        inputCol="category", outputCol="category_idx", handleInvalid="keep"
    )
    product_indexer = StringIndexer(
        inputCol="product_name", outputCol="product_idx", handleInvalid="keep"
    )
    weather_indexer = StringIndexer(
        inputCol="weather", outputCol="weather_idx", handleInvalid="keep"
    )

    # Feature columns
    feature_cols = [
        "platform_idx", "category_idx", "product_idx", "weather_idx",
        "hour_of_day", "day_of_week", "month", "pack_size",
        "effective_price", "mrp", "price_per_unit", "discount_pct",
        "delivery_minutes", "rating", "seasonal_factor",
        "bulk_discount_pct", "festival_demand_mult",
        "is_weekend_int", "is_peak_hour_int", "is_raining_int", "in_stock_int",
        "stock_remaining",
    ]

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip"
    )

    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="units_ordered",
        maxIter=50,
        maxDepth=5,
        stepSize=0.1,
    )

    return Pipeline(stages=[
        platform_indexer, category_indexer, product_indexer, weather_indexer,
        assembler, gbt
    ])


def build_trend_classifier_pipeline():
    """Build the RandomForestClassifier pipeline for trend classification."""
    # Index the trend label (target)
    label_indexer = StringIndexer(
        inputCol="trend_label", outputCol="label", handleInvalid="keep"
    )

    # String indexers for categorical features
    platform_indexer = StringIndexer(
        inputCol="platform", outputCol="platform_idx", handleInvalid="keep"
    )
    category_indexer = StringIndexer(
        inputCol="category", outputCol="category_idx", handleInvalid="keep"
    )

    # Feature columns (from daily aggregation)
    feature_cols = [
        "platform_idx", "category_idx", "month",
        "daily_demand", "daily_avg_price",
        "rolling_avg_7d",
    ]

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip"
    )

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=100,
        maxDepth=8,
    )

    return Pipeline(stages=[
        label_indexer, platform_indexer, category_indexer,
        assembler, rf
    ])


def evaluate_demand_model(predictions):
    """Evaluate GBTRegressor using MAPE."""
    # Compute MAPE manually (Spark doesn't have a built-in MAPE evaluator)
    mape_df = (
        predictions
        .filter(F.col("units_ordered") > 0)
        .withColumn(
            "ape",
            F.abs(F.col("prediction") - F.col("units_ordered")) / F.col("units_ordered")
        )
    )
    mape = mape_df.agg(F.avg("ape")).collect()[0][0] * 100

    # Also compute RMSE and MAE
    rmse_eval = RegressionEvaluator(
        labelCol="units_ordered", predictionCol="prediction", metricName="rmse"
    )
    mae_eval = RegressionEvaluator(
        labelCol="units_ordered", predictionCol="prediction", metricName="mae"
    )

    rmse = rmse_eval.evaluate(predictions)
    mae = mae_eval.evaluate(predictions)

    return {"mape": round(mape, 2), "rmse": round(rmse, 4), "mae": round(mae, 4)}


def evaluate_trend_model(predictions):
    """Evaluate RandomForestClassifier using weighted F1."""
    f1_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedFMeasure"
    )
    accuracy_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )

    f1 = f1_eval.evaluate(predictions)
    accuracy = accuracy_eval.evaluate(predictions)

    return {"weighted_f1": round(f1, 4), "accuracy": round(accuracy, 4)}


def main():
    """Main ML pipeline."""
    print("=" * 60)
    print("QuickCommerce Analytics — ML Pipeline")
    print(f"Location: {LOCATION}")
    print("=" * 60)

    # Create output directories
    os.makedirs(CURATED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    spark = create_spark_session()

    try:
        # ── Load and prepare data ──
        print("\n[Step 1] Loading unified data...")
        df = load_unified_data(spark)

        # ═══════════════════════════════════════════
        # MODEL 1: DEMAND FORECASTING (GBTRegressor)
        # ═══════════════════════════════════════════
        print("\n[Step 2] Training Demand Forecasting Model (GBTRegressor)...")

        # Time-based split: train on April–May (months 4–5), test on June (month 6)
        train_df = df.filter(F.col("month").isin(4, 5))
        test_df = df.filter(F.col("month") == 6)

        print(f"  Train records (Apr-May): {train_df.count():,}")
        print(f"  Test records (June):     {test_df.count():,}")

        # Build and train pipeline
        demand_pipeline = build_demand_forecast_pipeline()
        print("  Training GBTRegressor...")
        demand_model = demand_pipeline.fit(train_df)

        # Evaluate
        print("  Evaluating on test set...")
        demand_predictions = demand_model.transform(test_df)
        demand_metrics = evaluate_demand_model(demand_predictions)
        print(f"  ✓ MAPE: {demand_metrics['mape']}% (target ≤ 20%)")
        print(f"  ✓ RMSE: {demand_metrics['rmse']}")
        print(f"  ✓ MAE:  {demand_metrics['mae']}")

        # Save model
        demand_model_path = os.path.join(MODELS_DIR, "demand_gbt_model")
        demand_model.write().overwrite().save(demand_model_path)
        print(f"  ✓ Model saved → {demand_model_path}")

        # Generate forecasts for all data and save to curated zone
        print("  Generating full forecasts...")
        all_forecasts = demand_model.transform(df)
        forecast_output = (
            all_forecasts
            .select(
                "record_id", "snapshot_time", "product_name", "category",
                "platform", "pack_size", "pack_unit",
                "units_ordered", "effective_price",
                "hour_of_day", "month", "is_weekend", "weather",
                F.col("prediction").alias("predicted_demand"),
            )
            .withColumn("location", F.lit(LOCATION))
            .withColumn("forecast_horizon_hours", F.lit(24))
            .withColumn("model_version", F.lit("gbt_v1"))
        )
        forecast_output.write.mode("overwrite").parquet(DEMAND_FORECASTS)
        print(f"  ✓ Forecasts saved → {DEMAND_FORECASTS}")

        # ═══════════════════════════════════════════
        # MODEL 2: TREND CLASSIFICATION (RandomForest)
        # ═══════════════════════════════════════════
        print("\n[Step 3] Training Trend Classifier (RandomForestClassifier)...")

        # Compute trend labels from rolling averages
        trend_df = compute_trend_labels(df)

        # Filter nulls in rolling avg (need at least 14 days of history)
        trend_df = trend_df.na.drop(subset=["rolling_avg_7d", "rolling_avg_prev_7d"])

        # Time-based split
        trend_train = trend_df.filter(F.col("month").isin(4, 5))
        trend_test = trend_df.filter(F.col("month") == 6)

        print(f"  Train records (Apr-May): {trend_train.count():,}")
        print(f"  Test records (June):     {trend_test.count():,}")

        # Show label distribution
        print("  Label distribution (train):")
        trend_train.groupBy("trend_label").count().show()

        # Build and train pipeline
        trend_pipeline = build_trend_classifier_pipeline()
        print("  Training RandomForestClassifier...")
        trend_model = trend_pipeline.fit(trend_train)

        # Evaluate
        print("  Evaluating on test set...")
        trend_predictions = trend_model.transform(trend_test)
        trend_metrics = evaluate_trend_model(trend_predictions)
        print(f"  ✓ Weighted F1: {trend_metrics['weighted_f1']} (target ≥ 0.80)")
        print(f"  ✓ Accuracy:    {trend_metrics['accuracy']}")

        # Save model
        trend_model_path = os.path.join(MODELS_DIR, "trend_rf_model")
        trend_model.write().overwrite().save(trend_model_path)
        print(f"  ✓ Model saved → {trend_model_path}")

        # Generate trend labels for all data and save
        print("  Generating full trend labels...")
        all_trends = trend_model.transform(trend_df)

        # Get the label-to-string mapping from the StringIndexer
        label_mapping = trend_model.stages[0]  # StringIndexer for trend_label
        labels = label_mapping.labels

        # Convert prediction index back to string label
        trend_output = (
            all_trends
            .select(
                "product_name", "category", "platform", "date", "month",
                "daily_demand", "daily_avg_price",
                "rolling_avg_7d", "trend_label",
                F.col("prediction").alias("predicted_label_idx"),
            )
            .withColumn("location", F.lit(LOCATION))
        )
        trend_output.write.mode("overwrite").parquet(TREND_LABELS)
        print(f"  ✓ Trend labels saved → {TREND_LABELS}")

        # ── Summary ──
        print("\n" + "=" * 60)
        print("ML Pipeline Complete!")
        print(f"  Demand Model MAPE:    {demand_metrics['mape']}%")
        print(f"  Trend Model F1:       {trend_metrics['weighted_f1']}")
        print(f"  Forecasts saved to:   {DEMAND_FORECASTS}")
        print(f"  Trend labels saved to: {TREND_LABELS}")
        print("=" * 60)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
