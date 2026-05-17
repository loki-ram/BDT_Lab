"""
recommendation_engine.py — Preference-Weighted Scoring and Platform Ranking.

Implements the composite scoring logic from PRD Section 6.5 / 11.1:
  composite = w_price × PriceScore + w_delivery × DeliveryScore + w_quality × QualityScore

Used by the Streamlit dashboard and voice handler.
"""

import pandas as pd
import numpy as np


def normalise_weights(price_s, delivery_s, quality_s):
    """
    Normalise slider values to weights summing to 1.0.
    Edge case: all sliders at 0 → equal weights (33/33/34).
    """
    total = price_s + delivery_s + quality_s
    if total == 0:
        return 1 / 3, 1 / 3, 1 / 3

    return price_s / total, delivery_s / total, quality_s / total


def compute_price_score(prices):
    """
    Price Score(p) = 1 - (price(p) - min_price) / (max_price - min_price)
    Lower price → higher score.
    """
    min_p = min(prices)
    max_p = max(prices)
    if max_p == min_p:
        return [1.0] * len(prices)
    return [1.0 - (p - min_p) / (max_p - min_p) for p in prices]


def compute_delivery_score(delivery_times):
    """
    Delivery Score(p) = 1 - (delivery(p) - min_time) / (max_time - min_time)
    Shorter delivery → higher score.
    """
    min_t = min(delivery_times)
    max_t = max(delivery_times)
    if max_t == min_t:
        return [1.0] * len(delivery_times)
    return [1.0 - (t - min_t) / (max_t - min_t) for t in delivery_times]


def compute_quality_score(ratings, availability_rates):
    """
    Quality Score(p) = 0.5 × (rating(p) / 5.0) + 0.5 × availability_rate_7d(p)
    """
    return [
        0.5 * (r / 5.0) + 0.5 * a
        for r, a in zip(ratings, availability_rates)
    ]


def score_platforms(product_data, price_slider=50, delivery_slider=30, quality_slider=20):
    """
    Score and rank platforms for a given product.

    Args:
        product_data: list of dicts with keys:
            platform, effective_price, price_per_unit, delivery_minutes,
            rating, availability_rate_7d, in_stock
        price_slider: int (0–100)
        delivery_slider: int (0–100)
        quality_slider: int (0–100)

    Returns:
        list of dicts sorted by composite_score descending, with all scoring details.
    """
    if not product_data:
        return []

    # Normalise weights
    w_price, w_delivery, w_quality = normalise_weights(
        price_slider, delivery_slider, quality_slider
    )

    # Extract values
    platforms = [d["platform"] for d in product_data]
    prices = [d["effective_price"] for d in product_data]
    deliveries = [d["delivery_minutes"] for d in product_data]
    ratings = [d["rating"] for d in product_data]
    availability = [d.get("availability_rate_7d", 0.85) for d in product_data]
    in_stock_flags = [d.get("in_stock", True) for d in product_data]

    # Compute sub-scores
    price_scores = compute_price_score(prices)
    delivery_scores = compute_delivery_score(deliveries)
    quality_scores = compute_quality_score(ratings, availability)

    # Apply stock penalty: if product unavailable, quality score = 0
    quality_scores = [
        0.0 if not in_stock else qs
        for qs, in_stock in zip(quality_scores, in_stock_flags)
    ]

    # Compute composite scores
    results = []
    for i, platform in enumerate(platforms):
        composite = (
            w_price * price_scores[i]
            + w_delivery * delivery_scores[i]
            + w_quality * quality_scores[i]
        )

        results.append({
            "platform": platform,
            "effective_price": prices[i],
            "price_per_unit": product_data[i].get("price_per_unit", prices[i]),
            "delivery_minutes": deliveries[i],
            "rating": ratings[i],
            "availability_rate_7d": availability[i],
            "in_stock": in_stock_flags[i],
            "price_score": round(price_scores[i], 4),
            "delivery_score": round(delivery_scores[i], 4),
            "quality_score": round(quality_scores[i], 4),
            "composite_score": round(composite, 4),
        })

    # Sort by composite score descending
    results.sort(key=lambda x: x["composite_score"], reverse=True)

    return results


def get_recommendation(scored_platforms):
    """
    Extract recommendation summary from scored platforms.
    Returns dict with recommended, best_price, fastest, highest_quality platforms.
    """
    if not scored_platforms:
        return {}

    recommended = scored_platforms[0]["platform"]

    best_price = min(scored_platforms, key=lambda x: x["effective_price"])["platform"]
    fastest = min(scored_platforms, key=lambda x: x["delivery_minutes"])["platform"]
    highest_quality = max(scored_platforms, key=lambda x: x["quality_score"])["platform"]

    return {
        "recommended_platform": recommended,
        "best_price_platform": best_price,
        "fastest_delivery_platform": fastest,
        "highest_quality_platform": highest_quality,
        "score_breakdown": scored_platforms,
    }


def generate_response_text(product_name, recommendation, pack_size=None):
    """
    Generate a human-readable recommendation response (for voice TTS).
    """
    if not recommendation:
        return f"Sorry, I couldn't find data for {product_name}."

    rec = recommendation
    winner = rec["recommended_platform"]
    scores = rec["score_breakdown"]

    # Find the winner's details
    winner_data = next(s for s in scores if s["platform"] == winner)

    pack_str = f" {pack_size}" if pack_size else ""

    platform_names = {
        "blinkit": "Blinkit",
        "zepto": "Zepto",
        "swiggy": "Swiggy Instamart",
    }

    response = (
        f"For {product_name}{pack_str} in Jayanagar, "
        f"{platform_names.get(winner, winner)} is the best option — "
        f"₹{winner_data['effective_price']:.0f}, "
        f"{winner_data['delivery_minutes']}-minute delivery, "
        f"rated {winner_data['rating']:.1f}/5."
    )

    # Add price comparison if different platform is cheapest
    if rec["best_price_platform"] != winner:
        cheapest = next(
            s for s in scores if s["platform"] == rec["best_price_platform"]
        )
        response += (
            f" However, {platform_names.get(rec['best_price_platform'], rec['best_price_platform'])} "
            f"has the lowest price at ₹{cheapest['effective_price']:.0f}."
        )

    return response
