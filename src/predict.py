"""
predict.py
----------
Prediction module for House Price Prediction.
Loads the trained model + scaler and predicts house prices.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv

load_dotenv()

# ── Add src/ to path ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import engineer_features, get_feature_columns

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR   = os.environ.get('MODEL_DIR', os.path.join(BASE_DIR, 'models'))
MODEL_PATH  = os.path.join(MODEL_DIR, 'house_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
META_PATH   = os.path.join(MODEL_DIR, 'model_meta.pkl')


# ── Cached model objects ──────────────────────────────────────────────────────
_model  = None
_scaler = None
_meta   = None


def _load():
    """Load model + scaler into module-level cache (once)."""
    global _model, _scaler, _meta
    if _model is None:
        for path in (MODEL_PATH, SCALER_PATH):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model artifact not found: {path}\n"
                    "Run  python src/train.py  first."
                )
        _model  = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        _meta   = joblib.load(META_PATH) if os.path.exists(META_PATH) else {}


def predict(features: dict) -> dict:
    """
    Predict the price of a single house.

    Args:
        features: dict with keys matching FEATURE_COLUMNS:
            MedInc, HouseAge, AveRooms, AveBedrms,
            Population, AveOccup, Latitude, Longitude

    Returns:
        dict with:
            predicted_price  – point estimate in USD
            price_range      – {low, high} at ±10 %
            confidence       – string description
            features_used    – number of features
            model_name       – which model was selected at training
    """
    _load()

    # Build a single-row DataFrame
    row = pd.DataFrame([features])

    # Engineer the same derived features used at training
    row = engineer_features(row)

    # Select & order columns exactly as in training
    feature_cols = get_feature_columns(include_engineered=True)
    # Only keep columns that exist (graceful degradation)
    available_cols = [c for c in feature_cols if c in row.columns]
    X = row[available_cols]

    # Scale
    X_scaled = _scaler.transform(X)

    # Predict (model output is already in dollars — see train.py)
    raw_pred = float(_model.predict(X_scaled)[0])

    # Clamp to a sensible range ($50k – $5M)
    price = max(50_000, min(5_000_000, raw_pred))

    return {
        "predicted_price": round(price, -2),          # nearest $100
        "price_range": {
            "low":  round(price * 0.90, -2),
            "high": round(price * 1.10, -2),
        },
        "confidence":   "±10%",
        "features_used": len(available_cols),
        "model_name":   _meta.get("model_name", "Random Forest"),
    }


def predict_batch(records: list) -> list:
    """
    Predict prices for multiple houses.

    Args:
        records: list of feature dicts (same schema as predict())

    Returns:
        list of prediction dicts
    """
    return [predict(r) for r in records]


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_houses = [
        {
            "label":      "Luxury coastal home",
            "MedInc":     10.5,
            "HouseAge":   8.0,
            "AveRooms":   9.0,
            "AveBedrms":  1.5,
            "Population": 800,
            "AveOccup":   2.5,
            "Latitude":   37.85,
            "Longitude":  -122.25,
        },
        {
            "label":      "Mid-range suburban home",
            "MedInc":     5.5,
            "HouseAge":   20.0,
            "AveRooms":   6.0,
            "AveBedrms":  1.2,
            "Population": 1500,
            "AveOccup":   3.2,
            "Latitude":   34.05,
            "Longitude":  -118.25,
        },
        {
            "label":      "Affordable inland home",
            "MedInc":     2.8,
            "HouseAge":   35.0,
            "AveRooms":   4.5,
            "AveBedrms":  1.1,
            "Population": 3500,
            "AveOccup":   4.0,
            "Latitude":   36.78,
            "Longitude":  -119.42,
        },
    ]

    print("=" * 60)
    print("  HOUSE PRICE — PREDICTION DEMO")
    print("=" * 60)

    try:
        for house in sample_houses:
            label = house.pop("label")
            result = predict(house)
            print(f"\n{label}")
            print(f"  Predicted price : ${result['predicted_price']:>10,.0f}")
            print(f"  Range           : ${result['price_range']['low']:>10,.0f}  –  ${result['price_range']['high']:>10,.0f}")
            print(f"  Confidence      : {result['confidence']}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
