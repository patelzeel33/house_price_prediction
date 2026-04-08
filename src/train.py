"""
train.py
--------
Model training pipeline for House Price Prediction.

Trains three models and picks the best one:
  1. Linear Regression    (baseline)
  2. Gradient Boosting    (strong ensemble)
  3. Random Forest        (robust ensemble) ← default best

Saves the winner + scaler to /models/.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv

load_dotenv()

from sklearn.model_selection  import train_test_split, cross_val_score, KFold
from sklearn.preprocessing    import StandardScaler
from sklearn.linear_model     import LinearRegression, Ridge
from sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics          import (
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.pipeline         import Pipeline

# ── Add src/ to path ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import (
    load_data, generate_sample_data, preprocess_pipeline, get_feature_columns
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, 'data',   'housing.csv')
MODEL_DIR   = os.environ.get('MODEL_DIR', os.path.join(BASE_DIR, 'models'))
MODEL_PATH  = os.path.join(MODEL_DIR, 'house_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
META_PATH   = os.path.join(MODEL_DIR, 'model_meta.pkl')


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (ignores near-zero values)."""
    mask = np.abs(y_true) > 1000
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(model, X_test, y_test, name="Model"):
    """Print regression metrics for one model."""
    y_pred = model.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    mape_v = mape(y_test.values, y_pred)

    print(f"\n  {name}")
    print(f"    R²   : {r2:.4f}")
    print(f"    RMSE : ${rmse:,.0f}")
    print(f"    MAE  : ${mae:,.0f}")
    print(f"    MAPE : {mape_v:.1f}%")
    return {"r2": r2, "rmse": rmse, "mae": mae, "mape": mape_v}


def train():
    """Full training pipeline."""

    print("=" * 60)
    print("  HOUSE PRICE PREDICTION — MODEL TRAINING")
    print("=" * 60)

    # ── 1. Load / generate data ────────────────────────────────────────────
    if os.path.exists(DATA_PATH):
        raw = load_data(DATA_PATH)
    else:
        raw = generate_sample_data(
            n_samples=5000,
            save_path=DATA_PATH,
        )

    # ── 2. Preprocess ──────────────────────────────────────────────────────
    X, y = preprocess_pipeline(raw)
    feature_names = list(X.columns)

    # ── 3. Train / test split ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n[INFO] Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── 4. Scale features ─────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── 5. Define candidate models ────────────────────────────────────────
    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=1.0),
        "Random Forest":     RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
        ),
    }

    # ── 6. Train & evaluate all candidates ───────────────────────────────
    print("\n[INFO] Training & evaluating all models …")
    print("-" * 60)
    results = {}
    trained = {}

    for name, model in candidates.items():
        model.fit(X_train_s, y_train)
        metrics = evaluate_model(model, X_test_s, y_test, name)
        results[name] = metrics
        trained[name] = model

    # ── 7. Pick best model by R² ──────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["r2"])
    best_model  = trained[best_name]
    best_metrics = results[best_name]

    print(f"\n[✓] Best model: {best_name}  (R² = {best_metrics['r2']:.4f})")

    # ── 8. Cross-validation ───────────────────────────────────────────────
    print("\n[INFO] Running 5-fold cross-validation on best model …")
    kf  = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(best_model, X_train_s, y_train, cv=kf, scoring="r2")
    print(f"  CV R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

    # ── 9. Feature importance (if available) ─────────────────────────────
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        fi_df = pd.DataFrame({
            "feature":    feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        print("\n[INFO] Top 5 feature importances:")
        print(fi_df.head(5).to_string(index=False))

    # ── 10. Save model, scaler & metadata ────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler,     SCALER_PATH)
    joblib.dump({
        "model_name":    best_name,
        "feature_names": feature_names,
        "metrics":       best_metrics,
        "cv_r2_mean":    float(cv_r2.mean()),
        "cv_r2_std":     float(cv_r2.std()),
    }, META_PATH)

    print(f"\n[✓] Model  saved → {MODEL_PATH}")
    print(f"[✓] Scaler saved → {SCALER_PATH}")
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)

    return best_model, scaler, best_metrics


if __name__ == "__main__":
    train()
