"""
evaluate.py
-----------
Model evaluation and visualization for the House Price Prediction project.
Generates:
  - Actual vs Predicted scatter plot
  - Residuals distribution
  - Feature importance bar chart
  - Error metrics summary
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_data, generate_sample_data, preprocess_pipeline

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'house_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
META_PATH   = os.path.join(BASE_DIR, 'models', 'model_meta.pkl')
DATA_PATH   = os.path.join(BASE_DIR, 'data',   'housing.csv')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'models')


def evaluate():
    # ── Load artifacts ────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run python src/train.py first.")

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    meta   = joblib.load(META_PATH) if os.path.exists(META_PATH) else {}

    # ── Load & prepare data ───────────────────────────────────────────────
    if os.path.exists(DATA_PATH):
        raw = load_data(DATA_PATH)
    else:
        raw = generate_sample_data(n_samples=3000)

    X, y = preprocess_pipeline(raw)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_s = scaler.transform(X_test)
    y_pred   = model.predict(X_test_s)

    # ── Metrics ───────────────────────────────────────────────────────────
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    mask = y_test.values > 1000
    mape = np.mean(np.abs((y_test.values[mask] - y_pred[mask]) / y_test.values[mask])) * 100

    print("=" * 60)
    print("  MODEL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Model    : {meta.get('model_name', 'Unknown')}")
    print(f"  R²       : {r2:.4f}")
    print(f"  RMSE     : ${rmse:,.0f}")
    print(f"  MAE      : ${mae:,.0f}")
    print(f"  MAPE     : {mape:.1f}%")

    # ── Plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('House Price Prediction — Model Evaluation', fontsize=15, fontweight='bold')

    # 1. Actual vs Predicted
    axes[0].scatter(y_test / 1000, y_pred / 1000, alpha=0.3, s=10, color='steelblue')
    mn = min(y_test.min(), y_pred.min()) / 1000
    mx = max(y_test.max(), y_pred.max()) / 1000
    axes[0].plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect prediction')
    axes[0].set_xlabel('Actual Price ($K)')
    axes[0].set_ylabel('Predicted Price ($K)')
    axes[0].set_title(f'Actual vs Predicted  (R²={r2:.3f})')
    axes[0].legend()

    # 2. Residuals
    residuals = (y_pred - y_test.values) / 1000
    axes[1].hist(residuals, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[1].axvline(0, color='red', linestyle='--', lw=1.5)
    axes[1].set_xlabel('Residual ($K)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Residuals Distribution')

    # 3. Feature importance (if available)
    if hasattr(model, 'feature_importances_') and meta.get('feature_names'):
        fi = pd.Series(model.feature_importances_, index=meta['feature_names'])
        fi_sorted = fi.sort_values(ascending=True).tail(10)
        fi_sorted.plot(kind='barh', ax=axes[2], color='steelblue')
        axes[2].set_title('Feature Importances (top 10)')
        axes[2].set_xlabel('Importance')
    else:
        axes[2].text(0.5, 0.5, 'Feature importance\nnot available',
                     ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Feature Importances')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'evaluation_plots.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n[✓] Plots saved → {out_path}")
    plt.close()

    return {"r2": r2, "rmse": rmse, "mae": mae, "mape": mape}


if __name__ == "__main__":
    evaluate()
