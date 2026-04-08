"""
preprocess.py
-------------
Data cleaning, feature engineering, and preprocessing pipeline
for the House Price Prediction project.
"""

import os
import numpy as np
import pandas as pd


# ─── Feature Definitions ──────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    'MedInc',       # Median income in block group
    'HouseAge',     # Median house age in block group
    'AveRooms',     # Average number of rooms per household
    'AveBedrms',    # Average number of bedrooms per household
    'Population',   # Block group population
    'AveOccup',     # Average household size
    'Latitude',     # Block group latitude
    'Longitude',    # Block group longitude
]

TARGET_COLUMN = 'MedHouseVal'   # Median house value ($100,000s)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the housing dataset from a CSV file.

    Args:
        filepath: Path to the CSV file.

    Returns:
        pandas DataFrame with raw data.
    """
    print(f"[INFO] Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    return df


def generate_sample_data(n_samples: int = 3000, save_path: str = None) -> pd.DataFrame:
    """
    Generate a realistic synthetic housing dataset when the real dataset
    is not available.

    The data is generated so that:
      - Price correlates positively with income and rooms
      - Price correlates negatively with age and occupancy
      - Geographic clustering is applied (coastal vs inland premium)

    Args:
        n_samples:  Number of rows to generate.
        save_path:  If given, saves the CSV here.

    Returns:
        pandas DataFrame.
    """
    print(f"[INFO] Generating synthetic dataset ({n_samples} samples)...")
    rng = np.random.default_rng(42)

    # Base features
    med_inc     = rng.lognormal(mean=1.5, sigma=0.6, size=n_samples).clip(0.5, 15.0)
    house_age   = rng.uniform(1, 52, n_samples)
    ave_rooms   = rng.normal(5.5, 2.0, n_samples).clip(2.0, 20.0)
    ave_bedrms  = (ave_rooms / rng.uniform(3.5, 5.5, n_samples)).clip(0.8, 4.0)
    population  = rng.lognormal(mean=6.5, sigma=0.8, size=n_samples).clip(100, 35000)
    ave_occup   = rng.normal(3.0, 1.0, n_samples).clip(1.0, 8.0)
    latitude    = rng.uniform(32.5, 42.0, n_samples)
    longitude   = rng.uniform(-124.5, -114.0, n_samples)

    # Coastal premium (areas with longitude < -120 get a price boost)
    coastal_bonus = np.where(longitude < -120, rng.uniform(0.3, 0.8, n_samples), 0.0)

    # Price model (in $100,000 units, range ~0.5–5.0)
    price = (
        0.45 * med_inc
        + 0.008 * ave_rooms
        - 0.004 * house_age
        - 0.05  * ave_occup
        + coastal_bonus
        + rng.normal(0, 0.25, n_samples)   # noise
    ).clip(0.5, 5.0)

    df = pd.DataFrame({
        'MedInc':      np.round(med_inc, 4),
        'HouseAge':    np.round(house_age, 1),
        'AveRooms':    np.round(ave_rooms, 4),
        'AveBedrms':   np.round(ave_bedrms, 4),
        'Population':  np.round(population).astype(int),
        'AveOccup':    np.round(ave_occup, 4),
        'Latitude':    np.round(latitude, 4),
        'Longitude':   np.round(longitude, 4),
        'MedHouseVal': np.round(price, 4),
    })

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[INFO] Synthetic data saved to: {save_path}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data cleaning:
      1. Drop rows with missing values.
      2. Remove statistical outliers in the target column (IQR method).
      3. Remove physically impossible values (negative counts, etc.).

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    original_len = len(df)

    # 1. Drop missing values
    df = df.dropna()

    # 2. Remove rows with non-positive population or occupancy
    df = df[df['Population'] > 0]
    df = df[df['AveOccup']   > 0]
    df = df[df['AveRooms']   > 0]
    df = df[df['AveBedrms']  > 0]

    # 3. Remove target outliers via IQR
    if TARGET_COLUMN in df.columns:
        Q1 = df[TARGET_COLUMN].quantile(0.01)
        Q3 = df[TARGET_COLUMN].quantile(0.99)
        df = df[(df[TARGET_COLUMN] >= Q1) & (df[TARGET_COLUMN] <= Q3)]

    print(f"[INFO] Rows after cleaning: {len(df)} (removed {original_len - len(df)})")
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing ones:
      - rooms_per_bedroom: overall room spaciousness
      - income_per_room:   affordability index
      - population_density: crowding proxy

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame with additional engineered columns.
    """
    df = df.copy()

    # Ratio features
    df['rooms_per_bedroom']  = df['AveRooms'] / df['AveBedrms'].replace(0, np.nan)
    df['income_per_room']    = df['MedInc']   / df['AveRooms'].replace(0, np.nan)
    df['population_density'] = df['Population'] / df['AveOccup'].replace(0, np.nan)

    # Fill any NaN created by division
    df.fillna(df.median(numeric_only=True), inplace=True)

    return df


def get_feature_columns(include_engineered: bool = True) -> list:
    """
    Return the list of feature column names used for modelling.

    Args:
        include_engineered: Whether to include engineered features.

    Returns:
        List of column names.
    """
    base = list(FEATURE_COLUMNS)
    if include_engineered:
        base += ['rooms_per_bedroom', 'income_per_room', 'population_density']
    return base


def preprocess_pipeline(df: pd.DataFrame, include_engineered: bool = True):
    """
    Full preprocessing pipeline: clean → engineer → split X / y.

    Args:
        df:                   Raw DataFrame.
        include_engineered:   Whether to add engineered features.

    Returns:
        X (feature DataFrame), y (target Series)
    """
    df = clean_data(df)
    df = engineer_features(df)

    feature_cols = get_feature_columns(include_engineered)
    X = df[feature_cols]
    y = df[TARGET_COLUMN] * 100_000   # Convert to actual dollars

    print(f"[INFO] Feature matrix shape: {X.shape}")
    print(f"[INFO] Target range: ${y.min():,.0f} – ${y.max():,.0f}")
    return X, y


# ─── Quick demo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df  = generate_sample_data(n_samples=500)
    X, y = preprocess_pipeline(df)
    print("\nFirst 3 rows of feature matrix:")
    print(X.head(3))
    print(f"\nTarget sample: {y.head(3).values}")
