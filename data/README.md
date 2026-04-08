# Dataset

## California Housing Dataset

This project uses the **California Housing Dataset** — a classic ML benchmark.

### Option 1 — Auto-generated (default)
If no CSV is present, `train.py` auto-generates a realistic synthetic dataset.

### Option 2 — Scikit-learn built-in
```python
from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing(as_frame=True)
df   = data.frame
df.to_csv('data/housing.csv', index=False)
```

### Option 3 — Kaggle
https://www.kaggle.com/datasets/camnugent/california-housing-prices

---

## Column Descriptions

| Column       | Description                                   | Unit          |
|--------------|-----------------------------------------------|---------------|
| MedInc       | Median income in block group                  | ×$10,000      |
| HouseAge     | Median house age in block group               | Years         |
| AveRooms     | Average number of rooms per household         | Count         |
| AveBedrms    | Average number of bedrooms per household      | Count         |
| Population   | Block group population                        | Persons       |
| AveOccup     | Average number of household members           | Persons       |
| Latitude     | Block group latitude                          | Degrees       |
| Longitude    | Block group longitude                         | Degrees       |
| MedHouseVal  | Median house value (TARGET)                   | ×$100,000     |
