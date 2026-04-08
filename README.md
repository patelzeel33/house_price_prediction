# 🏠 House Price Prediction System
### End-to-End Machine Learning Project

---

## 📌 Project Overview

This project builds a complete **House Price Prediction** system using Machine Learning and Regression techniques. It predicts the **selling price of a house** based on features like size, location, number of bedrooms, age, and more — using a **Random Forest Regressor** trained on the California Housing dataset.

---

## 🗂️ Project Structure

```
house_price_prediction/
│
├── data/
│   └── housing.csv                    # Dataset (auto-generated or California Housing)
│
├── models/
│   ├── house_model.pkl                # Trained model (generated after training)
│   └── scaler.pkl                     # Feature scaler (generated after training)
│
├── notebooks/
│   └── EDA_and_Modeling.ipynb         # Jupyter Notebook: EDA + Training + Analysis
│
├── src/
│   ├── preprocess.py                  # Data cleaning & feature engineering
│   ├── train.py                       # Model training pipeline
│   ├── predict.py                     # Prediction module
│   └── evaluate.py                    # Model evaluation & visualization
│
├── tests/
│   └── test_predict.py                # Unit tests (18 tests)
│
├── templates/
│   └── index.html                     # Web UI (Flask frontend)
│
├── app.py                             # Flask web application & REST API
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python src/train.py
```

### 3. Run the Web App
```bash
python app.py
```
Open browser: `http://localhost:5000`

### 4. Run Tests
```bash
python tests/test_predict.py
```

### 5. Evaluate Model
```bash
python src/evaluate.py
```

---

## 🧪 Model Performance

| Metric                        | Score       |
|-------------------------------|-------------|
| R² Score                      | ~0.82       |
| RMSE (Root Mean Sq. Error)    | ~$45,000    |
| MAE  (Mean Absolute Error)    | ~$32,000    |
| MAPE (Mean Abs. % Error)      | ~18%        |

---

## 🛠️ Tech Stack

| Category         | Technology                          |
|------------------|-------------------------------------|
| Language         | Python 3.8+                         |
| ML Library       | scikit-learn                        |
| Algorithms       | Random Forest, Linear Regression    |
| Data Handling    | Pandas, NumPy                       |
| Visualization    | Matplotlib, Seaborn                 |
| Web Framework    | Flask                               |
| Serialization    | Joblib                              |
| Testing          | unittest                            |

---

## 📊 Features Used

| Feature            | Description                          |
|--------------------|--------------------------------------|
| MedInc             | Median income of households          |
| HouseAge           | Age of the house in years            |
| AveRooms           | Average number of rooms              |
| AveBedrms          | Average number of bedrooms           |
| Population         | Block population                     |
| AveOccup           | Average household occupancy          |
| Latitude           | Geographic latitude                  |
| Longitude          | Geographic longitude                 |

---

## 🔌 REST API

**Single Prediction:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"MedInc": 8.3, "HouseAge": 15, "AveRooms": 7.0,
       "AveBedrms": 1.1, "Population": 1200, "AveOccup": 3.0,
       "Latitude": 37.88, "Longitude": -122.23}'
```

**Response:**
```json
{
  "predicted_price": 285000,
  "price_range": {"low": 256500, "high": 313500},
  "confidence": "±10%",
  "features_used": 8
}
```

---

## 👤 Author
Built as a complete end-to-end ML pipeline for house price prediction.
