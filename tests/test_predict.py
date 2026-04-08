"""
test_predict.py
---------------
Unit tests for the House Price Prediction system.
Run with: python tests/test_predict.py
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from preprocess import (
    generate_sample_data, clean_data, engineer_features,
    preprocess_pipeline, get_feature_columns, TARGET_COLUMN
)


class TestDataGeneration(unittest.TestCase):

    def test_generates_correct_shape(self):
        df = generate_sample_data(n_samples=100)
        self.assertEqual(len(df), 100)

    def test_has_all_required_columns(self):
        df = generate_sample_data(n_samples=50)
        required = ['MedInc','HouseAge','AveRooms','AveBedrms',
                    'Population','AveOccup','Latitude','Longitude','MedHouseVal']
        for col in required:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_no_missing_values(self):
        df = generate_sample_data(n_samples=200)
        self.assertEqual(df.isnull().sum().sum(), 0)

    def test_price_in_reasonable_range(self):
        df = generate_sample_data(n_samples=500)
        self.assertTrue((df['MedHouseVal'] >= 0.5).all())
        self.assertTrue((df['MedHouseVal'] <= 5.0).all())

    def test_income_positive(self):
        df = generate_sample_data(n_samples=100)
        self.assertTrue((df['MedInc'] > 0).all())

    def test_population_positive(self):
        df = generate_sample_data(n_samples=100)
        self.assertTrue((df['Population'] > 0).all())


class TestDataCleaning(unittest.TestCase):

    def setUp(self):
        self.df = generate_sample_data(n_samples=300)

    def test_clean_removes_no_positives(self):
        cleaned = clean_data(self.df)
        self.assertGreater(len(cleaned), 0)

    def test_clean_returns_dataframe(self):
        cleaned = clean_data(self.df)
        self.assertIsInstance(cleaned, pd.DataFrame)

    def test_clean_removes_negative_population(self):
        df_bad = self.df.copy()
        df_bad.loc[0, 'Population'] = -1
        cleaned = clean_data(df_bad)
        self.assertTrue((cleaned['Population'] > 0).all())


class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        self.df = generate_sample_data(n_samples=200)

    def test_adds_engineered_columns(self):
        engineered = engineer_features(self.df)
        for col in ['rooms_per_bedroom', 'income_per_room', 'population_density']:
            self.assertIn(col, engineered.columns)

    def test_no_nan_after_engineering(self):
        engineered = engineer_features(self.df)
        self.assertEqual(engineered.isnull().sum().sum(), 0)

    def test_does_not_modify_original(self):
        original_cols = list(self.df.columns)
        _ = engineer_features(self.df)
        self.assertEqual(list(self.df.columns), original_cols)


class TestPreprocessPipeline(unittest.TestCase):

    def setUp(self):
        self.df = generate_sample_data(n_samples=400)

    def test_returns_X_and_y(self):
        X, y = preprocess_pipeline(self.df)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

    def test_X_and_y_same_length(self):
        X, y = preprocess_pipeline(self.df)
        self.assertEqual(len(X), len(y))

    def test_y_in_dollar_range(self):
        _, y = preprocess_pipeline(self.df)
        self.assertTrue((y > 10_000).all())
        self.assertTrue((y < 10_000_000).all())

    def test_feature_columns_present(self):
        X, _ = preprocess_pipeline(self.df)
        expected = get_feature_columns(include_engineered=True)
        for col in expected:
            self.assertIn(col, X.columns)


class TestPrediction(unittest.TestCase):
    """Requires trained model — skipped if not trained."""

    MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'house_model.pkl')

    SAMPLE_HOUSE = {
        'MedInc':     5.5,
        'HouseAge':   20.0,
        'AveRooms':   6.0,
        'AveBedrms':  1.1,
        'Population': 1500,
        'AveOccup':   3.0,
        'Latitude':   37.88,
        'Longitude': -122.23,
    }

    def setUp(self):
        if not os.path.exists(self.MODEL_PATH):
            self.skipTest("Model not trained. Run python src/train.py first.")

    def test_predict_returns_dict(self):
        from predict import predict
        result = predict(self.SAMPLE_HOUSE)
        self.assertIsInstance(result, dict)

    def test_predict_has_required_keys(self):
        from predict import predict
        result = predict(self.SAMPLE_HOUSE)
        for key in ['predicted_price', 'price_range', 'confidence', 'features_used']:
            self.assertIn(key, result)

    def test_price_is_positive(self):
        from predict import predict
        result = predict(self.SAMPLE_HOUSE)
        self.assertGreater(result['predicted_price'], 0)

    def test_price_in_realistic_range(self):
        from predict import predict
        result = predict(self.SAMPLE_HOUSE)
        self.assertGreater(result['predicted_price'], 50_000)
        self.assertLess(result['predicted_price'],    5_000_000)

    def test_range_low_less_than_high(self):
        from predict import predict
        result = predict(self.SAMPLE_HOUSE)
        self.assertLess(result['price_range']['low'], result['price_range']['high'])

    def test_luxury_home_more_expensive(self):
        from predict import predict
        luxury = dict(self.SAMPLE_HOUSE)
        luxury['MedInc']   = 12.0
        luxury['AveRooms'] = 10.0
        luxury['HouseAge'] = 5.0
        r_luxury   = predict(luxury)
        r_standard = predict(self.SAMPLE_HOUSE)
        self.assertGreater(r_luxury['predicted_price'], r_standard['predicted_price'])

    def test_batch_prediction(self):
        from predict import predict_batch
        results = predict_batch([self.SAMPLE_HOUSE, self.SAMPLE_HOUSE])
        self.assertEqual(len(results), 2)

    def test_features_used_count(self):
        from predict import predict
        result = predict(self.SAMPLE_HOUSE)
        self.assertGreaterEqual(result['features_used'], 8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
