"""
app.py
------
Flask web application for House Price Prediction.
Serves the UI and exposes a REST API for predictions.
"""

import os
import sys
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from predict import predict, predict_batch

app = Flask(__name__)


@app.route('/')
def index():
    """Main web UI."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_price():
    """
    POST /predict
    Body (JSON): house feature dict
    Response   : prediction dict
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body provided'}), 400

    required = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                'Population', 'AveOccup', 'Latitude', 'Longitude']
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    try:
        result = predict(data)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch_prices():
    """
    POST /predict/batch
    Body (JSON): {"records": [...]}
    """
    data = request.get_json()
    if not data or 'records' not in data:
        return jsonify({'error': 'Provide {"records": [...]}'}), 400
    try:
        results = predict_batch(data['records'])
        return jsonify({'results': results, 'count': len(results)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'service': 'House Price Predictor'})


if __name__ == '__main__':
    print("=" * 50)
    print("  House Price Prediction — Web App")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
