"""
api.app
------
Flask web application for House Price Prediction.
Serves the UI and exposes a REST API for predictions.
"""

import os
import sys
import logging
from flask import Flask, request, jsonify, render_template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Add src/ to path so we can import predict
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
from predict import predict, predict_batch

# Make sure template and static folders point correctly since we're in api/
app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))


@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled Exception: {e}", exc_info=True)
    return jsonify({'error': 'An internal server error occurred.'}), 500


@app.route('/')
def index():
    """Main web UI."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_price():
    """
    POST /predict
    Body (JSON): {"features": {"MedInc": 5.5, ...}} OR direct dictionary
    Response   : prediction dict
    """
    data = request.get_json()
    if not data:
        logger.warning("No JSON body provided.")
        return jsonify({'error': 'No JSON body provided'}), 400

    # Handle {"features": {...}} wrapping if it exists, otherwise assume direct
    features = data.get('features', data)

    if not isinstance(features, dict):
        return jsonify({'error': 'Invalid format. Expected dictionary of features.'}), 400

    required = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                'Population', 'AveOccup', 'Latitude', 'Longitude']
    
    missing = [f for f in required if f not in features]
    if missing:
        logger.warning(f"Missing fields in request: {missing}")
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    # Validate data types and cast to floats
    processed_features = {}
    for key in required:
        try:
            processed_features[key] = float(features[key])
        except (ValueError, TypeError):
            logger.warning(f"Invalid data type for {key}: {features[key]}")
            return jsonify({'error': f'Invalid data type for {key}, must be a number.'}), 400

    try:
        result = predict(processed_features)
        return jsonify(result)
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {e}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
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
    
    records = data['records']
    if not isinstance(records, list):
        return jsonify({'error': '"records" must be a list of dictionaries.'}), 400

    processed_records = []
    required = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                'Population', 'AveOccup', 'Latitude', 'Longitude']

    for i, record in enumerate(records):
        if not isinstance(record, dict):
            return jsonify({'error': f'Record at index {i} must be a dictionary.'}), 400
        
        missing = [f for f in required if f not in record]
        if missing:
            return jsonify({'error': f'Missing fields in record at index {i}: {missing}'}), 400
        
        processed_record = {}
        for key in required:
            try:
                processed_record[key] = float(record[key])
            except (ValueError, TypeError):
                return jsonify({'error': f'Invalid data type in record at index {i} for {key}, must be a number.'}), 400
        processed_records.append(processed_record)

    try:
        results = predict_batch(processed_records)
        return jsonify({'results': results, 'count': len(results)})
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        return jsonify({'error': 'Batch prediction failed.'}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'service': 'House Price Predictor'})


if __name__ == '__main__':
    print("=" * 50)
    print("  House Price Prediction — Web App")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
