#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Upgrading pip, setuptools and wheel..."
pip install --upgrade pip setuptools wheel

echo "Installing requirements..."
pip install -r requirements.txt

echo "Training model if necessary..."
# Ensure that the model is generated during the build step.
python src/train.py

echo "Build complete."
