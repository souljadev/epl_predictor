#!/bin/bash
# Auto-run EPL agent pipeline: predict -> wait for results -> ingest+retrain
# Usage: place results as data/raw/new_results.csv before running step 2.

echo "Running predictions..."
python -m src.orchestrator.cli --config config.yaml --fixtures data/raw/fixtures_today.csv

echo "Waiting for results... (manual step)"
echo "Place new results in data/raw/new_results.csv and press Enter."
read

last_pred=$(ls -t models/predictions/predictions_*.csv | head -n1)

echo "Ingesting and retraining..."
python -m src.agent.ingest_and_retrain --config config.yaml --new_results data/raw/new_results.csv --predictions $last_pred
