#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Ensure we are in the directory of the script
cd "$(dirname "$0")"
export PYTHONPATH=$PYTHONPATH:$(pwd)/../..

# Configuration
SOURCE_FILE="../superalloy_preprocess/output_data/all_alloys.jsonl"
DATA_FILE="../alloy_crew/models/training_data/train_77alloys.jsonl"
WEAVIATE_COMPOSE="../docker/docker-compose-weaviate.yml"
WEAVIATE_DATA_DIR="../docker/weaviate_data"

echo "==========================================="
echo "   Starting AlloyGraph Data Pipeline"
echo "==========================================="

# 1. Reset Weaviate
echo "[1/5] Resetting Weaviate Environment..."
docker compose -f $WEAVIATE_COMPOSE down
if [ -d "$WEAVIATE_DATA_DIR" ]; then
    echo "      Removing old data volume..."
    rm -rf $WEAVIATE_DATA_DIR
fi
docker compose -f $WEAVIATE_COMPOSE up -d
echo "      Waiting 10s for Weaviate to initialize..."
sleep 10

# 2. Setup Environment
echo "[2/5] Setting up Virtual Environment..."
if [ ! -d "../.venv" ]; then
    echo "      Creating virtual environment..."
    python3.12 -m venv ../.venv
fi

source ../.venv/bin/activate

echo "      Installing dependencies..."
pip install --upgrade pip
pip install -r ../requirements.txt

# 3. Generate Ontology
echo "[3/5] Generating Ontology Schema..."
python build_ontology.py

# 3.5. Feature Enrichment
echo "[3.5/5] Enriching Data with Computed Features..."
python enrich_jsonl_with_features.py "$SOURCE_FILE" "$DATA_FILE"

# 4. Populate GraphDB
echo "[4/5] Populating GraphDB from $DATA_FILE..."
export ALLOY_JSON=$DATA_FILE
python enrich_graphdb.py

# 5. Weaviate Ingestion
echo "[5/5] Setting up Weaviate..."
echo "      Creating Schema..."
python weaviate_schema.py
echo "      Ingesting Data..."
python weaviate_ingest.py

echo "==========================================="
echo "   Pipeline Completed Successfully! 🚀"
echo "==========================================="
