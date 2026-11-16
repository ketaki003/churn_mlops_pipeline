# End-to-End MLOps Customer Churn Prediction Pipeline

A containerized, reproducible machine learning pipeline built with **Docker Compose**, **Python**, **Scikit-Learn**, and **S3-compatible object storage (Moto / `boto3`)**. It automates raw dataset ingestion, preprocessing, model training, artifact serialization, and batch inference across isolated containers.

## Architecture
1. **Data Ingestion (`src/ingest.py`)**: Fetches the IBM Telco Customer Churn dataset (7,043 rows, 21 columns), initializes an S3 bucket (`mlops-churn-data`), and uploads raw CSV data (`2.73s`).
2. **Preprocessing & Training (`src/train.py`)**: Retrieves raw data from S3, builds a Scikit-Learn `ColumnTransformer` + class-balanced `RandomForestClassifier` pipeline, evaluates metrics on an 80/20 stratified split, and stores both the `.joblib` model artifact and JSON evaluation report in S3.
3. **Batch Inference (`src/predict.py`)**: Pulls the serialized model from S3 and executes batch churn probability scoring.

## Verified Evaluation Metrics
- **Dataset Size**: 7,043 customer records (5,634 train / 1,409 test)
- **ROC-AUC**: `0.8380`
- **Recall**: `0.6952`
- **F1-Score**: `0.6110`
- **Accuracy**: `0.7651`
- **Training Execution Time**: `1.20 seconds`
- **End-to-End S3 Artifact Load + Batch Inference**: `1.14 seconds`

## Quickstart
```bash
# Run the full MLOps pipeline and automatically tear down containers on completion
docker compose up --build --abort-on-container-exit

# Clean up containers and networks
docker compose down