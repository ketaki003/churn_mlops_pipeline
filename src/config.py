import os
import boto3
from botocore.client import Config

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:5000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "test")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "test")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "mlops-churn-data")

RAW_DATA_KEY = "raw/telco_churn.csv"
MODEL_KEY = "models/churn_rf_model.joblib"
METRICS_KEY = "metrics/evaluation_report.json"


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
        config=Config(signature_version="s3v4"),
    )