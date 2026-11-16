import io
import time
import pandas as pd
import numpy as np
from botocore.exceptions import ClientError
from config import get_s3_client, BUCKET_NAME, RAW_DATA_KEY

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def ensure_bucket_exists(s3_client, bucket_name):
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except ClientError:
        s3_client.create_bucket(Bucket=bucket_name)
        print(f"[INGEST] Created S3 bucket: {bucket_name}")


def load_or_generate_data():
    try:
        df = pd.read_csv(DATA_URL)
        print(f"[INGEST] Downloaded IBM Telco Churn dataset: {df.shape[0]} rows, {df.shape[1]} cols")
        return df
    except Exception as exc:
        print(f"[INGEST] Remote download failed ({exc}); generating deterministic dataset...")
        rng = np.random.default_rng(42)
        n = 5000
        df = pd.DataFrame({
            "customerID": [f"CUST-{i:05d}" for i in range(n)],
            "tenure": rng.integers(1, 72, size=n),
            "MonthlyCharges": np.round(rng.uniform(20.0, 118.0, size=n), 2),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20]),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], size=n),
            "TechSupport": rng.choice(["Yes", "No"], size=n),
        })
        df["TotalCharges"] = np.round(df["tenure"] * df["MonthlyCharges"], 2)
        logit = -1.5 + 1.2 * (df["Contract"] == "Month-to-month") - 0.03 * df["tenure"] + 0.015 * df["MonthlyCharges"]
        prob = 1 / (1 + np.exp(-logit))
        df["Churn"] = np.where(rng.uniform(0, 1, size=n) < prob, "Yes", "No")
        return df


def main():
    start = time.perf_counter()
    s3 = get_s3_client()
    ensure_bucket_exists(s3, BUCKET_NAME)

    df = load_or_generate_data()
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    s3.put_object(Bucket=BUCKET_NAME, Key=RAW_DATA_KEY, Body=csv_buffer.getvalue())
    elapsed = time.perf_counter() - start
    print(f"[INGEST] Uploaded {RAW_DATA_KEY} to s3://{BUCKET_NAME}/{RAW_DATA_KEY} in {elapsed:.2f}s")


if __name__ == "__main__":
    main()