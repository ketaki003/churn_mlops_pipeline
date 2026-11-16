import io
import time
import joblib
import pandas as pd
from config import get_s3_client, BUCKET_NAME, RAW_DATA_KEY, MODEL_KEY


def main():
    start = time.perf_counter()
    s3 = get_s3_client()

    model_obj = s3.get_object(Bucket=BUCKET_NAME, Key=MODEL_KEY)
    clf = joblib.load(io.BytesIO(model_obj["Body"].read()))

    data_obj = s3.get_object(Bucket=BUCKET_NAME, Key=RAW_DATA_KEY)
    df = pd.read_csv(io.BytesIO(data_obj["Body"].read())).head(10)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    X_sample = df.drop(columns=[c for c in ["customerID", "Churn"] if c in df.columns])
    probs = clf.predict_proba(X_sample)[:, 1]
    preds = clf.predict(X_sample)

    elapsed_ms = (time.perf_counter() - start) * 1000
    results = pd.DataFrame({
        "customerID": df.get("customerID", range(len(df))),
        "churn_probability": probs.round(4),
        "predicted_churn": ["Yes" if p == 1 else "No" for p in preds],
    })
    print(f"[PREDICT] Batch inference on {len(results)} records completed in {elapsed_ms:.2f} ms:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()