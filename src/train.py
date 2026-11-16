import io
import json
import time
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import get_s3_client, BUCKET_NAME, RAW_DATA_KEY, MODEL_KEY, METRICS_KEY


def main():
    start = time.perf_counter()
    s3 = get_s3_client()

    obj = s3.get_object(Bucket=BUCKET_NAME, Key=RAW_DATA_KEY)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    drop_cols = [c for c in ["customerID", "Churn"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = (df["Churn"] == "Yes").astype(int)

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    elapsed = time.perf_counter() - start
    metrics = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "training_time_sec": round(elapsed, 2),
    }

    model_buffer = io.BytesIO()
    joblib.dump(clf, model_buffer)
    model_buffer.seek(0)
    s3.put_object(Bucket=BUCKET_NAME, Key=MODEL_KEY, Body=model_buffer.getvalue())

    s3.put_object(Bucket=BUCKET_NAME, Key=METRICS_KEY, Body=json.dumps(metrics, indent=2))
    print(f"[TRAIN] Metrics saved to s3://{BUCKET_NAME}/{METRICS_KEY}:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()