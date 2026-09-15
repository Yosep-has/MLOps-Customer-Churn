import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = "telco_customer_churn_preprocessed.csv"


# Load dataset
df = pd.read_csv(DATA_PATH)

print("Ukuran dataset:", df.shape)


# Pisahkan fitur dan target
X = df.drop(columns=["Churn"])
y = df["Churn"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)


# MLflow autolog
mlflow.sklearn.autolog()


# Training
with mlflow.start_run() as run:

    model.fit(X_train, y_train)

    # Simpan model dalam format joblib
    os.makedirs("artifacts", exist_ok=True)

    joblib.dump(
        model,
        "artifacts/model.pkl"
    )

    # Simpan model dalam format MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    mlflow.sklearn.save_model(
        sk_model=model,
        path="artifacts/mlflow_model"
    )

    # Simpan Run ID
    with open("artifacts/run_id.txt", "w") as f:
        f.write(run.info.run_id)

    print("Model berhasil disimpan ke artifacts/model.pkl")
    print("MLflow model berhasil disimpan.")
    print("Run ID:", run.info.run_id)
    print("Jumlah data training:", len(X_train))
    print("Jumlah data testing :", len(X_test))