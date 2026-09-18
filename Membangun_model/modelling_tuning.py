import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. Checklist K2 Skilled
# ============================================================
CHECKLIST = {
    "Dataset preprocessing digunakan": True,
    "Model Scikit-Learn digunakan": True,
    "Hyperparameter tuning dilakukan": True,
    "Manual logging parameter MLflow": True,
    "Manual logging metric MLflow": True,
    "Accuracy dicatat": True,
    "Precision dicatat": True,
    "Recall dicatat": True,
    "F1 Score dicatat": True
}


# ============================================================
# 2. Load dataset
# ============================================================
DATA_PATH = (
    "telco_customer_churn_preprocessed.csv"
    if os.path.exists("telco_customer_churn_preprocessed.csv")
    else "Eksperimen_SML/preprocessing/namadataset_preprocessing/telco_customer_churn_preprocessed.csv"
)

df = pd.read_csv(DATA_PATH)

print("Ukuran dataset:", df.shape)


# ============================================================
# 3. Pisahkan fitur dan target
# ============================================================
X = df.drop(columns=["Churn"])
y = df["Churn"]


# ============================================================
# 4. Train-test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# ============================================================
# 5. Hyperparameter tuning
# ============================================================
model = RandomForestClassifier(
    random_state=42
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="f1",
    n_jobs=1
)


# ============================================================
# 6. MLflow manual logging
# ============================================================
with mlflow.start_run(run_name="RandomForest_Hyperparameter_Tuning"):

    # Melakukan tuning
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Prediksi
    y_pred = best_model.predict(X_test)

    # Evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # --------------------------------------------------------
    # Manual logging parameter
    # --------------------------------------------------------
    mlflow.log_param(
        "model",
        "RandomForestClassifier"
    )

    mlflow.log_param(
        "tuning_method",
        "GridSearchCV"
    )

    mlflow.log_param(
        "cv",
        3
    )

    mlflow.log_params(
        grid_search.best_params_
    )

    # --------------------------------------------------------
    # Manual logging metrics
    # --------------------------------------------------------
    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    # --------------------------------------------------------
    # Log model
    # --------------------------------------------------------
    mlflow.sklearn.log_model(
        best_model,
        "model"
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nHasil Evaluasi:")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)


# ============================================================
# 7. Tampilkan checklist
# ============================================================
print("\nChecklist K2 Skilled:")

for item, status in CHECKLIST.items():
    print(
        f"[{'x' if status else ' '}] {item}"
    )