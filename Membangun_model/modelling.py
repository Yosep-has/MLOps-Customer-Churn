import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# Path dataset hasil preprocessing
DATA_PATH = (
    "Eksperimen_SML/preprocessing/"
    "namadataset_preprocessing/"
    "telco_customer_churn_preprocessed.csv"
)


# Load dataset
df = pd.read_csv(DATA_PATH)

print("Ukuran dataset:", df.shape)


# Memisahkan fitur dan target
X = df.drop(columns=["Churn"])
y = df["Churn"]


# Membagi dataset menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# Model
model = RandomForestClassifier(
    random_state=42
)


# MLflow Autolog
mlflow.sklearn.autolog()


# Training dan logging dengan MLflow
with mlflow.start_run():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nHasil Evaluasi:")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)