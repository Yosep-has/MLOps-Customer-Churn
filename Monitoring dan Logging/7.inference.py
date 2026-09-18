import json
import requests
import pandas as pd


# URL model serving
URL = "http://127.0.0.1:5002/invocations"

# Dataset hasil preprocessing
DATA_PATH = (
    "Eksperimen_SML/preprocessing/"
    "namadataset_preprocessing/"
    "telco_customer_churn_preprocessed.csv"
)

# Membaca dataset
df = pd.read_csv(DATA_PATH)

# Ambil satu data untuk inference
sample = df.drop(columns=["Churn"]).iloc[[0]]

# Ubah menjadi format dataframe_split
payload = {
    "dataframe_split": {
        "columns": sample.columns.tolist(),
        "data": sample.values.tolist()
    }
}

# Kirim request ke model serving
response = requests.post(
    URL,
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload)
)

print("Status code:", response.status_code)
print("Response:", response.text)
