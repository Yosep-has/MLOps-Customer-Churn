import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def load_data(input_path):
    """Memuat dataset mentah."""
    return pd.read_csv(input_path)


def preprocess_data(df):
    """Melakukan preprocessing dataset Telco Customer Churn."""

    # Mengubah TotalCharges menjadi numerik
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Menangani nilai kosong pada TotalCharges
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Menghapus customerID
    df = df.drop(columns=["customerID"])

    # Encoding target Churn
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    return df

    
def encode_features(df):
    """Melakukan One-Hot Encoding pada fitur kategorikal."""

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_columns = df.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    numerical_features = df[numerical_columns].drop(
        columns=["Churn"]
    )

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    encoded_data = encoder.fit_transform(
        df[categorical_columns]
    )

    encoded_columns = encoder.get_feature_names_out(
        categorical_columns
    )

    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoded_columns,
        index=df.index
    )

    X = pd.concat(
        [numerical_features, encoded_df],
        axis=1
    )

    y = df["Churn"]

    return X, y

def save_preprocessed_data(X, y, output_path):
    """Menyimpan dataset hasil preprocessing ke CSV."""

    df_preprocessed = pd.concat([X, y], axis=1)

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df_preprocessed.to_csv(output_path, index=False)

    print("Dataset hasil preprocessing berhasil disimpan.")
    print("Lokasi:", output_path)
    print("Ukuran dataset:", df_preprocessed.shape)

if __name__ == "__main__":

    input_path = (
    "Eksperimen_SML/"
    "dataset_raw/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    output_path = (
    "Eksperimen_SML/"
    "preprocessing/"
    "namadataset_preprocessing/"
    "telco_customer_churn_preprocessed.csv"
    )

    df = load_data(input_path)

    df = preprocess_data(df)

    X, y = encode_features(df)

    save_preprocessed_data(
        X,
        y,
        output_path
    )