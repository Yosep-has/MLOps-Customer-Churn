from prometheus_client import Counter, Gauge, start_http_server
import time


# Counter: jumlah request inference
inference_requests = Counter(
    "inference_requests_total",
    "Total jumlah request inference"
)

# Counter: jumlah prediksi churn
churn_predictions = Counter(
    "churn_predictions_total",
    "Total jumlah prediksi churn"
)

# Counter: jumlah prediksi tidak churn
non_churn_predictions = Counter(
    "non_churn_predictions_total",
    "Total jumlah prediksi tidak churn"
)

# Gauge: status model serving
model_serving_status = Gauge(
    "model_serving_status",
    "Status model serving (1 = aktif, 0 = tidak aktif)"
)

# Gauge: waktu inference terakhir
last_inference_time = Gauge(
    "last_inference_time_seconds",
    "Waktu inference terakhir dalam detik"
)

# Gauge: total inference yang berhasil
successful_inferences = Gauge(
    "successful_inferences",
    "Total inference yang berhasil"
)


# Nilai awal
model_serving_status.set(1)
successful_inferences.set(0)


def simulate_inference():
    """
    Simulasi aktivitas inference untuk menghasilkan metrics.
    """
    inference_requests.inc()

    # Simulasi hasil prediksi
    prediction = 0

    if prediction == 1:
        churn_predictions.inc()
    else:
        non_churn_predictions.inc()

    successful_inferences.inc()
    last_inference_time.set(time.time())


if __name__ == "__main__":
    # Exporter berjalan pada port 8000
    start_http_server(8000)

    print("Prometheus exporter berjalan di:")
    print("http://127.0.0.1:8000/metrics")

    while True:
        simulate_inference()
        time.sleep(10)
