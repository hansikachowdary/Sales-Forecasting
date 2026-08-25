from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
from flask import Flask, jsonify, render_template, request
from models.forecasting_model import ForecastingService
from utils.preprocessing import detect_frequency, prepare_sales_data

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


def load_sample():
    return prepare_sales_data(pd.read_csv(BASE_DIR / "data" / "sample_sales.csv"))


current_data = load_sample()
service = ForecastingService(current_data)
service.evaluate()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "sales-forecasting"})


@app.get("/api/data")
def get_data():
    return jsonify({**service.payload(), "frequency": detect_frequency(current_data), "rows": len(current_data)})


@app.get("/api/metrics")
def get_metrics():
    return jsonify({"metrics": service.results, "best_model": service.best_model})


@app.post("/api/upload")
def upload():
    global current_data, service
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename.lower().endswith(".csv"):
        return jsonify({"error": "Please upload a CSV file."}), 400
    try:
        current_data = prepare_sales_data(pd.read_csv(uploaded))
        service = ForecastingService(current_data)
        service.evaluate()
        return jsonify({"message": "Dataset uploaded and evaluated.", "rows": len(current_data), "metrics": service.results, "best_model": service.best_model})
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        return jsonify({"error": "The dataset could not be processed. Check its format and try again."}), 500


@app.post("/api/train")
def train():
    try:
        metrics_result = service.evaluate()
        return jsonify({"metrics": metrics_result, "best_model": service.best_model})
    except Exception:
        return jsonify({"error": "Model training failed for this dataset."}), 500


@app.post("/api/forecast")
def forecast():
    body = request.get_json(silent=True) or {}
    try:
        horizon = int(body.get("horizon", 6))
        if horizon < 1 or horizon > 36:
            raise ValueError
        model = body.get("model") or service.best_model
        return jsonify(service.forecast(horizon, model))
    except (TypeError, ValueError):
        return jsonify({"error": "Forecast horizon must be an integer from 1 to 36."}), 400
    except Exception:
        return jsonify({"error": "Forecast generation failed."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
