# Sales Forecasting Using Time Series Analysis

A complete Flask web application for exploring monthly sales history, comparing forecasting models, and generating future projections. It is designed as an academic internship major project: every stage of the workflow is visible and explainable.

## Problem statement
Create a time-series forecasting model to predict future values based on historical data.

## Objectives
- Prepare and validate monthly sales data.
- Understand trend, seasonality, stationarity and autocorrelation.
- Compare Naive, Moving Average, ARIMA and SARIMA models.
- Evaluate models with MAE, RMSE and MAPE.
- Select the best model and forecast up to 36 future months.

## Features
- Included 3-year monthly sample dataset.
- CSV upload with date/sales column detection, missing-value handling, duplicate aggregation, sorting and monthly gap interpolation.
- Interactive historical and forecast charts using Chart.js.
- Chronological 80/20 train/test split; no random shuffling.
- Model comparison table and automatic best-model selection.
- Forecast horizon controls for 3, 6 and 12 months, plus custom API horizons from 1 to 36.
- Friendly API errors and responsive dashboard layout.

## Technology
Python, Flask, Pandas, NumPy, Scikit-learn, Statsmodels, Joblib-compatible Python environment, Chart.js and custom CSS.

## Project structure
```text
sales forcasting/
├── app.py
├── requirements.txt
├── render.yaml
├── data/sample_sales.csv
├── models/forecasting_model.py
├── utils/preprocessing.py
├── utils/evaluation.py
├── templates/index.html
└── static/{css/style.css,js/app.js}
```

## Installation and local run
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```
Open `http://127.0.0.1:5000`.

## Dataset format
Your CSV needs one date-like column (`Date`, `Month`, `Period`, or `Timestamp`) and one numeric sales-like column (`Sales`, `Sale`, `Revenue`, or `Value`). At least 12 valid monthly observations are required. The sample file contains 36 months.

## Forecasting methodology
1. Dates are parsed and converted to month starts.
2. Duplicate months are summed, rows are sorted, and missing months are interpolated.
3. The series is split chronologically: the first 80% trains the models and the final 20% tests them.
4. Naive uses the final training value; Moving Average uses the latest three-month mean.
5. ARIMA uses order `(1, 1, 1)` and SARIMA uses `(1, 1, 1)` with a `(1, 1, 1, 12)` yearly seasonal order.
6. MAE, RMSE and MAPE are calculated on held-out test data. Lowest RMSE wins.
7. The selected model is refit on all available history for the future forecast.

## Metrics
- **MAE:** average absolute prediction error.
- **RMSE:** square-root average squared error; larger errors receive more weight.
- **MAPE:** average percentage error, excluding zero actual values.

## API
- `GET /health` returns service status.
- `GET /api/data` returns normalized observations and rolling means.
- `GET /api/metrics` returns model metrics and the best model.
- `POST /api/upload` accepts multipart form field `file` containing a CSV.
- `POST /api/train` reevaluates all models.
- `POST /api/forecast` accepts JSON such as `{ "horizon": 6, "model": "SARIMA" }`.

## Testing
```bash
python -m compileall -q .
python -c "from app import app; client=app.test_client(); print(client.get('/health').json); print(client.get('/api/metrics').json['best_model']); print(len(client.post('/api/forecast', json={'horizon': 3}).json['forecast']))"
```

## Render deployment
The included `render.yaml` uses:
```text
Build: pip install -r requirements.txt
Start: gunicorn app:app
```
Create a Render Web Service from the repository. Render will read the blueprint and expose the Flask app without any hard-coded frontend hostnames.

## Screenshots
Add screenshots of the Overview, Analysis, Forecast and Performance sections here for submission.

## Future scope
Add exogenous variables such as promotions and holidays, automatic hyperparameter search, user accounts, persistent datasets, forecast export, and an automated stationarity test panel.

## Conclusion
This project demonstrates a reproducible end-to-end time-series forecasting workflow while keeping the assumptions, validation method and model comparison visible to the user.
