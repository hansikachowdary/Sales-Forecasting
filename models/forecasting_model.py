"""Baseline, ARIMA and SARIMA forecasting service."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from utils.evaluation import metrics

warnings.filterwarnings("ignore")


class ForecastingService:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.series = self.data.set_index("Date")["Sales"].astype(float)
        self.split_index = max(int(len(self.series) * 0.8), len(self.series) - 6)
        self.split_index = min(max(self.split_index, 6), len(self.series) - 1)
        self.results = {}
        self.best_model = None

    def _predict(self, name, train, horizon):
        if name == "Naive":
            return np.repeat(train.iloc[-1], horizon), None
        if name == "Moving Average":
            window = min(3, len(train))
            return np.repeat(train.iloc[-window:].mean(), horizon), None
        if name == "ARIMA":
            fitted = ARIMA(train, order=(1, 1, 1)).fit()
            return fitted.forecast(horizon).to_numpy(), fitted
        fitted = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        forecast = fitted.get_forecast(horizon)
        return forecast.predicted_mean.to_numpy(), forecast

    def evaluate(self):
        train = self.series.iloc[:self.split_index]
        test = self.series.iloc[self.split_index:]
        rows = []
        for name in ("Naive", "Moving Average", "ARIMA", "SARIMA"):
            try:
                predicted, _ = self._predict(name, train, len(test))
                score = metrics(test, predicted)
                rows.append({"model": name, **score})
            except Exception as error:
                rows.append({"model": name, "MAE": None, "RMSE": None, "MAPE": None, "error": str(error)})
        valid = [row for row in rows if row["RMSE"] is not None]
        self.results = rows
        self.best_model = min(valid, key=lambda row: row["RMSE"])["model"] if valid else "Naive"
        return rows

    def forecast(self, horizon: int, model_name: str | None = None):
        horizon = max(1, min(int(horizon), 36))
        name = model_name or self.best_model or "Naive"
        try:
            predicted, fitted = self._predict(name, self.series, horizon)
        except Exception:
            name = "Naive"
            predicted, fitted = self._predict(name, self.series, horizon)
        dates = pd.date_range(self.series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
        lower = upper = [None] * horizon
        if fitted is not None and hasattr(fitted, "conf_int"):
            interval = fitted.conf_int(alpha=0.2)
            lower = interval.iloc[:, 0].to_numpy()[-horizon:].tolist()
            upper = interval.iloc[:, 1].to_numpy()[-horizon:].tolist()
        return {"model": name, "forecast": [{"date": date.strftime("%Y-%m-%d"), "predicted": round(float(value), 2), "lower": None if lower[i] is None else round(float(lower[i]), 2), "upper": None if upper[i] is None else round(float(upper[i]), 2)} for i, (date, value) in enumerate(zip(dates, predicted))]}

    def payload(self):
        data = self.data.copy()
        rolling = self.series.rolling(3).mean()
        return {"data": [{"date": row.Date.strftime("%Y-%m-%d"), "sales": round(float(row.Sales), 2), "rolling_mean": None if pd.isna(rolling.loc[row.Date]) else round(float(rolling.loc[row.Date]), 2)} for row in data.itertuples()]}
