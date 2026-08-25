"""Forecast evaluation metrics."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    nonzero = actual != 0
    mape = np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100 if nonzero.any() else 0.0
    return {
        "MAE": round(float(mean_absolute_error(actual, predicted)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(actual, predicted))), 2),
        "MAPE": round(float(mape), 2),
    }
