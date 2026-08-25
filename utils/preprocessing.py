"""Input validation and time-series preparation helpers."""
from __future__ import annotations

import pandas as pd


def prepare_sales_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a CSV with a date-like column and a numeric sales column."""
    if frame.empty:
        raise ValueError("The uploaded file is empty.")

    normalized = {str(column).strip().lower(): column for column in frame.columns}
    date_key = next((key for key in normalized if key in {"date", "month", "period", "timestamp"}), None)
    sales_key = next((key for key in normalized if key in {"sales", "sale", "revenue", "value"}), None)
    if date_key is None or sales_key is None:
        raise ValueError("CSV must contain a date column and a sales column.")

    data = frame[[normalized[date_key], normalized[sales_key]]].copy()
    data.columns = ["Date", "Sales"]
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Sales"] = pd.to_numeric(data["Sales"], errors="coerce")
    data = data.dropna(subset=["Date", "Sales"])
    if data.empty:
        raise ValueError("No valid date and sales rows were found.")

    data["Date"] = data["Date"].dt.to_period("M").dt.to_timestamp()
    data = data.groupby("Date", as_index=False)["Sales"].sum().sort_values("Date")
    if len(data) < 12:
        raise ValueError("At least 12 monthly observations are required.")
    data["Sales"] = data["Sales"].clip(lower=0)
    data = data.set_index("Date").asfreq("MS")
    data["Sales"] = data["Sales"].interpolate().ffill().bfill()
    return data.reset_index()


def detect_frequency(data: pd.DataFrame) -> str:
    if len(data) < 3:
        return "Monthly"
    return "Monthly" if data["Date"].diff().dt.days.median() >= 27 else "Daily"
