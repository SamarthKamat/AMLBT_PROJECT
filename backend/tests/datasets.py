"""Synthetic datasets used by the tests. None of these match the seven bundled domain samples,
so they prove the engine works on schemas it was not built around."""
from __future__ import annotations
import numpy as np
import pandas as pd


def delivery_df(n: int = 700, seed: int = 7, anomalies: bool = True) -> pd.DataFrame:
    """Delivery logistics: Delivery_Time depends on Distance, Traffic, Weather and Vehicle_Type."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    distance = rng.uniform(1, 30, n).round(2)
    traffic = rng.choice(["Low", "Medium", "High"], n, p=[.3, .4, .3])
    weather = rng.choice(["Clear", "Rain", "Fog"], n, p=[.6, .3, .1])
    vehicle = rng.choice(["Bike", "Car", "Van"], n)
    city = rng.choice(["Goa", "Pune", "Mumbai", "Nagpur"], n)
    t = (8 + 2.2 * distance + np.select([traffic == "High", traffic == "Medium"], [15, 6], 0)
         + np.select([weather == "Rain", weather == "Fog"], [8, 12], 0)
         + np.select([vehicle == "Bike", vehicle == "Van"], [-3, 4], 0) + rng.normal(0, 3, n))
    t = t.round(1)
    if anomalies and n > 460:
        t[[10, 200, 450]] = [400.0, 380.0, 420.0]             # planted anomalies
    latent = (0.12 * distance + np.select([traffic == "High", traffic == "Medium"], [2.0, 0.8], 0)
              + np.select([weather == "Rain", weather == "Fog"], [1.0, 1.6], 0) + rng.normal(0, 1.2, n))
    late = np.where(latent > np.quantile(latent, 0.65), "Late", "OnTime")      # not derived from Delivery_Time
    return pd.DataFrame({"Order_ID": np.arange(100000, 100000 + n), "Order_Date": dates, "City": city, "Vehicle_Type": vehicle,
                         "Distance": distance, "Traffic": traffic, "Weather": weather, "Delivery_Time": t, "Status": late})


def hr_df(n: int = 400, seed: int = 3) -> pd.DataFrame:
    """HR style data without any date column."""
    rng = np.random.default_rng(seed)
    dept = rng.choice(["Sales", "Engineering", "Support", "Finance"], n)
    exp = rng.integers(0, 25, n)
    perf = rng.integers(1, 6, n)
    salary = (30000 + exp * 2100 + perf * 1800 + np.select([dept == "Engineering", dept == "Finance"], [14000, 6000], 0) + rng.normal(0, 4000, n)).round(0)
    return pd.DataFrame({"Employee_ID": [f"E{i:05d}" for i in range(n)], "Department": dept, "Experience": exp, "Performance": perf, "Salary": salary})


def retail_df(n: int = 1100, seed: int = 11) -> pd.DataFrame:
    """Daily retail data with weekly seasonality and an upward trend (for trend/forecast)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    t = np.arange(n)
    units = 100 + 0.12 * t + 15 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 5, n)
    price = rng.uniform(10, 20, n)
    return pd.DataFrame({"Day": dates, "Store": rng.choice(["North", "South"], n), "Units_Sold": units.round(1), "Unit_Price": price.round(2),
                         "Revenue": (units * price).round(2)})
