"""Forecasting: time-aware backtesting, sufficiency checks and honest baselines."""
import numpy as np
import pandas as pd
import pytest

import forecasting_engine as fe
from errors import UDIError
from executor import execute
from schemas import AnalysisPlan
from tests import datasets
from tests.conftest import make_ctx
from validator import validate_plan


def go(df, bundle=None, **plan):
    profile, roles, ctx = bundle or make_ctx(df)
    return execute(validate_plan(AnalysisPlan(**plan), df, profile, roles), ctx)


def test_forecast_continues_after_the_last_observation(retail):
    df, profile, roles, ctx = retail
    r = go(df, (profile, roles, ctx), operation="forecast", target_column="Units_Sold", frequency="D", forecast_horizon=14, aggregation="mean")
    rows = r["table"]
    assert len(rows) == 14 and rows[0]["period"] == (df.Day.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    assert all(x["lower_80"] < x["forecast"] < x["upper_80"] for x in rows)
    ev = r["evaluation"]
    assert ev["backtest"]["folds"] >= 1 and set(ev["metrics"]) == {"MAE", "RMSE", "MAPE"}
    assert any(c["model"] == "Naive (last value)" for c in ev["candidates"]) and ev["baseline"]
    assert ev["skill_vs_baseline_%"] is not None and ev["skill_vs_baseline_%"] > 0         # trend + weekly cycle beat naive
    # trending series: forecasts are around the end level, not around the historical mean
    assert np.mean([x["forecast"] for x in rows]) > df.Units_Sold.mean()
    chart = r["charts"][0]
    assert chart["type"] == "forecast" and any("actual" in d for d in chart["data"]) and any("forecast" in d for d in chart["data"])


def test_backtest_is_time_ordered_never_shuffled():
    for n, h in [(100, 14), (24, 6), (60, 30), (400, 30)]:
        h_bt, cuts = fe.plan_backtest(n, min(h, n // 3))
        assert cuts == sorted(cuts) and cuts[0] >= fe.MIN_TRAIN
        for c in cuts:
            assert c + h_bt <= n              # test block lies entirely after its training window
        assert all(b - a == h_bt for a, b in zip(cuts, cuts[1:]))


def test_row_order_does_not_change_forecast(retail):
    df, *_ = retail
    shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
    a = go(df, operation="forecast", target_column="Units_Sold", frequency="W", forecast_horizon=6, aggregation="mean")
    b = go(shuffled, operation="forecast", target_column="Units_Sold", frequency="W", forecast_horizon=6, aggregation="mean")
    assert [x["forecast"] for x in a["table"]] == pytest.approx([x["forecast"] for x in b["table"]])


def test_monthly_forecast_from_daily_data(retail):
    df, profile, roles, ctx = retail
    r = go(df, (profile, roles, ctx), operation="forecast", target_column="Revenue", frequency="M", forecast_horizon=1, aggregation="sum")
    assert len(r["table"]) == 1 and r["forecast"]["frequency"] == "M"
    last_month_total = df[df.Day >= "2025-12-01"].Revenue.sum() if df.Day.max() >= pd.Timestamp("2025-12-01") else None
    assert r["table"][0]["forecast"] > 0


def test_insufficient_history_is_explained():
    df = datasets.retail_df(60)
    with pytest.raises(UDIError) as e:
        go(df, operation="forecast", target_column="Units_Sold", frequency="M", forecast_horizon=1, aggregation="mean")
    assert e.value.code == "INSUFFICIENT_DATA" and "24" in e.value.message
    short = datasets.retail_df(20)
    with pytest.raises(UDIError) as e:
        go(short, operation="forecast", target_column="Units_Sold", aggregation="mean")
    assert e.value.code == "INSUFFICIENT_DATA"
    # horizon longer than a third of the history
    with pytest.raises(UDIError) as e:
        go(datasets.retail_df(90), operation="forecast", target_column="Units_Sold", frequency="D", forecast_horizon=60, aggregation="mean")
    assert e.value.code == "INSUFFICIENT_DATA" and "horizon" in e.value.message


def test_sparse_series_rejected():
    df = datasets.retail_df(400)
    keep = df.sample(60, random_state=0).sort_values("Day")                 # 60 scattered days across 400
    with pytest.raises(UDIError) as e:
        go(keep, operation="forecast", target_column="Units_Sold", frequency="D", forecast_horizon=7, aggregation="mean")
    assert e.value.code == "INSUFFICIENT_DATA"


def test_no_skill_is_reported_for_pure_noise():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"when": pd.date_range("2024-01-01", periods=200), "v": rng.normal(50, 5, 200)})
    r = go(df, operation="forecast", target_column="v", frequency="D", forecast_horizon=10, aggregation="mean")
    ev = r["evaluation"]
    # on noise nothing meaningfully beats the baselines; the report must not claim a large skill
    assert ev["skill_vs_baseline_%"] is None or ev["skill_vs_baseline_%"] < 30
    assert any("hold" in x or "assume" in x for x in r["limitations"])


def test_forecast_group_by_is_ignored_with_warning(retail):
    df, profile, roles, ctx = retail
    r = go(df, (profile, roles, ctx), operation="forecast", target_column="Units_Sold", group_by=["Store"], frequency="W", forecast_horizon=4, aggregation="mean")
    assert any("filter" in w for w in r["warnings"])
    only = go(df, (profile, roles, ctx), operation="forecast", target_column="Units_Sold", frequency="W", forecast_horizon=4, aggregation="mean",
              filters=[{"column": "Store", "operator": "eq", "value": "North"}])
    assert only["provenance"]["rows_analysed"] == (df.Store == "North").sum()
