"""Deterministic operations return exactly what pandas computes, on schemas unrelated to the bundled samples."""
import numpy as np
import pandas as pd
import pytest

from errors import UDIError
from executor import execute
from schemas import AnalysisPlan
from tests import datasets
from tests.conftest import make_ctx
from validator import validate_plan


def go(df, ctx_bundle=None, **plan):
    profile, roles, ctx = ctx_bundle or make_ctx(df)
    return execute(validate_plan(AnalysisPlan(**plan), df, profile, roles), ctx), ctx


# ---------------------------------------------------------------- summary
@pytest.mark.parametrize("agg,fn", [("mean", "mean"), ("sum", "sum"), ("median", "median"), ("min", "min"), ("max", "max"), ("std", "std")])
def test_summary_aggregations_match_pandas(delivery, agg, fn):
    df, profile, roles, ctx = delivery
    r, _ = go(df, (profile, roles, ctx), operation="get_summary", target_column="Delivery_Time", aggregation=agg)
    expected = getattr(df["Delivery_Time"], fn)()
    value = next(v for k, v in r["metrics"].items() if "Delivery_Time" in k and k != "Records")
    assert value == pytest.approx(expected)
    assert r["charts"] and r["table"]


def test_summary_count_distinct_distribution_and_overview(delivery):
    df, profile, roles, ctx = delivery
    r, _ = go(df, (profile, roles, ctx), operation="get_summary", aggregation="count")
    assert r["metrics"]["Records"] == len(df)
    r, _ = go(df, (profile, roles, ctx), operation="get_summary", target_column="City", aggregation="nunique")
    assert r["metrics"]["Distinct City"] == df["City"].nunique()
    r, _ = go(df, (profile, roles, ctx), operation="get_summary", target_column="Traffic", aggregation="percentage")
    top = df["Traffic"].value_counts(normalize=True).iloc[0] * 100
    assert r["metrics"]["Most common %"] == pytest.approx(round(top, 2))
    r, _ = go(df, (profile, roles, ctx), operation="get_summary")
    assert r["metrics"]["Records"] == len(df) and any(row["column"] == "Distance" for row in r["table"])


def test_summary_rejects_average_of_text(delivery):
    df, profile, roles, ctx = delivery
    with pytest.raises(UDIError) as e:
        go(df, (profile, roles, ctx), operation="get_summary", target_column="City", aggregation="mean")
    assert e.value.code == "INVALID_PLAN"


def test_summary_with_filter(delivery):
    df, profile, roles, ctx = delivery
    r, _ = go(df, (profile, roles, ctx), operation="get_summary", target_column="Delivery_Time", aggregation="mean",
              filters=[{"column": "City", "operator": "eq", "value": "Goa"}])
    assert r["metrics"]["Average of Delivery_Time"] == pytest.approx(df[df.City == "Goa"].Delivery_Time.mean())
    assert r["provenance"]["rows_analysed"] == (df.City == "Goa").sum()


# ---------------------------------------------------------------- comparison
def test_compare_groups_ranks_like_pandas(delivery):
    df, profile, roles, ctx = delivery
    r, _ = go(df, (profile, roles, ctx), operation="compare_groups", target_column="Delivery_Time", group_by=["Traffic"], aggregation="mean", sort_order="desc")
    exp = df.groupby("Traffic").Delivery_Time.mean().sort_values(ascending=False)
    assert r["metrics"]["Top group"] == exp.index[0] and r["metrics"]["Top value"] == pytest.approx(exp.iloc[0])
    assert r["metrics"]["Kruskal–Wallis p-value"] < 0.05
    asc, _ = go(df, (profile, roles, ctx), operation="compare_groups", target_column="Delivery_Time", group_by=["Traffic"], aggregation="mean", sort_order="asc")
    assert asc["metrics"]["Top group"] == exp.index[-1]


def test_compare_counts_two_groups_and_categorical_measure(delivery):
    df, profile, roles, ctx = delivery
    r, _ = go(df, (profile, roles, ctx), operation="compare_groups", group_by=["City"])
    assert sum(row["Records"] for row in r["table"]) == len(df)
    r, _ = go(df, (profile, roles, ctx), operation="compare_groups", target_column="Delivery_Time", group_by=["City", "Traffic"], aggregation="sum")
    assert len(r["table"]) == df.groupby(["City", "Traffic"]).ngroups
    r, _ = go(df, (profile, roles, ctx), operation="compare_groups", target_column="Status", group_by=["Traffic"])
    assert r["metrics"]["Chi-square p-value"] < 0.05


# ---------------------------------------------------------------- trend
def test_trend_detects_real_increase_and_no_trend(retail):
    df, profile, roles, ctx = retail
    r, _ = go(df, (profile, roles, ctx), operation="analyze_trend", target_column="Units_Sold", aggregation="mean")
    assert r["metrics"]["Direction"] == "increasing" and r["metrics"]["Fitted change %"] > 10
    assert r["charts"][0]["type"] == "line" and len(r["charts"][0]["data"]) > 10
    flat = datasets.delivery_df(anomalies=False)
    r2, _ = go(flat, operation="analyze_trend", target_column="Delivery_Time", aggregation="mean")
    assert r2["metrics"]["Direction"] == "no statistically significant trend"
    assert any("cause" in x for x in r2["limitations"])


def test_trend_uses_whatever_the_date_column_is_called_and_groups(retail):
    df, profile, roles, ctx = retail
    assert "Day" in df.columns and "Date" not in df.columns
    r, _ = go(df, (profile, roles, ctx), operation="analyze_trend", target_column="Revenue", aggregation="sum", group_by=["Store"], frequency="M")
    assert {row["Store"] for row in r["table"]} == {"North", "South"}


def test_trend_record_count_when_no_target(delivery):
    df, profile, roles, ctx = delivery
    r, _ = go(df, (profile, roles, ctx), operation="analyze_trend")
    assert r["provenance"]["aggregation"] == "count"


# ---------------------------------------------------------------- current state
def test_current_state_is_latest_valid_observation(delivery):
    df, profile, roles, ctx = delivery
    d2 = df.copy()
    d2.loc[d2.index[-1], "Delivery_Time"] = np.nan          # latest row has no value -> latest VALID observation
    r, _ = go(d2, operation="get_current_state", target_column="Delivery_Time")
    assert r["metrics"]["As of"] == d2.dropna(subset=["Delivery_Time"])["Order_Date"].max().strftime("%Y-%m-%d")
    assert r["metrics"]["Latest Delivery_Time"] == pytest.approx(d2.dropna(subset=["Delivery_Time"]).iloc[-1]["Delivery_Time"])
    g, _ = go(df, (profile, roles, ctx), operation="get_current_state", target_column="Delivery_Time", group_by=["City"])
    assert len(g["table"]) == df.City.nunique()


# ---------------------------------------------------------------- correlation
def test_correlation_identifies_real_drivers(delivery):
    df, profile, roles, ctx = delivery
    clean = datasets.delivery_df(anomalies=False)
    r, _ = go(clean, operation="calculate_correlation", target_column="Delivery_Time")
    top = r["table"][0]
    assert top["factor"] in ("Distance", "Status") or top["strength"] > 0.5
    dist = next(x for x in r["table"] if x["factor"] == "Distance")
    assert dist["strength"] == pytest.approx(clean.Distance.corr(clean.Delivery_Time), abs=1e-6)
    assert any(x["factor"] == "Traffic" and x["type"] == "categorical" for x in r["table"])
    assert "Order_ID" not in {x["factor"] for x in r["table"]}              # identifiers are not factors
    assert any("causation" in x for x in r["limitations"]) and "not causation" in r["summary"].lower() or "causation" in r["summary"].lower()


def test_correlation_overall_and_categorical_target(delivery):
    df, profile, roles, ctx = delivery
    r, _ = go(df, (profile, roles, ctx), operation="calculate_correlation")
    assert r["table"][0]["variable_1"] and r["metrics"]["Pairs examined"] >= 1
    r, _ = go(df, (profile, roles, ctx), operation="calculate_correlation", target_column="Status")
    assert any(x["measure"] == "Cramér's V" for x in r["table"])


# ---------------------------------------------------------------- anomalies
def test_anomalies_find_planted_values(delivery):
    df, profile, roles, ctx = delivery
    r, _ = go(df, (profile, roles, ctx), operation="detect_anomalies", target_column="Delivery_Time")
    flagged = {row["row_index"] for row in r["table"]}
    assert {10, 200, 450} <= flagged
    assert any("not mean" in x or "does not mean" in x for x in r["limitations"])
    for m in ("iqr", "zscore", "isolation_forest"):
        rm, _ = go(df, (profile, roles, ctx), operation="detect_anomalies", target_column="Delivery_Time", anomaly_method=m)
        assert {10, 200, 450} <= {row["row_index"] for row in rm["table"]}


def test_multivariate_anomalies_without_target(delivery):
    df, profile, roles, ctx = delivery
    r, _ = go(df, (profile, roles, ctx), operation="detect_anomalies")
    assert r["metrics"]["Method"].startswith("Isolation Forest") and r["table"]
    assert 10 in {row["row_index"] for row in r["table"]} or 200 in {row["row_index"] for row in r["table"]} or 450 in {row["row_index"] for row in r["table"]}


# ---------------------------------------------------------------- no ML for descriptive questions
def test_descriptive_operations_do_not_train_models(delivery):
    df, profile, roles, ctx = delivery
    fresh = make_ctx(df, "fresh")
    for plan in (dict(operation="get_summary", target_column="Delivery_Time", aggregation="mean"),
                 dict(operation="compare_groups", target_column="Delivery_Time", group_by=["City"], aggregation="mean"),
                 dict(operation="detect_anomalies", target_column="Delivery_Time"),
                 dict(operation="calculate_correlation", target_column="Delivery_Time"),
                 dict(operation="analyze_trend", target_column="Delivery_Time", aggregation="mean"),
                 dict(operation="get_current_state")):
        r, _ = go(df, fresh, **plan)
        assert r["provenance"]["uses_machine_learning"] is False
        assert r["evaluation"] is None
    assert len(fresh[2].cache) == 0


def test_results_are_strict_json(delivery):
    import json
    df, profile, roles, ctx = delivery
    r, _ = go(df, (profile, roles, ctx), operation="detect_anomalies", target_column="Delivery_Time")
    json.dumps(r, allow_nan=False)
