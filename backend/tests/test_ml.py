"""Dynamic model selection, honest evaluation, explainability, scenarios and the model cache."""
import time
import numpy as np
import pandas as pd
import pytest

from errors import UDIError
from executor import execute
from schemas import AnalysisPlan
from tests import datasets
from tests.conftest import make_ctx
from validator import validate_plan


def go(df, bundle, **plan):
    profile, roles, ctx = bundle
    return execute(validate_plan(AnalysisPlan(**plan), df, profile, roles), ctx)


@pytest.fixture(scope="module")
def clean_delivery():
    df = datasets.delivery_df(anomalies=False)
    return df, make_ctx(df, "clean-delivery")


def test_regression_report_contents(clean_delivery):
    df, bundle = clean_delivery
    r = go(df, bundle, operation="predict_value", target_column="Delivery_Time")
    ev = r["evaluation"]
    names = {c["model"] for c in ev["candidates"]}
    assert {"Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "Gradient Boosting Regressor"} <= names
    assert set(ev["metrics"]) == {"MAE", "RMSE", "R2"} and ev["metrics"]["R2"] > 0.9
    assert ev["selected_model"] == next(c["model"] for c in ev["candidates"] if c.get("selected"))
    assert ev["train_rows"] + ev["test_rows"] == 700 and ev["cv_folds"] >= 3
    assert "Order_ID" in [e["column"] for e in ev["features_excluded"]]                # identifier not used
    assert "Order_Date" in [e["column"] for e in ev["features_excluded"]]
    assert "Order_ID" not in ev["features_used"]
    # explainability: Distance is the dominant numeric driver in the data
    items = r["explainability"]["items"]
    assert items[0]["feature"] in ("Distance", "Traffic") and sum(i["share_%"] for i in items) == pytest.approx(100, abs=0.5)
    assert r["prediction"]["value"] > 0 and r["charts"][0]["type"] == "scatter"
    assert r["metrics"]["Selected model"] == ev["selected_model"]


def test_selection_adapts_to_the_data():
    rng = np.random.default_rng(0)
    n = 600
    x1, x2 = rng.uniform(-3, 3, n), rng.uniform(-3, 3, n)
    linear = pd.DataFrame({"a": x1, "b": x2, "y": 4 * x1 - 2 * x2 + rng.normal(0, .05, n)})
    nonlin = pd.DataFrame({"a": x1, "b": x2, "y": np.where(x1 > 0, 10, -10) * np.sign(x2) + rng.normal(0, .3, n)})   # interaction / step
    b1, b2 = make_ctx(linear, "lin"), make_ctx(nonlin, "nonlin")
    r1 = go(linear, b1, operation="predict_value", target_column="y")
    r2 = go(nonlin, b2, operation="predict_value", target_column="y")
    assert r1["evaluation"]["selected_model"] == "Linear Regression"
    assert r2["evaluation"]["selected_model"] != "Linear Regression"
    assert r2["evaluation"]["metrics"]["R2"] > 0.9


def test_prediction_with_user_inputs_and_direction(clean_delivery):
    df, bundle = clean_delivery
    near = go(df, bundle, operation="predict_value", target_column="Delivery_Time",
              input_values=[{"column": "Distance", "value": "2"}, {"column": "Traffic", "value": "Low"}, {"column": "Weather", "value": "Clear"}])
    far = go(df, bundle, operation="predict_value", target_column="Delivery_Time",
             input_values=[{"column": "Distance", "value": "28"}, {"column": "Traffic", "value": "High"}, {"column": "Weather", "value": "Clear"}])
    assert far["prediction"]["value"] > near["prediction"]["value"] + 30
    assert far["prediction"]["basis"] == "user-specified values"
    assert "Distance" in far["prediction"]["inputs_given"]
    with pytest.raises(UDIError) as e:
        go(df, bundle, operation="predict_value", target_column="Delivery_Time", input_values=[{"column": "Order_ID", "value": "5"}])
    assert e.value.code == "INVALID_PLAN"


def test_classification_report_contents(clean_delivery):
    df, bundle = clean_delivery
    r = go(df, bundle, operation="predict_class", target_column="Status")
    ev = r["evaluation"]
    assert {"Accuracy", "Precision", "Recall", "F1", "ROC_AUC"} <= set(ev["metrics"]) and ev["metrics"]["ROC_AUC"] > 0.8
    assert ev["metrics"]["Accuracy"] > ev["baseline"]["accuracy"]
    cm = ev["confusion_matrix"]
    assert cm["labels"] == ["Late", "OnTime"] and np.array(cm["matrix"]).sum() == ev["test_rows"]
    assert {c["model"] for c in ev["candidates"]} >= {"Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier", "Gradient Boosting Classifier"}
    assert r["prediction"]["class"] in ("Late", "OnTime") and abs(sum(p["probability"] for p in r["prediction"]["probabilities"]) - 1) < 1e-6
    assert r["tables"]["per_class"] and r["explainability"]["items"]


def test_multiclass_classification(clean_delivery):
    df, bundle = clean_delivery
    r = go(df, bundle, operation="predict_class", target_column="Traffic")
    assert len(r["evaluation"]["classes"]) == 3
    assert r["evaluation"]["metrics"]["F1"] >= 0 and np.array(r["evaluation"]["confusion_matrix"]["matrix"]).shape == (3, 3)


def test_numeric_coded_class_target():
    rng = np.random.default_rng(2)
    n = 300
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "z": rng.normal(size=n), "risk": (x + rng.normal(0, .5, n) > 0).astype(int)})
    r = go(df, make_ctx(df, "coded"), operation="predict_class", target_column="risk")
    assert r["evaluation"]["metrics"]["Accuracy"] > 0.75


def test_model_cache_reuse_and_invalidation(clean_delivery):
    df, _ = clean_delivery
    bundle = make_ctx(df, "cache-test")
    ctx = bundle[2]
    t0 = time.time()
    first = go(df, bundle, operation="predict_value", target_column="Delivery_Time")
    t_first = time.time() - t0
    assert first["evaluation"]["cached_model"] is False and ctx.cache.misses == 1
    t0 = time.time()
    second = go(df, bundle, operation="predict_value", target_column="Delivery_Time",
                input_values=[{"column": "Distance", "value": "10"}])
    t_second = time.time() - t0
    assert second["evaluation"]["cached_model"] is True and ctx.cache.hits == 1 and t_second < t_first / 3
    assert first["evaluation"]["selected_model"] == second["evaluation"]["selected_model"]
    # scenario on the same target/features reuses the trained model
    scen = go(df, bundle, operation="run_scenario", target_column="Delivery_Time",
              scenario_changes=[{"column": "Distance", "change_type": "percent", "value": 20}])
    assert scen["evaluation"]["cached_model"] is True and ctx.cache.hits == 2
    # different target / task / filters / features -> new models
    go(df, bundle, operation="predict_class", target_column="Status")
    go(df, bundle, operation="predict_value", target_column="Delivery_Time", filters=[{"column": "City", "operator": "eq", "value": "Goa"}])
    go(df, bundle, operation="predict_value", target_column="Delivery_Time", feature_columns=["Distance"])
    assert len(ctx.cache) == 4
    # clearing the session (new chat / new upload) clears the cache
    ctx.cache.clear()
    assert len(ctx.cache) == 0


def test_scenario_semantics(clean_delivery):
    df, bundle = clean_delivery
    up = go(df, bundle, operation="run_scenario", target_column="Delivery_Time", scenario_changes=[{"column": "Distance", "change_type": "percent", "value": 20}])
    down = go(df, bundle, operation="run_scenario", target_column="Delivery_Time", scenario_changes=[{"column": "Distance", "change_type": "percent", "value": -20}])
    s = up["scenario"]
    assert s["scenario"] > s["baseline"] and down["scenario"]["scenario"] < down["scenario"]["baseline"]
    assert s["change"] == pytest.approx(s["scenario"] - s["baseline"]) and s["change_pct"] == pytest.approx(s["change"] / s["baseline"] * 100)
    # the estimate should be in the right ballpark of the true +20% distance effect (2.2 min/km * mean distance * 0.2)
    true_delta = 2.2 * df.Distance.mean() * 0.2
    assert s["change"] == pytest.approx(true_delta, rel=0.35)
    assert any("does not establish causal impact" in x for x in up["limitations"]) and "causal" in up["summary"]
    assert up["charts"][1]["type"] == "line" and len(up["charts"][1]["data"]) == 13        # sensitivity curve
    absolute = go(df, bundle, operation="run_scenario", target_column="Delivery_Time", scenario_changes=[{"column": "Distance", "change_type": "absolute", "value": 5}])
    assert absolute["scenario"]["change"] > 0
    both = go(df, bundle, operation="run_scenario", target_column="Delivery_Time",
              scenario_changes=[{"column": "Distance", "change_type": "percent", "value": 10}, {"column": "Distance", "change_type": "set", "value": 15}])
    assert both["scenario"]["scenario"] != both["scenario"]["baseline"]


def test_scenario_flags_irrelevant_driver():
    rng = np.random.default_rng(5)
    n = 400
    df = pd.DataFrame({"driver": rng.normal(10, 2, n), "noise_col": rng.normal(0, 1, n)})
    df["out"] = 3 * df["driver"] + rng.normal(0, .5, n)
    r = go(df, make_ctx(df, "irrelevant"), operation="run_scenario", target_column="out", scenario_changes=[{"column": "noise_col", "change_type": "percent", "value": 50}])
    assert any("nearly irrelevant" in x for x in r["limitations"])


def test_leakage_and_weak_model_warnings():
    rng = np.random.default_rng(3)
    n = 300
    a = rng.normal(100, 20, n)
    df = pd.DataFrame({"a": a, "b": rng.normal(size=n)})
    df["total"] = a * 2
    leak = go(df, make_ctx(df, "leak"), operation="predict_value", target_column="total")
    assert any("almost perfectly correlated" in x for x in leak["limitations"])
    weak = pd.DataFrame({"u": rng.normal(size=n), "v": rng.normal(size=n), "y": rng.normal(size=n)})
    r = go(weak, make_ctx(weak, "weak"), operation="predict_value", target_column="y")
    assert any("explains little" in x for x in r["limitations"])


def test_feature_selection_request_is_honoured(clean_delivery):
    df, bundle = clean_delivery
    r = go(df, bundle, operation="predict_value", target_column="Delivery_Time", feature_columns=["Distance", "Traffic", "Order_ID"])
    assert set(r["evaluation"]["features_used"]) == {"Distance", "Traffic"}
    assert any("Order_ID" in x for x in r["limitations"])


def test_reference_models_are_never_loaded_at_runtime():
    import ml_engine, executor, app, planner, validator
    for mod in (ml_engine, executor, planner, validator):
        src = open(mod.__file__, encoding="utf8").read()
        assert "reference_models" not in src and "joblib" not in src, mod.__name__
    src = open(app.__file__, encoding="utf8").read()
    assert "load_selected" not in src and "can_use" not in src and "joblib" not in src
