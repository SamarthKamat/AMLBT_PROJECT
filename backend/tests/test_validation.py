"""Plans are validated against the real dataset: unknown columns rejected, insufficient data explained, nothing substituted."""
import numpy as np
import pandas as pd
import pytest

from errors import UDIError
from schemas import AnalysisPlan
from tests import datasets
from tests.conftest import make_ctx
from validator import ClarificationNeeded, validate_plan


def run(df, **plan):
    profile, roles, _ = make_ctx(df)
    return validate_plan(AnalysisPlan(**plan), df, profile, roles)


def test_missing_target_is_rejected_not_substituted():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"Sales": rng.uniform(100, 900, 60), "Cost": rng.uniform(50, 400, 60), "Revenue": rng.uniform(200, 1500, 60)})
    with pytest.raises(UDIError) as e:
        run(df, operation="predict_value", target_column="Profit")
    err = e.value
    assert err.code == "INVALID_COLUMN" and err.retryable is False
    assert "does not contain a 'Profit' variable" in err.message
    assert "Sales" in err.message and "Cost" in err.message and "Revenue" in err.message
    assert err.to_dict()["code"] == "INVALID_COLUMN" and "error" in err.to_dict()


def test_case_and_spacing_variants_are_canonicalised_but_other_columns_are_not():
    df = datasets.delivery_df()
    vp = run(df, operation="get_summary", target_column="delivery time", aggregation="mean")
    assert vp.plan.target_column == "Delivery_Time"
    with pytest.raises(UDIError) as e:
        run(df, operation="get_summary", target_column="Delivery", aggregation="mean")
    assert e.value.code == "INVALID_COLUMN"


@pytest.mark.parametrize("field,value", [("group_by", ["Nope"]), ("time_column", "Nope"), ("feature_columns", ["Distance", "Nope"]),
                                         ("filters", [{"column": "Nope", "operator": "eq", "value": "x"}])])
def test_any_unknown_column_reference_is_rejected(field, value):
    df = datasets.delivery_df()
    base = dict(operation="compare_groups", target_column="Delivery_Time", group_by=["City"], aggregation="mean")
    base[field] = value
    with pytest.raises(UDIError) as e:
        run(df, **base)
    assert e.value.code == "INVALID_COLUMN"


def test_non_numeric_target_rejected_for_regression():
    with pytest.raises(UDIError) as e:
        run(datasets.delivery_df(), operation="predict_value", target_column="City")
    assert e.value.code == "INVALID_PLAN" and "not a numeric column" in e.value.message


def test_unsupported_operation():
    with pytest.raises(UDIError) as e:
        run(datasets.delivery_df(), operation="unsupported", reasoning="Needs a controlled experiment.")
    assert e.value.code == "UNSUPPORTED_OPERATION" and "controlled experiment" in e.value.message


def test_clarification_when_required_input_missing():
    df = datasets.delivery_df()
    with pytest.raises(ClarificationNeeded) as c:
        run(df, operation="run_scenario", scenario_changes=[{"column": "Distance", "change_type": "percent", "value": 10}])
    assert "Delivery_Time" in c.value.options
    with pytest.raises(ClarificationNeeded):
        run(df, operation="compare_groups", target_column="Delivery_Time")
    with pytest.raises(ClarificationNeeded):
        run(df, operation="get_summary", aggregation="mean")


def test_filter_values_must_exist():
    df = datasets.delivery_df()
    ok = run(df, operation="get_summary", target_column="Delivery_Time", aggregation="mean", filters=[{"column": "City", "operator": "eq", "value": "goa"}])
    assert set(ok.data["City"]) == {"Goa"}
    with pytest.raises(UDIError) as e:
        run(df, operation="get_summary", target_column="Delivery_Time", aggregation="mean", filters=[{"column": "City", "operator": "eq", "value": "Atlantis"}])
    assert e.value.code == "INVALID_PLAN" and "Goa" in e.value.message


def test_filters_that_remove_every_row():
    df = datasets.delivery_df()
    with pytest.raises(UDIError) as e:
        run(df, operation="get_summary", target_column="Delivery_Time", aggregation="mean", filters=[{"column": "Distance", "operator": "gt", "value": "9999"}])
    assert e.value.code == "INSUFFICIENT_DATA"


def test_date_range_filter():
    df = datasets.retail_df()
    vp = run(df, operation="get_summary", target_column="Revenue", aggregation="sum", start_date="2023-02-01", end_date="2023-02-28")
    assert len(vp.data) == 28


def test_insufficient_data_regression_classification_forecast_anomaly():
    small = datasets.delivery_df(25)
    with pytest.raises(UDIError) as e:
        run(small, operation="predict_value", target_column="Delivery_Time")
    assert e.value.code == "INSUFFICIENT_DATA" and "30" in e.value.message
    # class with fewer than 5 observations
    df = datasets.delivery_df(200)
    df.loc[df.index[:3], "Status"] = "Rare"
    with pytest.raises(UDIError) as e:
        run(df, operation="predict_class", target_column="Status")
    assert e.value.code == "INSUFFICIENT_DATA" and "Rare" in e.value.message
    # single class
    df2 = datasets.delivery_df(100)
    df2["Status"] = "Late"
    with pytest.raises(UDIError) as e:
        run(df2, operation="predict_class", target_column="Status")
    assert e.value.code == "INSUFFICIENT_DATA"
    with pytest.raises(UDIError) as e:
        run(datasets.delivery_df(12), operation="detect_anomalies", target_column="Delivery_Time")
    assert e.value.code == "INSUFFICIENT_DATA"


def test_continuous_column_is_not_a_class_label():
    with pytest.raises(UDIError) as e:
        run(datasets.delivery_df(), operation="predict_class", target_column="Delivery_Time")
    assert e.value.code == "INVALID_PLAN" and "continuous" in e.value.message


def test_no_time_column_blocks_time_operations():
    df = datasets.hr_df()
    for op, extra in (("forecast", {"target_column": "Salary"}), ("analyze_trend", {"target_column": "Salary"}), ("get_current_state", {})):
        with pytest.raises(UDIError) as e:
            run(df, operation=op, **extra)
        assert e.value.code == "INSUFFICIENT_DATA" and "date" in e.value.message.lower()


def test_scenario_validation():
    df = datasets.delivery_df()
    with pytest.raises(UDIError):
        run(df, operation="run_scenario", target_column="Delivery_Time", scenario_changes=[{"column": "Delivery_Time", "change_type": "percent", "value": 5}])
    with pytest.raises(UDIError):
        run(df, operation="run_scenario", target_column="Delivery_Time", scenario_changes=[{"column": "Distance", "change_type": "percent", "value": -100}])
    with pytest.raises(UDIError) as e:
        run(df, operation="run_scenario", target_column="Delivery_Time", scenario_changes=[{"column": "City", "change_type": "percent", "value": 5}])
    assert e.value.code == "INVALID_PLAN"
    with pytest.raises(UDIError) as e:
        run(df, operation="run_scenario", target_column="Delivery_Time", scenario_changes=[{"column": "Speed", "change_type": "percent", "value": 5}])
    assert e.value.code == "INVALID_COLUMN"


def test_prediction_input_values_checked():
    df = datasets.delivery_df()
    with pytest.raises(UDIError):
        run(df, operation="predict_value", target_column="Delivery_Time", input_values=[{"column": "Traffic", "value": "Gridlock"}])
    with pytest.raises(UDIError):
        run(df, operation="predict_value", target_column="Delivery_Time", input_values=[{"column": "Distance", "value": "far"}])
    assert run(df, operation="predict_value", target_column="Delivery_Time", input_values=[{"column": "Traffic", "value": "high"}]).plan.input_values[0].value == "High"
