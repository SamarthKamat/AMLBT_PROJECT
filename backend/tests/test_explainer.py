from explainer import check_grounded, digest
from executor import execute
from schemas import AnalysisPlan
from tests.conftest import make_ctx
from validator import validate_plan


def test_grounding_accepts_rounded_and_percent_forms(delivery):
    df, profile, roles, ctx = delivery
    r = execute(validate_plan(AnalysisPlan(operation="compare_groups", target_column="Delivery_Time", group_by=["City"], aggregation="mean"), df, profile, roles), ctx)
    v = digest(r)
    top = r["table"][0]["Average of Delivery_Time"]
    assert check_grounded(f"The leading city averages {top:.1f} minutes.", [v]) == []
    assert check_grounded(f"The leading city averages {top:.2f} minutes.", [v]) == []


def test_grounding_rejects_invented_numbers(delivery):
    df, profile, roles, ctx = delivery
    r = execute(validate_plan(AnalysisPlan(operation="get_summary", target_column="Delivery_Time", aggregation="mean"), df, profile, roles), ctx)
    assert check_grounded("Revenue will reach 48213.77 next quarter.", [digest(r)]) == [48213.77]


def test_digest_is_small_and_has_no_raw_dataset(delivery):
    df, profile, roles, ctx = delivery
    r = execute(validate_plan(AnalysisPlan(operation="detect_anomalies", target_column="Delivery_Time"), df, profile, roles), ctx)
    s = str(digest(r))
    assert len(s) < 6000
