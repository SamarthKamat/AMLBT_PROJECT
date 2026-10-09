"""Domain reference models are reference/validation artifacts only. These tests prove (a) they are intact and reproduce their
reported metrics, (b) uploaded data trains its own models, and (c) the dynamic pipeline is benchmarked against the reference."""
import pandas as pd
import pytest

import config
from config import DOMAINS
from executor import execute
from reference_models import list_reference_models, validate_domain
from schemas import AnalysisPlan
from tests.conftest import make_ctx
from validator import validate_plan


def test_seven_domains_are_context_only():
    assert set(DOMAINS) == {"company", "hospital", "college", "municipality", "ngo", "community", "event_organizer"}
    for d in DOMAINS.values():
        assert set(d) == {"label", "icon", "description", "context"}              # no target / model keys: nothing locks a model to a domain


@pytest.mark.parametrize("domain", list(DOMAINS))
def test_reference_models_reproduce_reported_metrics(domain):
    r = validate_domain(domain)
    assert r["regression_reproduced"] and r["classification_reproduced"] and r["artifacts_present"] >= 4          # selected regression + classification, forecasting, anomaly (the other candidates are an optional add-on)


def test_listing_marks_them_reference_only():
    items = list_reference_models()
    assert len(items) == 7 and all("reference" in i["purpose"] for i in items)


def test_all_sample_datasets_load_and_profile():
    for d in DOMAINS:
        df = pd.read_csv(config.SAMPLE_DIR / f"{d}.csv")
        assert len(df) >= 400
        profile, roles, ctx = make_ctx(df, d)
        assert profile["rows"] == len(df)


def test_dynamic_pipeline_matches_reference_on_company_profit():
    """Same data, trained from scratch at question time: should be competitive with the shipped reference model."""
    meta = [i for i in list_reference_models() if i["domain"] == "company"][0]
    target = meta["regression_target"]
    ref_r2 = meta["regression_candidates"][meta["selected_regression"].split("_regression_")[1].replace(".joblib", "")]["R2"]
    df = pd.read_csv(config.SAMPLE_DIR / "company.csv")
    profile, roles, ctx = make_ctx(df, "company")
    r = execute(validate_plan(AnalysisPlan(operation="predict_value", target_column=target), df, profile, roles), ctx)
    dyn = r["evaluation"]["metrics"]["R2"]
    assert dyn >= ref_r2 - 0.08, (dyn, ref_r2)
    assert r["evaluation"]["selected_model"] and r["provenance"]["uses_machine_learning"]


def test_dynamic_pipeline_works_on_every_sample_without_domain_logic():
    """Each domain sample: pick a numeric target from the data itself (not from config) and run the dynamic regression."""
    for d in DOMAINS:
        df = pd.read_csv(config.SAMPLE_DIR / f"{d}.csv")
        profile, roles, ctx = make_ctx(df, d)
        numeric = [c for c, v in roles.items() if v["role"] == "numeric" and not v.get("high_cardinality") is None]
        target = [c for c in numeric if df[c].nunique() > 30][0]
        r = execute(validate_plan(AnalysisPlan(operation="predict_value", target_column=target), df, profile, roles), ctx)
        assert r["evaluation"]["metrics"]["RMSE"] >= 0 and r["evaluation"]["selected_model"]
