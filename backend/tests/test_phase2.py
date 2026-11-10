"""Phase 2 tests: semantics, relationships, data quality + cleaning, SHAP, AutoML selection, forecasting 2.0,
business operations, reports, local planner, auth / rate limiting / persistence and the new API endpoints."""
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from app import create_app
from data_profiler import profile_dataframe
from data_quality import assess_quality, clean_dataset
from executor import execute
from explainability_engine import shapley_values
from local_planner import LocalPlanner
from ml_engine import _one_se_select
from ops import RateLimiter
from relationships import analyse_relationships
from schemas import AnalysisPlan, OPERATIONS
from semantics import infer_semantics, default_aggregation
from session_manager import SessionStore
from storage import Storage
from tests import datasets
from tests.conftest import make_ctx
from tests.fake_llm import DownLLM
from validator import validate_plan
from errors import UDIError


def sales_df(n=600, seed=5):
    rng = np.random.default_rng(seed)
    city = rng.choice(["Pune", "Goa", "Delhi", "Chennai", "Kolkata", "Jaipur"], n)
    state = pd.Series(city).map({"Pune": "Maharashtra", "Goa": "Goa", "Delhi": "Delhi", "Chennai": "Tamil Nadu", "Kolkata": "West Bengal", "Jaipur": "Rajasthan"})
    country = np.where(rng.random(n) < 1, "India", "India")
    region = pd.Series(city).map({"Pune": "West", "Goa": "West", "Delhi": "North", "Chennai": "South", "Kolkata": "East", "Jaipur": "North"})
    units = rng.integers(1, 20, n)
    price = rng.uniform(50, 500, n).round(2)
    sales = (units * price).round(2)
    cost = (sales * rng.uniform(0.5, 0.95, n)).round(2)
    return pd.DataFrame({"Order_Date": pd.date_range("2023-01-01", periods=n, freq="D"), "City": city, "State": state, "Region": region, "Country": country,
                         "Units": units, "Unit_Price": price, "Sales": sales, "Cost": cost, "Profit": (sales - cost).round(2),
                         "Category": rng.choice(["Tech", "Furniture", "Office"], n)})


@pytest.fixture(scope="module")
def sales():
    df = sales_df()
    return (df, *make_ctx(df, "sales"))


# ----------------------------------------------------------------------------- semantics & relationships
def test_semantics_and_default_aggregation(sales):
    df, prof, roles, ctx = sales
    sem = infer_semantics(df, prof)
    assert sem["Sales"]["semantic"] == "money" and sem["Sales"]["additive"]
    assert sem["Unit_Price"]["additive"] is False                      # a price is a level, not summable
    assert sem["City"]["semantic"] == "geography"
    assert default_aggregation(sem, "Sales") == "sum" and default_aggregation(sem, "Unit_Price") == "mean"


def test_relationships_find_hierarchy_and_formula(sales):
    df, prof, roles, ctx = sales
    rel = analyse_relationships(df, prof, infer_semantics(df, prof))
    links = {(h["child"], h["parent"]) for h in rel["hierarchies"]}
    assert ("City", "Region") in links or ("City", "State") in links
    formulas = {f["column"]: f["formula"] for f in rel["derived"]}
    assert "Profit" in formulas and "Sales" in formulas["Profit"] and "Cost" in formulas["Profit"]
    assert len(rel["dictionary"]) == df.shape[1]
    m = rel["association"]["matrix"]
    assert all(abs(m[i][i] - 1) < 1e-9 for i in range(len(m)))


# ----------------------------------------------------------------------------- data quality
def test_quality_detects_injected_defects_and_cleaning_fixes_them():
    df = datasets.hr_df(400).copy()
    rng = np.random.default_rng(0)
    df.loc[rng.choice(400, 40, replace=False), "Salary"] = np.nan
    df.loc[rng.choice(400, 10, replace=False), "Experience"] = -3
    df["Department"] = df["Department"].astype(object)
    df.loc[rng.choice(400, 15, replace=False), "Department"] = "sales"           # case variant of 'Sales'
    df = pd.concat([df, df.head(12)], ignore_index=True)                         # 12 duplicates
    prof = profile_dataframe(df)
    sem = infer_semantics(df, prof)
    q = assess_quality(df, prof, sem)
    dims = {i["dimension"] for i in q["issues"]}
    assert {"completeness", "uniqueness", "validity", "consistency"} <= dims
    assert q["duplicate_rows"] == 12 and 0 < q["score"] < 100
    clean, log = clean_dataset(df, prof, sem)
    p2 = profile_dataframe(clean)
    q2 = assess_quality(clean, p2, infer_semantics(clean, p2))
    assert q2["score"] > q["score"] and q2["duplicate_rows"] == 0 and clean.isna().sum().sum() == 0
    assert (clean["Experience"] >= 0).all() and "sales" not in set(clean["Department"])
    assert {l["action"] for l in log} >= {"drop_duplicates", "impute_missing", "standardize_categories", "invalid_to_missing"}


def test_cleaning_rejects_unknown_action(sales):
    df, prof, roles, ctx = sales
    with pytest.raises(ValueError):
        clean_dataset(df, prof, infer_semantics(df, prof), ["format_hard_drive"])


# ----------------------------------------------------------------------------- explainability & AutoML
def test_shap_is_exact_for_linear_models_and_additive():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(150, 4)), columns=list("abcd"))
    m = LinearRegression().fit(X, X @ np.array([3.0, -2.0, 0.0, 1.0]) + 5)
    sv = shapley_values(m.predict, X.head(6), X, 8)
    exact = (X.head(6) - sv["background"].mean()).to_numpy() * m.coef_
    assert np.abs(sv["phi"] - exact).max() < 1e-8 and sv["additivity_gap"] < 1e-8


def test_one_standard_error_rule_prefers_simpler_model():
    rows = [{"model": "XGBoost Regressor", "cv_RMSE": 10.0, "cv_RMSE_std": 2.0}, {"model": "Linear Regression", "cv_RMSE": 10.5, "cv_RMSE_std": 2.0}]
    best, why = _one_se_select(rows, "cv_RMSE", "cv_RMSE_std", 5, higher_better=False)
    assert best["model"] == "Linear Regression" and "one-standard-error" in why
    rows[1]["cv_RMSE"] = 20.0
    assert _one_se_select(rows, "cv_RMSE", "cv_RMSE_std", 5, higher_better=False)[0]["model"] == "XGBoost Regressor"


def test_classification_reports_shap_rationale_and_temporal_check(delivery):
    df, prof, roles, ctx = delivery
    res = execute(validate_plan(AnalysisPlan(operation="predict_class", target_column="Status"), df, prof, roles), ctx)
    ev = res["evaluation"]
    assert ev["selection_rationale"] and ev["temporal_validation"]["metrics"]["F1"] > 0.4
    assert res["explainability"]["shap"]["items"] and abs(res["explainability"]["local"]["additivity_gap"]) < 1e-6
    assert any(c["type"] == "waterfall" for c in res["charts"])


def test_forecast_includes_statistical_models_and_95_band(retail):
    df, prof, roles, ctx = retail
    res = execute(validate_plan(AnalysisPlan(operation="forecast", target_column="Units_Sold", frequency="W", forecast_horizon=6, aggregation="sum"), df, prof, roles), ctx)
    names = [c["model"] for c in res["tables"]["candidate_models"]]
    assert any("ETS" in n for n in names) and any("ARIMA" in n for n in names)
    r0 = res["forecast"]["periods"][0]
    assert r0["lower_95"] <= r0["lower_80"] <= r0["forecast"] <= r0["upper_80"] <= r0["upper_95"]


def test_forecast_never_substitutes_an_invalid_target(retail):
    df, prof, roles, ctx = retail
    with pytest.raises(UDIError) as e:
        validate_plan(AnalysisPlan(operation="forecast", target_column="Footfall"), df, prof, roles)
    assert e.value.code == "INVALID_COLUMN"


# ----------------------------------------------------------------------------- business operations
def test_explain_change_decomposition_adds_up(sales):
    df, prof, roles, ctx = sales
    res = execute(validate_plan(AnalysisPlan(operation="explain_change", target_column="Sales", aggregation="sum", frequency="M", group_by=["Region"]), df, prof, roles), ctx)
    contrib = sum(r["contribution"] for r in res["table"])
    assert abs(contrib - res["metrics"]["Change"]) < 1e-6 * max(1, abs(res["metrics"]["Change"]))
    assert res["charts"][0]["type"] == "waterfall"


def test_pivot_query_segment_and_eda(sales):
    df, prof, roles, ctx = sales
    pv = execute(validate_plan(AnalysisPlan(operation="pivot_table", target_column="Sales", group_by=["Region"], pivot_column="Category", aggregation="sum"), df, prof, roles), ctx)
    assert abs(sum(r["row_total"] for r in pv["table"]) - df["Sales"].sum()) < 1e-6
    q = execute(validate_plan(AnalysisPlan(operation="query_records", target_column="Profit", sort_order="desc", top_n=5,
                                           filters=[{"column": "City", "operator": "eq", "value": "Goa"}]), df, prof, roles), ctx)
    assert len(q["table"]) == 5 and q["table"][0]["Profit"] == df[df.City == "Goa"]["Profit"].max()
    seg = execute(validate_plan(AnalysisPlan(operation="segment_records"), df, prof, roles), ctx)
    assert 2 <= seg["metrics"]["Segments"] <= 6 and sum(r["records"] for r in seg["table"]) == len(df)
    eda = execute(validate_plan(AnalysisPlan(operation="run_eda"), df, prof, roles), ctx)
    assert {c["type"] for c in eda["charts"]} >= {"bar", "box", "heatmap"} and eda["insights"]


def test_report_has_sections_recommendations_and_exports(sales):
    df, prof, roles, ctx = sales
    from report_engine import report_pdf, report_xlsx
    res = execute(validate_plan(AnalysisPlan(operation="generate_report", business_function="finance"), df, prof, roles), ctx)
    rep = res["report"]
    assert rep["sections"] and rep["recommendations"] and rep["kpis"] and rep["agent_trace"]
    assert any(k["label"] == "Profit margin" for k in rep["kpis"])
    assert report_pdf(rep)[:4] == b"%PDF" and report_xlsx(rep)[:2] == b"PK"


# ----------------------------------------------------------------------------- local planner
@pytest.mark.parametrize("q,op,target,group", [
    ("Which region has the highest sales?", "compare_groups", "Sales", ["Region"]),
    ("What is the average profit?", "get_summary", "Profit", None),
    ("Forecast sales for the next 3 months", "forecast", "Sales", None),
    ("Are there any outliers in cost?", "detect_anomalies", "Cost", None),
    ("Why did profit drop last month?", "explain_change", "Profit", None),
    ("Total sales by region and category", "pivot_table", "Sales", ["Region"]),
    ("Check the data quality", "assess_data_quality", None, None),
    ("How are the columns related?", "describe_relationships", None, None),
    ("Generate a finance report", "generate_report", None, None),
    ("What is the weather tomorrow?", "unsupported", None, None),
])
def test_local_planner_routes_questions(sales, q, op, target, group):
    df, prof, roles, ctx = sales
    plan, info = LocalPlanner().plan_question(q, df, prof, infer_semantics(df, prof))
    assert plan.operation == op, (q, info)
    if target:
        assert plan.target_column == target
    if group:
        assert plan.group_by == group


def test_local_planner_passes_unknown_variable_to_validator(sales):
    df, prof, roles, ctx = sales
    plan, _ = LocalPlanner().plan_question("What is the average bonus?", df, prof, infer_semantics(df, prof))
    with pytest.raises(UDIError) as e:
        validate_plan(plan, df, prof, roles)
    assert e.value.code == "INVALID_COLUMN"


# ----------------------------------------------------------------------------- API, persistence, security
def _client(tmp, **kw):
    app = create_app(llm=DownLLM(), planner_mode="local", data_dir=Path(tmp), **kw)
    return app.test_client(), app


def _upload(c, df, sid=None, headers=None):
    data = {"domain": "company", "file": (io.BytesIO(df.to_csv(index=False).encode()), "sales.csv")}
    if sid:
        data["session_id"] = sid
    return c.post("/api/upload", data=data, content_type="multipart/form-data", headers=headers or {})


def test_phase2_endpoints_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        c, _ = _client(tmp)
        df = sales_df(300)
        up = _upload(c, df).get_json()
        sid = up["session_id"]
        assert up["quality"]["score"] > 0 and up["suggestions"]
        d = c.get(f"/api/session/{sid}/dictionary").get_json()
        assert d["dictionary"] and d["heatmap"]["type"] == "heatmap"
        assert c.get(f"/api/session/{sid}/eda").get_json()["operation"] == "run_eda"
        q = c.get(f"/api/session/{sid}/quality").get_json()
        assert "issues" in q and q["actions"]["default"]
        cl = c.post(f"/api/session/{sid}/clean", json={"actions": ["drop_duplicates", "impute_missing"]}).get_json()
        assert "comparison" in cl
        assert c.post(f"/api/session/{sid}/revert").get_json()["reverted"]
        ans = c.post("/api/chat", json={"session_id": sid, "message": "Which city has the highest sales?"}).get_json()
        assert ans["planner"] == "local" and ans["plan"]["operation"] == "compare_groups"
        top = df.groupby("City")["Sales"].sum().idxmax()
        assert ans["result"]["metrics"]["Top group"] == top and ans["agent_trace"]
        job = c.post(f"/api/session/{sid}/report", json={"function": "sales"}).get_json()["job_id"]
        import time
        for _ in range(240):
            st = c.get(f"/api/jobs/{job}").get_json()
            if st["status"] in ("done", "error"):
                break
            time.sleep(0.5)
        assert st["status"] == "done", st
        assert c.get(f"/api/jobs/{job}/download?format=pdf").data[:4] == b"%PDF"
        assert c.get(f"/api/jobs/{job}/download?format=xlsx").data[:2] == b"PK"
        assert c.get(f"/api/session/{sid}/download").status_code == 200
        assert len(c.get(f"/api/session/{sid}/history").get_json()["routing"]) == 1
        assert c.get("/api/metrics").get_json()["operations"]["compare_groups"] == 1
        assert set(c.get("/api/health").get_json()["operations"]) == set(OPERATIONS)
        assert c.delete(f"/api/session/{sid}").get_json()["deleted"]
        assert c.get(f"/api/session/{sid}").status_code == 404


def test_session_survives_restart_via_storage():
    with tempfile.TemporaryDirectory() as tmp:
        c, _ = _client(tmp)
        sid = _upload(c, sales_df(200)).get_json()["session_id"]
        c.post("/api/chat", json={"session_id": sid, "message": "What is the average profit?"})
        c2, _ = _client(tmp)                                     # a fresh app = a server restart
        st = c2.get(f"/api/session/{sid}").get_json()
        assert st["has_dataset"] and st["profile"]["rows"] == 200 and st["turns"]


def test_auth_ownership_and_rate_limit():
    with tempfile.TemporaryDirectory() as tmp:
        c, _ = _client(tmp, auth_enabled=True)
        assert c.post("/api/session", json={}).status_code == 401
        a = c.post("/api/auth/register", json={"username": "alice", "password": "secret123"}).get_json()
        b = c.post("/api/auth/register", json={"username": "bob", "password": "secret456"}).get_json()
        assert a["user"]["role"] == "admin" and b["user"]["role"] == "analyst"
        ha, hb = {"Authorization": f"Bearer {a['token']}"}, {"Authorization": f"Bearer {b['token']}"}
        sid = _upload(c, sales_df(100), headers=ha).get_json()["session_id"]
        assert c.get(f"/api/session/{sid}", headers=hb).status_code == 403
        assert c.get(f"/api/session/{sid}", headers=ha).status_code == 200
        assert c.get("/api/metrics", headers=hb).status_code == 403
        assert c.post("/api/auth/login", json={"username": "alice", "password": "wrong"}).status_code == 401
    rl = RateLimiter({"chat": 2})
    rl.check("u", "chat"); rl.check("u", "chat")
    with pytest.raises(UDIError) as e:
        rl.check("u", "chat")
    assert e.value.code == "RATE_LIMITED"
