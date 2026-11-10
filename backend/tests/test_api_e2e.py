"""End-to-end API tests: upload -> question -> plan -> validation -> operation -> grounded answer.

Gemini is replaced by ScriptedLLM (hand-written plans, as a correct Gemini would return). What is proven here is the
pipeline: different questions on ONE dataset are routed to different operations and computed from the data.
"""
import io
import json

import pandas as pd
import pytest

import config
from app import create_app
from schemas import OPERATIONS
from session_manager import SessionStore
from tests import datasets
from tests.fake_llm import DownLLM, ScriptedLLM

QUESTIONS = [
    ("Give me an overview of this dataset", dict(operation="get_summary", aggregation="count"), "get_summary"),
    ("What is the latest recorded delivery time?", dict(operation="get_current_state", target_column="Delivery_Time"), "get_current_state"),
    ("Compare delivery time across cities", dict(operation="compare_groups", target_column="Delivery_Time", group_by=["City"], aggregation="mean"), "compare_groups"),
    ("Is delivery time rising over time?", dict(operation="analyze_trend", target_column="Delivery_Time", time_column="Order_Date"), "analyze_trend"),
    ("Which orders look unusual?", dict(operation="detect_anomalies", target_column="Delivery_Time"), "detect_anomalies"),
    ("How does distance relate to delivery time?", dict(operation="calculate_correlation", target_column="Delivery_Time", feature_columns=["Distance"]), "calculate_correlation"),
    ("Predict the delivery time for a 12 km trip in heavy traffic", dict(operation="predict_value", target_column="Delivery_Time",
        input_values=[{"column": "Distance", "value": "12"}, {"column": "Traffic", "value": "High"}]), "predict_value"),
    ("Will an order be late?", dict(operation="predict_class", target_column="Status"), "predict_class"),
    ("Forecast delivery time for the next 14 days", dict(operation="forecast", target_column="Delivery_Time", time_column="Order_Date", frequency="D", forecast_horizon=14, aggregation="mean"), "forecast"),
    ("What if distances grow by 20%?", dict(operation="run_scenario", target_column="Delivery_Time", scenario_changes=[{"column": "Distance", "change_type": "percent", "value": 20}]), "run_scenario"),
    ("Total distance by vehicle type", dict(operation="compare_groups", target_column="Distance", group_by=["Vehicle_Type"], aggregation="sum"), "compare_groups"),
    ("Highest delivery time city", dict(operation="compare_groups", target_column="Delivery_Time", group_by=["City"], aggregation="mean", sort_order="desc", top_n=1), "compare_groups"),
    # Phase 2 operations
    ("Show the 5 slowest deliveries in Goa", dict(operation="query_records", target_column="Delivery_Time", sort_order="desc", top_n=5,
                                                   filters=[{"column": "City", "operator": "eq", "value": "Goa"}]), "query_records"),
    ("Average delivery time by city and traffic", dict(operation="pivot_table", target_column="Delivery_Time", group_by=["City"], pivot_column="Traffic", aggregation="mean"), "pivot_table"),
    ("Perform EDA", dict(operation="run_eda"), "run_eda"),
    ("Check data quality", dict(operation="assess_data_quality"), "assess_data_quality"),
    ("How are the columns related?", dict(operation="describe_relationships"), "describe_relationships"),
    ("Why did delivery time change last month?", dict(operation="explain_change", target_column="Delivery_Time", aggregation="mean", frequency="M"), "explain_change"),
    ("Segment the deliveries", dict(operation="segment_records"), "segment_records"),
    ("Generate an operations report", dict(operation="generate_report", business_function="operations"), "generate_report"),
]


@pytest.fixture(scope="module")
def csv_bytes():
    return datasets.delivery_df(anomalies=True).to_csv(index=False).encode()


def new_client(llm=None, store=None, planner_mode=None):
    app = create_app(llm=llm, store=store, planner_mode=planner_mode)
    return app.test_client(), app


def upload(client, content, name="delivery.csv", domain="company", session_id=None):
    data = {"domain": domain, "file": (io.BytesIO(content), name)}
    if session_id:
        data["session_id"] = session_id
    return client.post("/api/upload", data=data, content_type="multipart/form-data")


def ask(client, sid, q):
    return client.post("/api/chat", json={"session_id": sid, "message": q})


AVG = ("What is the average delivery time?", dict(operation="get_summary", target_column="Delivery_Time", aggregation="mean"))


@pytest.fixture()
def env(csv_bytes):
    llm = ScriptedLLM()
    for q, plan, _ in QUESTIONS:
        llm.add(q, **plan)
    llm.add(AVG[0], **AVG[1])
    client, app = new_client(llm)
    up = upload(client, csv_bytes)
    assert up.status_code == 200
    return client, llm, up.get_json()["session_id"], up.get_json()


def test_health_and_meta(env):
    client, *_ = env
    h = client.get("/api/health").get_json()
    assert h["ok"] and h["gemini"]["ready"] and set(h["operations"]) == set(OPERATIONS)
    assert len(client.get("/api/domains").get_json()) == 7
    assert len(client.get("/api/reference-models").get_json()) >= 7


def test_upload_returns_profile_and_preview(env):
    *_, up = env
    p = up["profile"]
    assert p["rows"] == 700 and {c["name"] for c in p["columns"]} >= {"Distance", "Delivery_Time", "Status"}
    assert len(up["preview"]) == 10 and up["domain"] == "company"


def test_twelve_different_questions_route_to_different_operations(env):
    client, llm, sid, _ = env
    used = []
    for q, _plan, expected_op in QUESTIONS:
        r = ask(client, sid, q)
        body = r.get_json()
        assert r.status_code == 200, (q, body)
        assert body["type"] == "answer" and body["plan"]["operation"] == expected_op, q
        assert body["result"]["success"] and body["result"]["operation"] == expected_op
        assert body["answer"] and body["answer_source"] in ("gemini", "python_summary")
        used.append(expected_op)
    assert set(used) == set(OPERATIONS)                                   # all eighteen operations exercised on one dataset
    route = client.get(f"/api/session/{sid}/routing").get_json()
    ops = [x["operation"] for x in route["routing"]]
    assert ops == used and len({*ops}) == len(OPERATIONS)
    ml = {x["operation"] for x in route["routing"] if x["ml"]}
    assert ml == {"predict_value", "predict_class", "forecast", "run_scenario", "segment_records", "generate_report"} | ({"detect_anomalies"} & ml)
    assert not any(x["ml"] for x in route["routing"] if x["operation"] in ("get_summary", "get_current_state", "compare_groups", "analyze_trend", "calculate_correlation"))
    assert route["model_cache"]["hits"] >= 1                              # the scenario reused the regression model trained for the prediction


def test_plan_for_two_questions_differs(env):
    client, llm, sid, _ = env
    a = ask(client, sid, QUESTIONS[1][0]).get_json()["plan"]
    b = ask(client, sid, QUESTIONS[2][0]).get_json()["plan"]
    assert a["operation"] != b["operation"]


def test_answers_come_from_computation(env):
    client, llm, sid, _ = env
    df = datasets.delivery_df(anomalies=True)
    avg = ask(client, sid, AVG[0]).get_json()
    assert avg["plan"]["operation"] == "get_summary"
    assert any(isinstance(v, (int, float)) and v == pytest.approx(df.Delivery_Time.mean(), rel=1e-6) for v in avg["result"]["metrics"].values()), avg["result"]["metrics"]
    latest = ask(client, sid, QUESTIONS[1][0]).get_json()
    assert latest["result"]["metrics"]["Latest Delivery_Time"] == df.sort_values("Order_Date").Delivery_Time.iloc[-1]
    top = ask(client, sid, "Highest delivery time city").get_json()
    expected_city = df.groupby("City").Delivery_Time.mean().idxmax()
    assert top["result"]["table"][0]["City"] == expected_city and expected_city in top["result"]["summary"]
    assert len(top["result"]["charts"][0]["data"]) == 1                                # top_n limits the chart; the table keeps the full comparison


def test_followup_context_is_passed_to_planner(env):
    client, llm, sid, _ = env
    ask(client, sid, "Compare delivery time across cities")
    llm.add("Only for Goa", operation="compare_groups", target_column="Delivery_Time", group_by=["Vehicle_Type"], aggregation="mean",
            filters=[{"column": "City", "operator": "eq", "value": "goa"}], uses_previous_context=True)
    r = ask(client, sid, "Only for Goa")
    body = r.get_json()
    assert r.status_code == 200 and body["plan"]["uses_previous_context"] and body["result"]["provenance"]["filters"] == ["City eq Goa"]   # value matched case-insensitively
    hist = llm.plan_calls[-1]["history"]
    assert hist and hist[-1]["question"] == "Compare delivery time across cities" and hist[-1]["plan"]["operation"] == "compare_groups"
    assert body["result"]["provenance"]["rows_analysed"] == (datasets.delivery_df().City == "Goa").sum()
    assert not any(c in json.dumps(llm.plan_calls[-1]["history"]) for c in ("Order_ID",))


def test_clarification_flow(env):
    client, llm, sid, _ = env
    llm.add("Which one is best?", operation="compare_groups", needs_clarification=True,
            clarification_question="Which metric should I rank by, and across which grouping?")
    r = ask(client, sid, "Which one is best?").get_json()
    assert r["type"] == "clarification" and "metric" in r["answer"] and r["result"] is None
    llm.add("Delivery time by city", operation="compare_groups", target_column="Delivery_Time", group_by=["City"], aggregation="mean")
    r2 = ask(client, sid, "Delivery time by city").get_json()
    assert r2["type"] == "answer"
    assert llm.plan_calls[-1]["pending"]["original_question"] == "Which one is best?"          # the pending clarification reached the planner
    r3 = ask(client, sid, AVG[0]).get_json()
    assert r3["type"] == "answer" and llm.plan_calls[-1]["pending"] is None                     # cleared after the answer


def test_missing_target_triggers_clarification_with_options(env):
    client, llm, sid, _ = env
    llm.add("Predict something", operation="predict_value")
    r = ask(client, sid, "Predict something").get_json()
    assert r["type"] == "clarification" and "Delivery_Time" in r["options"]


def test_invalid_column_rejected_with_available_columns(env):
    client, llm, sid, _ = env
    llm.add("Average profit", operation="get_current_state", target_column="Profit", aggregation="mean")
    r = ask(client, sid, "Average profit")
    body = r.get_json()
    assert r.status_code == 422 and body["code"] == "INVALID_COLUMN" and body["retryable"] is False
    assert "does not contain" in body["error"] and "Profit" in body["error"] and "Distance" in body["error"] and "Delivery_Time" in body["error"]
    assert "result" not in body
    # the rejection is logged as an error route, not silently substituted
    routes = client.get(f"/api/session/{sid}/routing").get_json()["routing"]
    assert routes[-1]["status"] == "error" and routes[-1]["error"] == "INVALID_COLUMN"


def test_unsupported_operation(env):
    client, llm, sid, _ = env
    llm.add("Write me a poem", operation="unsupported", reasoning="not a data question")
    r = ask(client, sid, "Write me a poem")
    assert r.status_code in (400, 422) and r.get_json()["code"] == "UNSUPPORTED_OPERATION"


def test_gemini_down_gives_structured_error_and_no_result(csv_bytes):
    """Strict Gemini mode (UDI_PLANNER=gemini) keeps the Phase 1 guarantee: no Gemini, no guessed plan."""
    client, _ = new_client(DownLLM(), planner_mode="gemini")
    sid = upload(client, csv_bytes).get_json()["session_id"]
    r = ask(client, sid, AVG[0])
    body = r.get_json()
    assert r.status_code >= 400 and body["code"] == "GEMINI_UNAVAILABLE" and "result" not in body and "answer" not in body
    h = client.get("/api/health").get_json()
    assert h["gemini_enabled"] is False and h["planner"]["mode"] == "gemini"
    # data upload and profiling still work without Gemini
    assert upload(client, csv_bytes).status_code == 200


def test_auto_mode_uses_local_planner_when_gemini_is_down(csv_bytes):
    """Default auto mode: without Gemini the trained local intent model plans the question; the result is still computed by Python."""
    client, _ = new_client(DownLLM(), planner_mode="auto")
    sid = upload(client, csv_bytes).get_json()["session_id"]
    body = ask(client, sid, AVG[0]).get_json()
    assert body["type"] == "answer" and body["planner"] == "local"
    assert body["plan"]["operation"] == "get_summary" and body["plan"]["target_column"] == "Delivery_Time" and body["plan"]["aggregation"] == "mean"
    expected = datasets.delivery_df(anomalies=True)["Delivery_Time"].mean()
    assert abs(body["result"]["metrics"]["Average of Delivery_Time"] - expected) < 1e-6
    assert client.get("/api/health").get_json()["planner"]["effective"] == "local"


def test_model_unavailable_code_passthrough(csv_bytes):
    client, _ = new_client(DownLLM("GEMINI_MODEL_UNAVAILABLE", "Model 'x' not found."), planner_mode="gemini")
    sid = upload(client, csv_bytes).get_json()["session_id"]
    body = ask(client, sid, "hello").get_json()
    assert body["code"] == "GEMINI_MODEL_UNAVAILABLE"


def test_explanation_fallbacks(csv_bytes):
    for mode, note in (("fail", "unavailable"), ("invent", "not in the verified result")):
        llm = ScriptedLLM(explain_mode=mode)
        llm.add("avg", operation="get_summary", target_column="Delivery_Time", aggregation="mean")
        client, _ = new_client(llm)
        sid = upload(client, csv_bytes).get_json()["session_id"]
        body = ask(client, sid, "avg").get_json()
        assert body["answer_source"] == "python_summary" and note in body["explanation_note"] and "987654" not in body["answer"]
        assert body["result"]["success"]


def test_new_chat_resets_everything(env):
    client, llm, sid, _ = env
    ask(client, sid, AVG[0])
    r = client.post("/api/new-chat", json={"session_id": sid, "domain": "hospital"}).get_json()
    new_sid = r["session_id"]
    assert new_sid != sid and r["reset"]
    assert client.get(f"/api/session/{sid}/routing").status_code == 404                       # old session gone
    body = ask(client, new_sid, "What is the average delivery time?")
    assert body.status_code == 400 and body.get_json()["code"] == "NO_DATASET"                # dataset must be uploaded again
    assert ask(client, sid, "x").get_json()["code"] == "SESSION_NOT_FOUND"


def test_reupload_resets_context_and_cache(env, csv_bytes):
    client, llm, sid, _ = env
    ask(client, sid, "Will an order be late?")
    assert client.get(f"/api/session/{sid}/routing").get_json()["model_cache"]["entries"] == 1
    up = upload(client, csv_bytes, session_id=sid).get_json()
    assert up["session_id"] == sid
    info = client.get(f"/api/session/{sid}/routing").get_json()
    assert info["routing"] == [] and info["model_cache"]["entries"] == 0


def test_sessions_are_isolated(csv_bytes):
    llm = ScriptedLLM()
    llm.add("avg", operation="get_summary", target_column="Salary", aggregation="mean")
    client, _ = new_client(llm)
    a = upload(client, csv_bytes).get_json()["session_id"]
    hr = datasets.hr_df().to_csv(index=False).encode()
    b = upload(client, hr, "hr.csv", "hospital").get_json()["session_id"]
    assert ask(client, a, "avg").get_json()["code"] == "INVALID_COLUMN"                       # Salary exists only in session b
    assert ask(client, b, "avg").get_json()["type"] == "answer"


def test_xlsx_upload_works(csv_bytes):
    buf = io.BytesIO()
    datasets.hr_df().to_excel(buf, index=False)
    llm = ScriptedLLM()
    llm.add("avg salary", operation="get_summary", target_column="Salary", aggregation="mean")
    client, _ = new_client(llm)
    r = upload(client, buf.getvalue(), "hr.xlsx", "company")
    assert r.status_code == 200 and r.get_json()["profile"]["rows"] == 400
    assert ask(client, r.get_json()["session_id"], "avg salary").get_json()["type"] == "answer"


def test_upload_errors(csv_bytes):
    client, _ = new_client(ScriptedLLM())
    assert upload(client, b"", "empty.csv").get_json()["code"] == "DATASET_PARSE_ERROR"
    assert upload(client, b"x", "notes.txt").get_json()["code"] == "DATASET_PARSE_ERROR"
    assert upload(client, b"\x00\x01garbage", "bad.xlsx").get_json()["code"] == "DATASET_PARSE_ERROR"
    assert upload(client, csv_bytes, domain="nonsense").get_json()["code"] == "BAD_REQUEST"
    assert client.post("/api/upload", data={"domain": "company"}, content_type="multipart/form-data").get_json()["code"] == "BAD_REQUEST"


def test_file_too_large(csv_bytes, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 1)
    client, app = new_client(ScriptedLLM())
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
    r = upload(client, b"a,b\n" + b"1,2\n" * 400_000, "big.csv")
    assert r.status_code == 413 and r.get_json()["code"] == "FILE_TOO_LARGE"


def test_chat_input_validation(env):
    client, _, sid, _ = env
    assert ask(client, sid, "   ").get_json()["code"] == "BAD_REQUEST"
    assert client.post("/api/chat", data="not json", content_type="text/plain").get_json()["code"] == "SESSION_NOT_FOUND"
    assert client.get("/api/nothing").status_code == 404


def test_responses_are_strict_json(env):
    client, _, sid, _ = env
    for q, *_ in QUESTIONS[:9]:
        raw = ask(client, sid, q).get_data(as_text=True)
        json.loads(raw, parse_constant=lambda c: pytest.fail(f"non-JSON constant {c} in response to {q}"))


def test_insufficient_data_error_from_api():
    df = datasets.delivery_df(20, anomalies=False)
    llm = ScriptedLLM()
    llm.add("late?", operation="predict_class", target_column="Status")
    client, _ = new_client(llm)
    sid = upload(client, df.to_csv(index=False).encode()).get_json()["session_id"]
    body = ask(client, sid, "late?").get_json()
    assert body["code"] == "INSUFFICIENT_DATA"


def test_session_store_eviction():
    store = SessionStore(ttl=0, max_sessions=2)
    a = store.create()
    import time; time.sleep(0.01)
    store.create()
    assert store.get(a.id) is None                      # ttl=0 -> expired on next create
    s = SessionStore(ttl=1000, max_sessions=2)
    ids = [s.create().id for _ in range(4)]
    assert sum(1 for i in ids if s.get(i)) <= 3
