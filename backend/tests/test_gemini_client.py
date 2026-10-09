"""Gemini client behaviour with a mocked SDK client (no network). Live Gemini access is NOT exercised here."""
import json
import re
from pathlib import Path

import pytest
from google.genai import errors as ge

import gemini_client
from errors import UDIError
from gemini_client import GeminiService, PLAN_JSON_SCHEMA, map_gemini_error
from data_profiler import profile_dataframe, profile_for_llm
from schemas import AnalysisPlan
from tests import datasets

BACKEND = Path(__file__).resolve().parents[1]


class Resp:
    def __init__(self, text): self.text = text


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return Resp(o)

    def get(self, model):
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o

    def list(self):
        class M:
            def __init__(s, n): s.name, s.supported_actions = n, ["generateContent"]
        return [M("models/gemini-alpha"), M("models/gemini-beta")]


class FakeClient:
    def __init__(self, outcomes): self.models = FakeModels(outcomes)


def svc(outcomes, model="test-model", key="k"):
    c = FakeClient(outcomes)
    return GeminiService(api_key=key, model=model, client=c), c.models


PLAN = {"operation": "compare_groups", "target_column": "Delivery_Time", "group_by": ["City"], "aggregation": "mean", "reasoning": "r"}


def call_plan(s, q="q"):
    return s.plan(q, {"columns": []}, "Company", "ctx", [], None)


def test_valid_plan_parsed_and_request_shape():
    s, m = svc([json.dumps(PLAN)])
    plan = call_plan(s)
    assert isinstance(plan, AnalysisPlan) and plan.operation == "compare_groups"
    cfg = m.calls[0]["config"]
    assert m.calls[0]["model"] == "test-model"
    assert cfg.response_mime_type == "application/json" and cfg.temperature == 0
    assert cfg.response_json_schema == PLAN_JSON_SCHEMA


def test_fenced_and_wrapped_json_accepted():
    s, _ = svc(["```json\n" + json.dumps(PLAN) + "\n```"])
    assert call_plan(s).operation == "compare_groups"
    s, _ = svc(["Here is the plan: " + json.dumps(PLAN) + " Hope it helps"])
    assert call_plan(s).operation == "compare_groups"


def test_malformed_json_retried_once_then_invalid_plan():
    s, m = svc(["not json at all", json.dumps(PLAN)])
    assert call_plan(s).operation == "compare_groups" and len(m.calls) == 2
    assert "not a valid plan" in m.calls[1]["contents"]
    s, m = svc(["nope", "still nope"])
    with pytest.raises(UDIError) as e:
        call_plan(s)
    assert e.value.code == "INVALID_PLAN" and e.value.retryable and len(m.calls) == 2


def test_unknown_operation_from_gemini_is_not_accepted():
    s, _ = svc([json.dumps({**PLAN, "operation": "run_python"})] * 2)
    with pytest.raises(UDIError) as e:
        call_plan(s)
    assert e.value.code == "INVALID_PLAN"


def test_empty_response_is_an_error():
    s, _ = svc([""])
    with pytest.raises(UDIError) as e:
        call_plan(s)
    assert e.value.code == "INVALID_PLAN"


def test_schema_rejected_400_retries_without_schema():
    s, m = svc([ge.ClientError(400, {"error": {"message": "Unsupported schema keyword", "status": "INVALID_ARGUMENT"}}), json.dumps(PLAN)])
    assert call_plan(s).operation == "compare_groups"
    assert getattr(m.calls[0]["config"], "response_json_schema", None) and not getattr(m.calls[1]["config"], "response_json_schema", None)


@pytest.mark.parametrize("exc,code,retry", [
    (ge.ClientError(404, {"error": {"message": "models/test-model is not found for API version", "status": "NOT_FOUND"}}), "GEMINI_MODEL_UNAVAILABLE", False),
    (ge.ClientError(403, {"error": {"message": "Permission denied", "status": "PERMISSION_DENIED"}}), "GEMINI_UNAVAILABLE", False),
    (ge.ClientError(401, {"error": {"message": "bad", "status": "UNAUTHENTICATED"}}), "GEMINI_UNAVAILABLE", False),
    (ge.ClientError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}}), "GEMINI_UNAVAILABLE", True),
    (ge.ServerError(503, {"error": {"message": "overloaded", "status": "UNAVAILABLE"}}), "GEMINI_UNAVAILABLE", True),
    (TimeoutError("timed out"), "GEMINI_UNAVAILABLE", True),
    (ConnectionError("connection reset"), "GEMINI_UNAVAILABLE", True),
    (RuntimeError("weird"), "GEMINI_UNAVAILABLE", True),
])
def test_error_mapping(exc, code, retry):
    s, _ = svc([exc])
    with pytest.raises(UDIError) as e:
        call_plan(s)
    assert e.value.code == code and e.value.retryable is retry
    assert e.value.message and "Traceback" not in e.value.message


def test_model_error_names_the_model_and_the_fix():
    err = map_gemini_error(ge.ClientError(404, {"error": {"message": "models/old is not found", "status": "NOT_FOUND"}}), "old")
    assert "'old'" in err.message and "GEMINI_MODEL" in err.message


def test_missing_key_or_model_fails_cleanly():
    with pytest.raises(UDIError) as e:
        call_plan(GeminiService(api_key="", model="m"))
    assert e.value.code == "GEMINI_UNAVAILABLE" and "GEMINI_API_KEY" in e.value.message
    with pytest.raises(UDIError) as e:
        call_plan(GeminiService(api_key="k", model=""))
    assert e.value.code == "GEMINI_MODEL_UNAVAILABLE" and "GEMINI_MODEL" in e.value.message
    st = GeminiService(api_key="", model="m").status()
    assert st["ready"] is False and st["status"] == "no_api_key"
    assert GeminiService(api_key="k", model="").status()["status"] == "no_model"


def test_status_live_check_ok_and_model_unavailable():
    s, _ = svc([object()])
    assert s.status(live=True)["status"] == "ok"
    s, _ = svc([ge.ClientError(404, {"error": {"message": "models/test-model is not found", "status": "NOT_FOUND"}})])
    st = s.status(live=True)
    assert st["ready"] is False and st["status"] == "model_unavailable" and "gemini-alpha" in st["available_models"]
    s, _ = svc([ConnectionError("proxy blocked")])
    assert s.status(live=True)["status"] == "unreachable"
    assert svc([])[0].status(live=False)["status"] == "configured"


def test_explain_returns_text_and_sends_only_verified_digest():
    s, m = svc(["The average is 54.5 minutes."])
    out = s.explain("avg?", AnalysisPlan(operation="get_summary", target_column="Delivery_Time"), {"summary": "Average Delivery_Time is 54.5", "metrics": {"Average": 54.5}})
    assert out.startswith("The average")
    sent = json.loads(m.calls[0]["contents"])
    assert set(sent) == {"question", "plan", "verified_result"}
    s, _ = svc(["   "])
    with pytest.raises(UDIError):
        s.explain("q", AnalysisPlan(operation="get_summary"), {})


def test_dataset_rows_are_not_sent_to_gemini():
    df = datasets.delivery_df()
    prof = profile_dataframe(df, "d.csv")
    s, m = svc([json.dumps(PLAN)])
    s.plan("q", profile_for_llm(prof, df, 3), "Company", "ctx", [], None)
    sent = m.calls[0]["contents"]
    assert len(sent) < 25_000
    present = sum(1 for oid in df.Order_ID.astype(str) if oid in sent)
    assert present <= 3 + 1                                          # only the sample rows (3) can appear
    assert "100500" not in sent and "100321" not in sent
    s0, m0 = svc([json.dumps(PLAN)])
    s0.plan("q", profile_for_llm(prof, df, 0), "Company", "ctx", [], None)
    assert "sample_rows" not in m0.calls[0]["contents"] or '"sample_rows": []' in m0.calls[0]["contents"]
    assert not any(str(v) in m0.calls[0]["contents"] for v in df.Order_ID.head(5))


def test_prompts_forbid_invention_and_computation():
    sysmsg = gemini_client.PLAN_SYSTEM.lower()
    for phrase in ("never invent", "do not calculate", "do not write or run code", "unsupported"):
        assert phrase in sysmsg
    for op in ("get_summary", "get_current_state", "compare_groups", "analyze_trend", "detect_anomalies", "calculate_correlation", "predict_class", "predict_value", "forecast", "run_scenario"):
        assert op in gemini_client.PLAN_SYSTEM
    assert "only" in gemini_client.EXPLAIN_SYSTEM.lower()


def test_plan_schema_is_self_contained():
    txt = json.dumps(PLAN_JSON_SCHEMA)
    assert "$ref" not in txt and "$defs" not in txt
    ops = PLAN_JSON_SCHEMA["properties"]["operation"]["enum"]
    assert set(ops) >= {"get_summary", "run_scenario", "unsupported"}


def test_source_has_no_code_execution_or_keyword_routing():
    banned = [r"\bexec\(", r"\beval\(", r"(?<!re\.)\bcompile\(", r"subprocess", r"os\.system", r"rule_based", r"MOCK_MODE", r"UDI_MOCK", r"keyword_route", r"__import__"]
    for f in BACKEND.glob("*.py"):
        if f.name in ("train_models.py", "generate_samples.py"):
            continue
        src = f.read_text(encoding="utf8")
        for pat in banned:
            assert not re.search(pat, src), f"{f.name}: {pat}"


def test_no_hardcoded_model_or_key():
    src = (BACKEND / "config.py").read_text(encoding="utf8")
    assert not re.search(r"gemini-\d", src) and "AIza" not in src
    for f in BACKEND.rglob("*"):
        if f.is_file() and f.suffix in (".py", ".md", ".txt", ".example", ".bat", ".sh", ".json") and "node_modules" not in f.parts:
            assert not re.search(r"AIza[0-9A-Za-z_\-]{20,}", f.read_text(encoding="utf8", errors="ignore")), f
