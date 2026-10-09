"""TEST DOUBLE standing in for Gemini.

It returns hand-written plans (what a correct Gemini would return for each question) so that validation, routing and
execution can be tested offline and deterministically. It is used only by tests; the application never falls back to it.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from errors import UDIError
from schemas import AnalysisPlan


class ScriptedLLM:
    def __init__(self, plans: Optional[Dict[str, Dict[str, Any]]] = None, explain_mode: str = "summary"):
        self.plans = plans or {}
        self.explain_mode = explain_mode
        self.plan_calls: List[Dict[str, Any]] = []

    def add(self, question: str, **plan) -> None:
        self.plans[question] = plan

    def status(self, live: bool = False) -> Dict[str, Any]:
        return {"configured": True, "model": "scripted-test-double", "ready": True, "status": "ok", "detail": "test double"}

    def plan(self, question, llm_profile, domain, domain_context, history, pending=None) -> AnalysisPlan:
        self.plan_calls.append({"question": question, "history": history, "pending": pending, "profile": llm_profile, "domain": domain})
        if question not in self.plans:
            raise UDIError("INVALID_PLAN", f"No scripted plan for: {question}")
        return AnalysisPlan(**self.plans[question])

    def explain(self, question, plan, verified) -> str:
        if self.explain_mode == "fail":
            raise UDIError("GEMINI_UNAVAILABLE", "scripted outage", retryable=True)
        if self.explain_mode == "invent":
            return "The answer is 987654.321 according to my own estimate."
        return verified["summary"]


class DownLLM:
    """Gemini that is not configured / not reachable: planning must fail with a structured error, never guess."""
    def __init__(self, code: str = "GEMINI_UNAVAILABLE", message: str = "GEMINI_API_KEY is not set."):
        self.code, self.message = code, message

    def status(self, live: bool = False):
        return {"configured": False, "model": None, "ready": False, "status": "no_api_key", "detail": self.message}

    def plan(self, *a, **k):
        raise UDIError(self.code, self.message, retryable=False)

    def explain(self, *a, **k):
        raise UDIError(self.code, self.message, retryable=False)
