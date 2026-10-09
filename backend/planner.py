"""Orchestration around Gemini: build the conversational context, request a plan, explain a verified result."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple

from config import DOMAINS
from errors import UDIError
from explainer import check_grounded, digest
from schemas import AnalysisPlan
from session_manager import Session

log = logging.getLogger("udi.planner")


def compact_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in plan.items() if v not in (None, [], "", False) and k not in ("reasoning", "intent")}


def history_for_llm(session: Session) -> List[Dict[str, Any]]:
    out = []
    for t in session.turns[-4:]:
        item: Dict[str, Any] = {"question": t["question"], "status": t["status"], "plan": compact_plan(t.get("plan") or {})}
        if t["status"] == "ok":
            d = t.get("digest") or {}
            item["result"] = {"summary": d.get("summary"), "metrics": {k: v for k, v in list((d.get("metrics") or {}).items())[:6]}}
        elif t["status"] == "clarification":
            item["clarification_asked"] = t.get("clarification")
        else:
            item["error"] = t.get("error")
        out.append(item)
    return out


class PlanningService:
    def __init__(self, llm: Any):
        self.llm = llm

    def plan(self, question: str, session: Session) -> AnalysisPlan:
        dom = DOMAINS.get(session.domain or "", {})
        return self.llm.plan(question, session.llm_profile, dom.get("label", session.domain or "Unspecified"), dom.get("context", ""),
                             history_for_llm(session), session.pending_clarification)


def explain_result(llm: Any, question: str, plan: AnalysisPlan, result: Dict[str, Any]) -> Tuple[str, str, Optional[str]]:
    """(answer_text, source, note). Gemini wording is accepted only if every number in it traces back to the verified result.

    If Gemini is unreachable or its wording is not grounded, the Python-written summary (which is always computed from the
    data) is returned instead and the note says so. Nothing is invented in either case.
    """
    verified = digest(result)
    try:
        text = llm.explain(question, plan, verified)
    except UDIError as exc:
        log.warning("Explanation unavailable (%s): %s", exc.code, exc.message)
        return result["summary"], "python_summary", f"Gemini explanation unavailable ({exc.code}); showing the verified Python summary."
    bad = check_grounded(text, [verified, plan.model_dump(), question])
    if bad:
        log.warning("Gemini explanation rejected: ungrounded numbers %s", bad[:5])
        return result["summary"], "python_summary", "Gemini's wording contained numbers that are not in the verified result, so the verified Python summary is shown instead."
    return text, "gemini", None
