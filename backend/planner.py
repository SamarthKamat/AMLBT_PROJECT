"""Planning service (Phase 2): question -> structured plan, from Gemini or from the local intent model.

Modes (UDI_PLANNER):
  auto    (default) Gemini when it is configured; the trained local planner when Gemini is not configured or a
          Gemini call fails. The answer says which planner produced the plan.
  gemini  Gemini only (Phase 1 behaviour): no Gemini means a structured GEMINI_* error, nothing is guessed.
  local   the local planner only (fully offline; no data or schema leaves the machine).
Whichever planner is used, the plan passes through the same validation gate before anything is executed, and the
planner never produces numbers. Explanations: Gemini wording is accepted only if every number in it traces back to
the verified result; otherwise the Python-written summary is shown.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple

import config
from config import DOMAINS
from errors import UDIError
from explainer import check_grounded, digest
from local_planner import LocalPlanner
from schemas import AnalysisPlan
from session_manager import Session

log = logging.getLogger("udi.planner")
FALLBACK_CODES = {"GEMINI_UNAVAILABLE", "GEMINI_MODEL_UNAVAILABLE", "INVALID_PLAN"}


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
    def __init__(self, llm: Any, mode: Optional[str] = None, local: Optional[LocalPlanner] = None):
        self.llm = llm
        self.mode = (mode or config.PLANNER_MODE or "auto").lower()
        self.local = local or LocalPlanner()

    def gemini_ready(self) -> bool:
        try:
            return bool(self.llm.status(live=False).get("ready"))
        except Exception:
            return False

    def _local(self, question: str, session: Session) -> Tuple[AnalysisPlan, Dict[str, Any]]:
        sem = session.ctx.semantics if session.ctx else {}
        return self.local.plan_question(question, session.df, session.profile, sem, session.turns[-4:], session.pending_clarification,
                                        key=session.ctx.dataset_id if session.ctx else "")

    def plan(self, question: str, session: Session) -> Tuple[AnalysisPlan, str, Optional[str], Dict[str, Any]]:
        """(plan, planner_source, note, planner_info)."""
        if self.mode == "local" or (self.mode == "auto" and not self.gemini_ready()):
            p, info = self._local(question, session)
            note = None                                   # the answer footer always names the planner that was used
            return p, "local", note, info
        dom = DOMAINS.get(session.domain or "", {})
        try:
            p = self.llm.plan(question, session.llm_profile, dom.get("label", session.domain or "Unspecified"), dom.get("context", ""),
                              history_for_llm(session), session.pending_clarification)
            return p, "gemini", None, {}
        except UDIError as e:
            if self.mode != "auto" or e.code not in FALLBACK_CODES:
                raise
            log.warning("Gemini planning failed (%s); using the local planner", e.code)
            p, info = self._local(question, session)
            return p, "local", f"Gemini could not plan this question ({e.code}); UDI's local intent model planned it instead.", info


def explain_result(llm: Any, question: str, plan: AnalysisPlan, result: Dict[str, Any], use_llm: bool = True) -> Tuple[str, str, Optional[str]]:
    """(answer_text, source, note). Gemini wording is accepted only if every number in it traces back to the verified result."""
    if not use_llm:
        return result["summary"], "python_summary", None
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
