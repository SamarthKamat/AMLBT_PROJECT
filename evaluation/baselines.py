"""Comparison planners for the benchmark (they plug into UDI exactly where Gemini does, so the validation gate and the
operations are identical and only the question -> plan step differs).

KeywordPlanner   a conventional rule-based router: a fixed keyword table chooses the operation, columns are linked
                 only when their literal name appears in the question, the first numeric column is the target and the
                 first categorical column the grouping. It represents the non-LLM approach Phase 1 argued against.
"""
from __future__ import annotations
import re
from typing import Any, Dict, List

from schemas import AnalysisPlan

RULES = [
    ("generate_report", r"\breport\b"), ("assess_data_quality", r"\bquality\b|\bmissing\b|\bduplicate|\bclean\b"),
    ("describe_relationships", r"\brelate\b|\brelationships?\b|\bcolumns? relate"), ("detect_anomalies", r"outlier|anomal|unusual|abnormal|suspicious|odd"),
    ("forecast", r"\bforecast|\bnext\b|\bproject\b|\bcoming\b"), ("explain_change", r"\bwhy\b"), ("segment_records", r"\bsegment|\bcluster"),
    ("run_eda", r"\beda\b|\bexplor"), ("run_scenario", r"\bwhat if\b|\bif\b.*\b(increase|cut|rise|drop)"), ("calculate_correlation", r"correlat|related|linked|associated|drives?\b"),
    ("predict_class", r"\bwill\b|\bpredict\b.*\b(risk|churn|attrition|subscri)|\bclassif"), ("predict_value", r"\bpredict|\bestimate"),
    ("analyze_trend", r"\btrend|over time|\bincreasing|\bdeclining|\brising|\bgoing up"), ("query_records", r"^(show|list|find)\b"),
    ("get_current_state", r"\blatest\b|\bcurrent\b|\bmost recent\b"), ("pivot_table", r"\bby\b \w+ and \w+"),
    ("compare_groups", r"\bwhich\b|\bcompare\b|\bby\b|\bper\b|\brank"), ("get_summary", r"\baverage\b|\bmean\b|\btotal\b|\bhow many\b|\bmedian\b|\bdistribution\b|\bsplit\b"),
]


class KeywordPlanner:
    name = "keyword-rule baseline"

    def status(self, live: bool = False) -> Dict[str, Any]:
        return {"configured": True, "model": self.name, "ready": True, "status": "ok", "detail": self.name}

    def plan(self, question, llm_profile, domain, domain_context, history, pending=None) -> AnalysisPlan:
        q = question.lower()
        op = next((o for o, pat in RULES if re.search(pat, q)), "unsupported")
        cols = llm_profile["columns"]
        def norm(n): return re.sub(r"[_\-]+", " ", n.lower())
        hits = sorted([c for c in cols if norm(c["name"]) in q or c["name"].lower() in q], key=lambda c: q.find(norm(c["name"])) if norm(c["name"]) in q else q.find(c["name"].lower()))
        nums = [c["name"] for c in hits if c["role"] == "numeric"]
        cats = [c["name"] for c in hits if c["role"] in ("categorical", "binary")]
        plan: Dict[str, Any] = {"operation": op}
        agg = "mean" if re.search(r"average|mean", q) else "sum" if "total" in q else "median" if "median" in q else ("count" if "how many" in q else None)
        if op in ("predict_class",):
            plan["target_column"] = (cats or nums or [None])[0]
        elif op in ("explain_change", "forecast", "analyze_trend", "predict_value", "run_scenario", "detect_anomalies", "calculate_correlation", "query_records", "get_current_state"):
            plan["target_column"] = (nums or [None])[0]
            if op in ("forecast", "analyze_trend", "explain_change") and nums:
                plan["aggregation"] = agg if agg in ("mean", "sum") else "mean"
        elif op in ("compare_groups", "pivot_table"):
            plan["target_column"] = nums[0] if nums else None
            plan["group_by"] = cats[:1] or None
            if op == "pivot_table" and len(cats) > 1:
                plan["pivot_column"] = cats[1]
            if nums:
                plan["aggregation"] = agg if agg in ("mean", "sum", "median") else "mean"
            plan["sort_order"] = "asc" if re.search(r"lowest|least|fewest|worst", q) else "desc"
        elif op == "get_summary":
            plan["target_column"] = (nums or cats or [None])[0]
            plan["aggregation"] = agg if plan["target_column"] in nums else ("distribution" if plan["target_column"] else "count")
        elif op == "generate_report":
            plan["business_function"] = next((f for f in ("marketing", "hr", "operations", "finance", "sales", "customer") if f in q), "general")
        return AnalysisPlan(**plan)

    def explain(self, question, plan, verified) -> str:
        return verified["summary"]
