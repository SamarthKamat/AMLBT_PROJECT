"""Guards that keep Gemini's wording tied to verified Python results.

* digest(): the only result content Gemini ever sees (compact, already verified).
* check_grounded(): rejects an explanation that contains a number not present in the verified result.
"""
from __future__ import annotations
import re
from typing import Any, Dict, Iterable, List, Set

_NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")


def _numbers_in_text(text: str) -> List[float]:
    out = []
    for m in _NUM.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _collect(obj: Any, acc: Set[float]) -> None:
    if obj is None or isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        acc.add(float(obj))
    elif isinstance(obj, str):
        acc.update(_numbers_in_text(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _collect(k, acc)
            _collect(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect(v, acc)


def digest(result: Dict[str, Any], max_rows: int = 6) -> Dict[str, Any]:
    ev = result.get("evaluation") or {}
    out = {
        "operation": result.get("operation"), "title": result.get("title"), "summary": result.get("summary"),
        "metrics": result.get("metrics"), "table_title": result.get("table_title"), "table_rows": (result.get("table") or [])[:max_rows],
        "limitations": (result.get("limitations") or [])[:4], "assumptions": (result.get("assumptions") or [])[:3],
        "provenance": {k: v for k, v in (result.get("provenance") or {}).items() if k in ("rows_analysed", "filters", "target", "group_by", "aggregation")},
    }
    if ev:
        out["model"] = {"selected": ev.get("selected_model"), "metrics": ev.get("metrics"), "test_rows": ev.get("test_rows"), "baseline": ev.get("baseline")}
    if result.get("explainability"):
        out["top_features"] = [{"feature": i["feature"], "share_%": round(i["share_%"], 1)} for i in result["explainability"]["items"][:5]]
    if result.get("prediction"):
        out["prediction"] = result["prediction"]
    if result.get("scenario"):
        out["scenario"] = result["scenario"]
    if result.get("forecast"):
        f = result["forecast"]
        out["forecast"] = {"horizon": f["horizon"], "frequency": f["frequency"], "periods": f["periods"][:6]}
    return out


def check_grounded(text: str, allowed_sources: Iterable[Any]) -> List[float]:
    """Return numbers in `text` that cannot be traced to the verified result (empty list = grounded)."""
    allowed: Set[float] = set()
    for src in allowed_sources:
        _collect(src, allowed)
    bad: List[float] = []
    for n in _numbers_in_text(text):
        if n == int(n) and 0 <= abs(n) <= 10:        # small counts/ordinals ("2 models", "top 3")
            continue
        if _matches(n, allowed):
            continue
        bad.append(n)
    return bad


def _matches(n: float, allowed: Set[float]) -> bool:
    for a in allowed:
        for cand in (a, a * 100, a / 100):
            if abs(n - cand) <= max(0.006 * abs(cand), 0.0051):
                return True
            for k in range(0, 4):
                if round(cand, k) == n:
                    return True
    return False
