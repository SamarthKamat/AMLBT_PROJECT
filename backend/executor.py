"""Controlled operation library. Gemini can only pick a name from this table; it can never run code."""
from __future__ import annotations
import time
from typing import Any, Callable, Dict

from analytics_engine import compare_groups, get_current_state, get_summary
from anomaly_engine import detect_anomalies
from context import ExecContext
from diagnostics_engine import analyze_trend, calculate_correlation
from errors import UDIError
from forecasting_engine import forecast
from ml_engine import predict_class, predict_value, run_scenario
from result_formatter import clean_result
from schemas import OPERATIONS
from validator import ValidatedPlan

OPERATION_TABLE: Dict[str, Callable[[ValidatedPlan, ExecContext], Dict[str, Any]]] = {
    "get_summary": get_summary,
    "get_current_state": get_current_state,
    "compare_groups": compare_groups,
    "analyze_trend": analyze_trend,
    "detect_anomalies": detect_anomalies,
    "calculate_correlation": calculate_correlation,
    "predict_class": predict_class,
    "predict_value": predict_value,
    "forecast": forecast,
    "run_scenario": run_scenario,
}
assert set(OPERATION_TABLE) == set(OPERATIONS), "operation table out of sync with schema"

# operations that train machine-learning models; every other operation is deterministic pandas/scipy
ML_OPERATIONS = {"predict_class", "predict_value", "run_scenario", "forecast"}


def execute(vp: ValidatedPlan, ctx: ExecContext) -> Dict[str, Any]:
    op = vp.plan.operation
    fn = OPERATION_TABLE.get(op)
    if fn is None:
        raise UDIError("UNSUPPORTED_OPERATION", f"Operation '{op}' is not available.")
    started = time.perf_counter()
    try:
        result = fn(vp, ctx)
    except UDIError:
        raise
    except Exception as exc:
        raise UDIError("BACKEND_ERROR", f"The {op} analysis failed unexpectedly: {type(exc).__name__}: {exc}")
    result["warnings"] = list(dict.fromkeys(list(result.get("warnings") or []) + vp.warnings))
    result["assumptions"] = list(dict.fromkeys(list(result.get("assumptions") or []) + vp.assumptions))
    p = vp.plan
    result["provenance"] = {
        "rows_in_dataset": vp.total_rows, "rows_analysed": len(vp.data), "filters": vp.filters_applied,
        "target": p.target_column, "group_by": p.group_by, "time_column": vp.time_column if op in ("get_current_state", "analyze_trend", "forecast") else None,
        "aggregation": p.aggregation, "uses_machine_learning": op in ML_OPERATIONS,
        "seconds": round(time.perf_counter() - started, 3),
    }
    return clean_result(result)
