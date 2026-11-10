"""Result object helpers and JSON sanitising.

Result contract (every operation returns this shape; unused parts are empty):
  success, operation, title, summary (Python-written, verified), metrics{label: scalar},
  table[], table_title, tables{name: rows}, charts[], evaluation{}, explainability{},
  forecast{}, scenario{}, prediction{}, limitations[], assumptions[], warnings[], provenance{}
"""
from __future__ import annotations
import datetime as _dt
import math
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def jsonsafe(obj: Any) -> Any:
    """Recursively convert numpy/pandas objects to plain JSON types (NaN/inf -> null)."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (pd.Timestamp, _dt.datetime, _dt.date)):
        if pd.isna(obj):
            return None
        return obj.isoformat() if not isinstance(obj, pd.Timestamp) else (obj.strftime("%Y-%m-%d") if (obj.hour, obj.minute, obj.second) == (0, 0, 0) else obj.isoformat(sep=" "))
    if isinstance(obj, (pd.Timedelta, np.timedelta64)):
        return str(obj)
    if isinstance(obj, pd.Period):
        return str(obj)
    if obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, dict):
        return {str(k): jsonsafe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonsafe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonsafe(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return [jsonsafe(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return records(obj)
    return str(obj)


def records(df: pd.DataFrame, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if limit is not None:
        df = df.head(limit)
    return [{str(k): jsonsafe(v) for k, v in row.items()} for row in df.to_dict("records")]


def make_result(operation: str, summary: str, **kw) -> Dict[str, Any]:
    res: Dict[str, Any] = {
        "success": True, "operation": operation, "title": kw.pop("title", None), "summary": summary,
        "metrics": {}, "table": [], "table_title": None, "tables": {}, "charts": [],
        "evaluation": None, "explainability": None, "forecast": None, "scenario": None, "prediction": None,
        "limitations": [], "assumptions": [], "warnings": [], "provenance": {}, "insights": [],
    }
    res.update(kw)
    return res


def fmt_num(v: Any, digits: int = 2) -> str:
    """Compact, locale-neutral number formatting used in Python-written summaries."""
    if v is None:
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f) or math.isinf(f):
        return "n/a"
    if f == round(f) and abs(f) < 1e15:
        return f"{f:,.0f}"
    if abs(f) < 0.01:
        return f"{f:.4g}"
    return f"{f:,.{digits}f}"


def downsample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    idx = np.linspace(0, len(df) - 1, n).round().astype(int)
    return df.iloc[np.unique(idx)]


def clean_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Drop non-serialisable internals (fitted models) and make everything JSON-safe."""
    out = {k: v for k, v in result.items() if not k.startswith("_")}
    return jsonsafe(out)
