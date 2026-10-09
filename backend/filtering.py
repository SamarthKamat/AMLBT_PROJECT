"""Row filters requested in a plan (e.g. a follow-up like "only for <value>").

Values are checked against the real data: a filter value that does not exist in a text column
is rejected, never guessed.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from errors import UDIError
from schemas import AnalysisPlan, Filter


def _norm(text: Any) -> str:
    return str(text).strip().casefold()


def _coerce_scalar(series: pd.Series, raw: Optional[str], column: str):
    if raw is None:
        raise UDIError("INVALID_PLAN", f"Filter on '{column}' is missing a value.")
    if pd.api.types.is_bool_dtype(series):
        return _norm(raw) in ("true", "1", "yes")
    if pd.api.types.is_numeric_dtype(series):
        try:
            return float(str(raw).replace(",", ""))
        except ValueError:
            raise UDIError("INVALID_PLAN", f"Filter value '{raw}' is not numeric but column '{column}' is numeric.")
    if pd.api.types.is_datetime64_any_dtype(series):
        v = pd.to_datetime(raw, errors="coerce")
        if pd.isna(v):
            raise UDIError("INVALID_PLAN", f"Filter value '{raw}' is not a valid date for column '{column}'.")
        return v
    return str(raw)


def _canonical_text_values(series: pd.Series, wanted: List[str], column: str) -> List[str]:
    lookup = {_norm(v): v for v in series.dropna().astype(str).unique()}
    out, missing = [], []
    for w in wanted:
        if _norm(w) in lookup:
            out.append(lookup[_norm(w)])
        else:
            missing.append(w)
    if missing:
        avail = sorted(lookup.values())[:15]
        raise UDIError(
            "INVALID_PLAN",
            f"Value(s) {missing} not found in column '{column}'. Available values include: {', '.join(avail)}.",
            details={"column": column, "available_values": avail},
        )
    return out


def filter_mask(df: pd.DataFrame, f: Filter) -> pd.Series:
    col = f.column
    s = df[col]
    op = f.operator
    is_text = not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_bool_dtype(s))
    if op in ("in", "not_in"):
        vals = f.values or ([f.value] if f.value is not None else [])
        if not vals:
            raise UDIError("INVALID_PLAN", f"Filter on '{col}' has no values.")
        if is_text:
            canon = _canonical_text_values(s, vals, col)
            m = s.astype(str).isin(canon)
        else:
            m = s.isin([_coerce_scalar(s, v, col) for v in vals])
        return m if op == "in" else ~m
    if op == "contains":
        if f.value is None:
            raise UDIError("INVALID_PLAN", f"Filter on '{col}' is missing a value.")
        return s.astype(str).str.contains(str(f.value), case=False, regex=False, na=False)
    if op == "between":
        lo, hi = _coerce_scalar(s, f.value, col), _coerce_scalar(s, f.value2, col)
        return (s >= lo) & (s <= hi)
    if is_text:
        if op not in ("eq", "ne"):
            raise UDIError("INVALID_PLAN", f"Operator '{op}' is not valid for text column '{col}'.")
        canon = _canonical_text_values(s, [f.value if f.value is not None else ""], col)[0]
        m = s.astype(str) == canon
        return m if op == "eq" else ~m
    v = _coerce_scalar(s, f.value, col)
    return {"eq": s == v, "ne": s != v, "gt": s > v, "gte": s >= v, "lt": s < v, "lte": s <= v}[op]


def apply_filters(df: pd.DataFrame, plan: AnalysisPlan, time_column: Optional[str]) -> (pd.DataFrame, List[str]):
    """Return (filtered_df, human_readable_filters)."""
    mask = pd.Series(True, index=df.index)
    described: List[str] = []
    for f in plan.filters or []:
        mask &= filter_mask(df, f).fillna(False)
        shown = f.values if f.values else (f"{f.value}–{f.value2}" if f.operator == "between" else f.value)
        col = df[f.column]
        if f.operator in ("eq", "ne", "in", "not_in") and not (pd.api.types.is_numeric_dtype(col) or pd.api.types.is_datetime64_any_dtype(col) or pd.api.types.is_bool_dtype(col)):
            shown = _canonical_text_values(col, f.values or [f.value], f.column)       # show the value as it appears in the data
            shown = shown if (f.values or len(shown) > 1) else shown[0]
        described.append(f"{f.column} {f.operator} {shown}")
    if plan.start_date or plan.end_date:
        if not time_column:
            raise UDIError("INVALID_PLAN", "A date range was requested but the dataset has no date/time column.")
        d = df[time_column]
        if plan.start_date:
            lo = pd.to_datetime(plan.start_date, errors="coerce")
            if pd.isna(lo):
                raise UDIError("INVALID_PLAN", f"Start date '{plan.start_date}' is not a valid date.")
            mask &= d >= lo
            described.append(f"{time_column} ≥ {plan.start_date}")
        if plan.end_date:
            hi = pd.to_datetime(plan.end_date, errors="coerce")
            if pd.isna(hi):
                raise UDIError("INVALID_PLAN", f"End date '{plan.end_date}' is not a valid date.")
            # a bare date means "through the end of that day"
            if len(str(plan.end_date).strip()) <= 10:
                hi = hi + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            mask &= d <= hi
            described.append(f"{time_column} ≤ {plan.end_date}")
    return df.loc[mask.to_numpy()], described


def filters_signature(plan: AnalysisPlan) -> str:
    parts = [f"{f.column}|{f.operator}|{f.value}|{f.value2}|{','.join(f.values or [])}" for f in (plan.filters or [])]
    parts.append(f"{plan.start_date}|{plan.end_date}")
    return ";".join(sorted(parts))
