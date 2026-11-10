"""Plan validation: every claim in a Gemini plan is checked against the real dataset before anything runs.

* unknown columns are REJECTED (with the columns that do exist) - never silently replaced
* only case/spacing variants of an existing column name are canonicalised
* operations whose data requirements are not met raise INSUFFICIENT_DATA with the numbers
* missing-but-askable inputs (no target, no group column, ...) raise ClarificationNeeded
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from errors import UDIError
from filtering import apply_filters
from schemas import AnalysisPlan, InputValue, OPERATIONS

OP_LABEL = {
    "get_summary": "summary", "get_current_state": "current-state lookup", "compare_groups": "group comparison",
    "analyze_trend": "trend analysis", "detect_anomalies": "anomaly detection", "calculate_correlation": "correlation analysis",
    "predict_class": "classification", "predict_value": "prediction", "forecast": "forecast", "run_scenario": "scenario analysis",
    "query_records": "record lookup", "pivot_table": "pivot table", "run_eda": "exploratory analysis", "assess_data_quality": "data-quality check",
    "describe_relationships": "relationship analysis", "explain_change": "root-cause analysis", "segment_records": "segmentation",
    "generate_report": "business report",
}
NUMERIC_AGGS = {"mean", "sum", "median", "min", "max", "std"}
MIN_ROWS = {"regression": 30, "classification": 40, "anomaly": 20, "correlation": 20, "trend": 6}
MAX_CLASSES = 30
MIN_CLASS_COUNT = 5


class ClarificationNeeded(Exception):
    def __init__(self, question: str, options: Optional[List[str]] = None):
        super().__init__(question)
        self.question = question
        self.options = options or []


@dataclass
class ValidatedPlan:
    plan: AnalysisPlan
    data: pd.DataFrame
    time_column: Optional[str] = None
    filters_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    total_rows: int = 0


def roles_from_profile(profile: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {c["name"]: {"role": c["role"], "high_cardinality": c.get("high_cardinality", False), "reason": c.get("note", "")}
            for c in profile["columns"]}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().casefold()).strip("_")


def _is_numeric(df: pd.DataFrame, roles: Dict, c: str) -> bool:
    return (c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
            and roles.get(c, {}).get("role") in ("numeric", "binary"))


def numeric_columns(df: pd.DataFrame, roles: Dict) -> List[str]:
    return [c for c in df.columns if _is_numeric(df, roles, c)]


def categorical_columns(df: pd.DataFrame, roles: Dict) -> List[str]:
    return [c for c in df.columns if roles.get(c, {}).get("role") in ("categorical", "binary")]


def _join(items: List[str], limit: int = 12) -> str:
    items = list(items)[:limit]
    if not items:
        return "none"
    return ", ".join(items[:-1]) + f" and {items[-1]}" if len(items) > 1 else items[0]


def resolve_column(name: Optional[str], df: pd.DataFrame, roles: Dict, op: str, kind: str = "any") -> Optional[str]:
    """Canonicalise `name` to an existing column or raise INVALID_COLUMN. kind: numeric|categorical|datetime|any"""
    if name is None or str(name).strip() == "":
        return None
    if name in df.columns:
        return name
    matches = [c for c in df.columns if _slug(c) == _slug(name)]
    if len(matches) == 1:
        return matches[0]
    pool = {"numeric": numeric_columns(df, roles), "categorical": categorical_columns(df, roles),
            "datetime": [c for c in df.columns if roles.get(c, {}).get("role") == "datetime"]}.get(kind, list(df.columns))
    noun = {"numeric": "numerical", "categorical": "categorical", "datetime": "date/time"}.get(kind, "")
    msg = (f"I cannot perform this {OP_LABEL.get(op, 'analysis')} because the dataset does not contain a '{name}' variable. "
           f"Available {noun + ' ' if noun else ''}variables include {_join(pool)}.")
    raise UDIError("INVALID_COLUMN", msg, details={"requested": name, "available": pool, "operation": op})


def _require_kind(df, roles, col, op, kind, what):
    if kind == "numeric" and not _is_numeric(df, roles, col):
        raise UDIError("INVALID_PLAN",
                       f"'{col}' is not a numeric column, so it cannot be used as the {what} for {OP_LABEL.get(op, op)}. "
                       f"Numeric columns: {_join(numeric_columns(df, roles))}.", details={"column": col})
    if kind == "datetime" and roles.get(col, {}).get("role") != "datetime":
        raise UDIError("INVALID_PLAN", f"'{col}' is not a date/time column.", details={"column": col})


def _time_col(plan: AnalysisPlan, df, roles, profile, op) -> Optional[str]:
    if plan.time_column:
        c = resolve_column(plan.time_column, df, roles, op, "datetime")
        _require_kind(df, roles, c, op, "datetime", "time axis")
        return c
    return profile.get("primary_time_column")


def _resolve_list(cols, df, roles, op, kind="any") -> Optional[List[str]]:
    if not cols:
        return None
    out = []
    for c in cols:
        r = resolve_column(c, df, roles, op, kind)
        if r and r not in out:
            out.append(r)
    return out or None


def _need_target(plan: AnalysisPlan, df, roles, op: str, kind: str = "numeric") -> None:
    if not plan.target_column:
        opts = numeric_columns(df, roles) if kind == "numeric" else [c for c in df.columns if roles.get(c, {}).get("role") not in ("identifier", "text", "empty", "constant", "datetime")]
        raise ClarificationNeeded(plan.clarification_question or
                                  f"Which {'numeric ' if kind == 'numeric' else ''}column should the {OP_LABEL.get(op, 'analysis')} focus on?", opts[:15])


def validate_plan(plan: AnalysisPlan, df: pd.DataFrame, profile: Dict[str, Any],
                  roles: Optional[Dict[str, Dict[str, Any]]] = None) -> ValidatedPlan:
    roles = roles or roles_from_profile(profile)
    op = plan.operation
    if op == "unsupported" or op not in OPERATIONS:
        raise UDIError("UNSUPPORTED_OPERATION",
                       (plan.reasoning or "This question needs an analysis UDI does not support.") +
                       " UDI can summarise, list matching records, build pivot tables, look up the latest state, compare groups, analyse trends, "
                       "explain changes, detect anomalies, measure correlations and column relationships, run EDA and data-quality checks, "
                       "segment records, predict values/classes, forecast, run what-if scenarios and generate business reports on the uploaded data.",
                       details={"supported_operations": OPERATIONS})
    if plan.needs_clarification and plan.clarification_question:
        raise ClarificationNeeded(plan.clarification_question, [])

    warnings: List[str] = []
    assumptions: List[str] = []
    p = plan.model_copy(deep=True)

    # ---- canonicalise every referenced column (reject unknown ones) --------------------
    kind_for_target = {"predict_value": "numeric", "forecast": "numeric", "run_scenario": "numeric", "analyze_trend": "numeric"}.get(op, "any")
    if op == "detect_anomalies":
        kind_for_target = "numeric"
    p.target_column = resolve_column(p.target_column, df, roles, op, kind_for_target)
    p.group_by = _resolve_list(p.group_by, df, roles, op, "categorical" if op in ("compare_groups", "pivot_table", "explain_change") else "any")
    p.pivot_column = resolve_column(p.pivot_column, df, roles, op, "categorical")
    p.feature_columns = _resolve_list(p.feature_columns, df, roles, op)
    for f in p.filters or []:
        f.column = resolve_column(f.column, df, roles, op)
    for ch in p.scenario_changes or []:
        ch.column = resolve_column(ch.column, df, roles, op, "numeric")
    for iv in p.input_values or []:
        iv.column = resolve_column(iv.column, df, roles, op)

    t_col = _time_col(p, df, roles, profile, op)
    p.time_column = t_col if op in ("get_current_state", "analyze_trend", "forecast", "explain_change") or p.start_date or p.end_date else p.time_column

    # ---- operation-specific requirements ----------------------------------------------
    tgt = p.target_column
    if op == "get_summary":
        agg = p.aggregation
        if agg in NUMERIC_AGGS:
            if not tgt:
                raise ClarificationNeeded(p.clarification_question or f"Which numeric column should I take the {agg} of?", numeric_columns(df, roles)[:15])
            if not _is_numeric(df, roles, tgt) and not pd.api.types.is_datetime64_any_dtype(df[tgt]):
                raise UDIError("INVALID_PLAN", f"Cannot compute the {agg} of '{tgt}' because it is not numeric. "
                                               f"Numeric columns: {_join(numeric_columns(df, roles))}.", details={"column": tgt})
        if agg in ("distribution", "percentage"):
            if not tgt:
                raise ClarificationNeeded(p.clarification_question or "Which categorical column should I describe?", categorical_columns(df, roles)[:15])
            if roles.get(tgt, {}).get("role") not in ("categorical", "binary"):
                raise UDIError("INVALID_PLAN", f"A {agg} breakdown needs a categorical column; '{tgt}' is {roles.get(tgt, {}).get('role', 'not categorical')}. "
                                               f"Categorical columns: {_join(categorical_columns(df, roles))}.", details={"column": tgt})
    elif op == "get_current_state":
        if not t_col:
            raise UDIError("INSUFFICIENT_DATA", "The dataset has no date/time column, so the most recent observation cannot be determined.")
        if p.group_by and len(p.group_by) > 1:
            p.group_by = p.group_by[:1]
            warnings.append("Only the first grouping column was used for the latest-state lookup.")
    elif op == "compare_groups":
        if not p.group_by:
            raise ClarificationNeeded(p.clarification_question or "Which column should I compare groups of?", categorical_columns(df, roles)[:15])
        for g in p.group_by:
            if roles.get(g, {}).get("role") in ("identifier", "text", "empty", "constant", "datetime") and not (g in categorical_columns(df, roles)):
                raise UDIError("INVALID_PLAN", f"'{g}' cannot be used to form comparison groups ({roles.get(g, {}).get('role')}).", details={"column": g})
        if len(p.group_by) > 2:
            p.group_by = p.group_by[:2]
            warnings.append("Only the first two grouping columns were used.")
        if tgt and tgt in p.group_by:
            raise UDIError("INVALID_PLAN", f"'{tgt}' is both the measure and the grouping column.")
        if tgt and _is_numeric(df, roles, tgt) and p.aggregation in (None,):
            p.aggregation = "mean"
            assumptions.append(f"No aggregation was specified, so the average of '{tgt}' per group is used.")
        if tgt and not _is_numeric(df, roles, tgt) and p.aggregation in NUMERIC_AGGS:
            raise UDIError("INVALID_PLAN", f"Cannot take the {p.aggregation} of non-numeric '{tgt}'.")
    elif op == "analyze_trend":
        if not t_col:
            raise UDIError("INSUFFICIENT_DATA", "Trend analysis needs a date/time column and the dataset has none.")
        if tgt:
            _require_kind(df, roles, tgt, op, "numeric", "measure")
        if p.group_by:
            p.group_by = p.group_by[:1]
        if p.aggregation is None:
            p.aggregation = "mean" if tgt else "count"
            assumptions.append(f"No aggregation was specified, so each period uses the {'average of ' + tgt if tgt else 'record count'}.")
        elif p.aggregation not in NUMERIC_AGGS | {"count", "nunique"}:
            p.aggregation = "mean" if tgt else "count"
    elif op == "detect_anomalies":
        if p.anomaly_method is None:
            p.anomaly_method = "auto"
        if tgt:
            _require_kind(df, roles, tgt, op, "numeric", "target")
        elif not numeric_columns(df, roles):
            raise UDIError("INSUFFICIENT_DATA", "Anomaly detection needs at least one numeric column and the dataset has none.")
    elif op == "calculate_correlation":
        nums = numeric_columns(df, roles)
        if tgt is None and len(nums) < 2:
            raise UDIError("INSUFFICIENT_DATA", f"Correlation needs at least two numeric columns; the dataset has {len(nums)}.")
        if tgt is not None and roles.get(tgt, {}).get("role") in ("identifier", "text", "constant", "empty", "datetime"):
            raise UDIError("INVALID_PLAN", f"'{tgt}' ({roles.get(tgt, {}).get('role')}) cannot be analysed for correlation.")
        p.correlation_method = p.correlation_method or "auto"
    elif op in ("predict_value", "run_scenario"):
        if not tgt:
            raise ClarificationNeeded(p.clarification_question or
                                      ("Which numeric outcome should I estimate the effect on?" if op == "run_scenario" else "Which numeric column should I predict?"),
                                      numeric_columns(df, roles)[:15])
        _require_kind(df, roles, tgt, op, "numeric", "target")
        if roles.get(tgt, {}).get("role") in ("identifier", "constant"):
            raise UDIError("INVALID_PLAN", f"'{tgt}' is {roles[tgt]['role']} and cannot be predicted meaningfully.")
        if roles.get(tgt, {}).get("role") == "binary" and op == "predict_value":
            warnings.append(f"'{tgt}' has only two distinct values; predict_class may be more appropriate.")
        if op == "run_scenario":
            if not p.scenario_changes:
                raise ClarificationNeeded(p.clarification_question or "What change should I simulate (for example +10% on a numeric column)?", numeric_columns(df, roles)[:15])
            for ch in p.scenario_changes:
                _require_kind(df, roles, ch.column, op, "numeric", "scenario driver")
                if ch.column == tgt:
                    raise UDIError("INVALID_PLAN", f"The scenario changes the target '{tgt}' itself; choose a driver column instead.")
                if ch.change_type == "percent" and ch.value <= -100:
                    raise UDIError("INVALID_PLAN", "A percentage decrease of 100% or more is not meaningful.")
                if roles.get(ch.column, {}).get("role") in ("identifier", "constant"):
                    raise UDIError("INVALID_PLAN", f"'{ch.column}' ({roles[ch.column]['role']}) cannot be varied in a scenario.")
    elif op == "predict_class":
        _need_target(p, df, roles, op, "any")
        role = roles.get(tgt, {}).get("role")
        if role in ("identifier", "text", "datetime", "empty"):
            raise UDIError("INVALID_PLAN", f"'{tgt}' ({role}) cannot be used as a classification target.")
        # a constant column has one class: handled below as INSUFFICIENT_DATA (needs at least two classes)
    elif op == "forecast":
        if not t_col:
            raise UDIError("INSUFFICIENT_DATA", "Forecasting needs a date/time column and the dataset has none.")
        _need_target(p, df, roles, op, "numeric")
        _require_kind(df, roles, tgt, op, "numeric", "forecast target")
        if p.group_by:
            warnings.append("Forecasts are produced for the aggregate series. To forecast one segment, filter to it (e.g. 'only for <value>').")
            p.group_by = None
        if p.aggregation is None or p.aggregation not in NUMERIC_AGGS:
            p.aggregation = "mean"
            assumptions.append(f"No aggregation was specified, so repeated rows in a period are averaged.")

    elif op == "query_records":
        if tgt and not _is_numeric(df, roles, tgt) and not pd.api.types.is_datetime64_any_dtype(df[tgt]):
            assumptions.append(f"'{tgt}' is not numeric, so records are listed without sorting by it.")
        if p.top_n is None:
            p.top_n = 20
    elif op == "pivot_table":
        if not p.group_by:
            raise ClarificationNeeded(p.clarification_question or "Which column should form the rows of the pivot table?", categorical_columns(df, roles)[:15])
        if not p.pivot_column and len(p.group_by) < 2:
            raise ClarificationNeeded(p.clarification_question or f"Which second column should form the columns of the pivot table (rows: {p.group_by[0]})?",
                                      [c for c in categorical_columns(df, roles) if c != p.group_by[0]][:15])
        if not p.pivot_column:
            p.pivot_column = p.group_by[1]
        p.group_by = p.group_by[:1]
        if p.pivot_column == p.group_by[0]:
            raise UDIError("INVALID_PLAN", "The rows and columns of a pivot table must be different columns.")
        if tgt and not (roles.get(tgt, {}).get("role") == "binary" or (roles.get(tgt, {}).get("role") == "categorical" and df[tgt].nunique() == 2)):
            _require_kind(df, roles, tgt, op, "numeric", "measure")
        if p.aggregation not in NUMERIC_AGGS | {None, "count"}:
            p.aggregation = None
    elif op == "explain_change":
        if not t_col:
            raise UDIError("INSUFFICIENT_DATA", "Explaining a change over time needs a date/time column and the dataset has none.")
        if tgt:
            _require_kind(df, roles, tgt, op, "numeric", "measure")
        if p.aggregation not in (None, "sum", "mean", "count"):
            p.aggregation = None
    elif op == "segment_records":
        if len([c for c in numeric_columns(df, roles) if roles.get(c, {}).get("role") == "numeric"]) < 2:
            raise UDIError("INSUFFICIENT_DATA", "Segmentation needs at least two numeric columns.")
    elif op == "generate_report":
        p.business_function = p.business_function or "general"
    elif op in ("run_eda", "assess_data_quality", "describe_relationships"):
        pass

    # ---- filters / date ranges against the real values ----------------------------------
    data, filters_applied = apply_filters(df, p, t_col)
    if len(data) == 0:
        raise UDIError("INSUFFICIENT_DATA", "No rows remain after applying the requested filters" +
                       (f" ({'; '.join(filters_applied)})." if filters_applied else "."), details={"filters": filters_applied})

    # ---- data sufficiency ---------------------------------------------------------------
    _check_sufficiency(p, data, roles, t_col)

    # input values for what-if/prediction rows
    for iv in p.input_values or []:
        _check_input_value(iv, df, roles)

    return ValidatedPlan(plan=p, data=data, time_column=t_col, filters_applied=filters_applied,
                         warnings=warnings, assumptions=assumptions, total_rows=len(df))


def _check_input_value(iv: InputValue, df: pd.DataFrame, roles: Dict) -> None:
    s = df[iv.column]
    if pd.api.types.is_numeric_dtype(s):
        try:
            float(str(iv.value).replace(",", ""))
        except ValueError:
            raise UDIError("INVALID_PLAN", f"Input value '{iv.value}' for numeric column '{iv.column}' is not a number.")
    elif not pd.api.types.is_datetime64_any_dtype(s):
        lookup = {str(v).casefold(): str(v) for v in s.dropna().unique()}
        if str(iv.value).casefold() not in lookup:
            raise UDIError("INVALID_PLAN", f"Value '{iv.value}' does not occur in column '{iv.column}'. Values include: {', '.join(list(lookup.values())[:12])}.")
        iv.value = lookup[str(iv.value).casefold()]


def _check_sufficiency(p: AnalysisPlan, data: pd.DataFrame, roles: Dict, t_col: Optional[str]) -> None:
    op, tgt = p.operation, p.target_column
    n = len(data)
    if op in ("predict_value", "run_scenario"):
        labeled = int(data[tgt].notna().sum())
        if labeled < MIN_ROWS["regression"]:
            raise UDIError("INSUFFICIENT_DATA", f"Predicting '{tgt}' needs at least {MIN_ROWS['regression']} rows with a value; only {labeled} are available" +
                           (" after filtering." if (p.filters or p.start_date or p.end_date) else "."), details={"labeled_rows": labeled})
        if data[tgt].nunique(dropna=True) < 3 and roles.get(tgt, {}).get("role") != "binary":
            raise UDIError("INSUFFICIENT_DATA", f"'{tgt}' has too little variation to model (fewer than 3 distinct values).")
    elif op == "predict_class":
        y = data[tgt].dropna()
        k = int(y.nunique())
        if k < 2:
            raise UDIError("INSUFFICIENT_DATA", f"Classification needs at least two classes in '{tgt}'; the data has {k}.", details={"classes": k})
        if k > MAX_CLASSES:
            raise UDIError("INVALID_PLAN", f"'{tgt}' has {k} distinct values, which looks continuous rather than a class label. "
                                           f"Use a prediction (regression) for numeric outcomes, or choose a column with at most {MAX_CLASSES} classes.", details={"classes": k})
        if len(y) < MIN_ROWS["classification"]:
            raise UDIError("INSUFFICIENT_DATA", f"Classification needs at least {MIN_ROWS['classification']} labelled rows; only {len(y)} are available.")
        counts = y.astype(str).value_counts()
        small = counts[counts < MIN_CLASS_COUNT]
        if len(small):
            raise UDIError("INSUFFICIENT_DATA",
                           f"Class(es) {', '.join(map(str, small.index[:5]))} of '{tgt}' have fewer than {MIN_CLASS_COUNT} observations "
                           f"({', '.join(f'{k}={v}' for k, v in small.head(5).items())}), too few to train and evaluate a classifier.",
                           details={"class_counts": counts.to_dict()})
    elif op == "detect_anomalies" and tgt:
        if int(data[tgt].notna().sum()) < MIN_ROWS["anomaly"]:
            raise UDIError("INSUFFICIENT_DATA", f"Anomaly detection needs at least {MIN_ROWS['anomaly']} values of '{tgt}'; only {int(data[tgt].notna().sum())} are available.")
    elif op == "detect_anomalies" and n < MIN_ROWS["anomaly"]:
        raise UDIError("INSUFFICIENT_DATA", f"Anomaly detection needs at least {MIN_ROWS['anomaly']} rows; only {n} are available.")
    elif op == "calculate_correlation" and n < MIN_ROWS["correlation"]:
        raise UDIError("INSUFFICIENT_DATA", f"Correlation needs at least {MIN_ROWS['correlation']} rows; only {n} are available.")
    elif op in ("analyze_trend", "forecast") and t_col:
        k = int(data[t_col].notna().sum())
        if k < MIN_ROWS["trend"]:
            raise UDIError("INSUFFICIENT_DATA", f"Only {k} rows have a valid '{t_col}' value; at least {MIN_ROWS['trend']} are needed.")
