"""The structured analysis plan.

Gemini fills this in; Python validates it against the real dataset before anything runs.
Gemini never returns results, only intent + column references.
"""
from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

# Phase 1 operations (10) + Phase 2 operations (8). Gemini / the local planner may only choose from this list.
OPERATIONS = [
    "get_summary", "get_current_state", "compare_groups", "analyze_trend", "detect_anomalies",
    "calculate_correlation", "predict_class", "predict_value", "forecast", "run_scenario",
    "query_records", "pivot_table", "run_eda", "assess_data_quality", "describe_relationships",
    "explain_change", "segment_records", "generate_report",
]
Operation = Literal[
    "get_summary", "get_current_state", "compare_groups", "analyze_trend", "detect_anomalies",
    "calculate_correlation", "predict_class", "predict_value", "forecast", "run_scenario",
    "query_records", "pivot_table", "run_eda", "assess_data_quality", "describe_relationships",
    "explain_change", "segment_records", "generate_report", "unsupported",
]
BusinessFunction = Literal["general", "sales", "marketing", "hr", "operations", "finance", "customer"]
ChartType = Literal["auto", "bar", "line", "pie", "scatter", "heatmap", "box", "histogram", "area", "table"]
Aggregation = Literal["mean", "sum", "median", "min", "max", "std", "count", "nunique", "distribution", "percentage"]
Frequency = Literal["H", "D", "W", "M", "Q", "Y"]
FilterOp = Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "between", "contains"]


class Filter(BaseModel):
    model_config = ConfigDict(extra="ignore")
    column: str
    operator: FilterOp = "eq"
    value: Optional[str] = Field(default=None, description="Value as text (numbers/dates too). For 'between' this is the lower bound.")
    value2: Optional[str] = Field(default=None, description="Upper bound for 'between'.")
    values: Optional[List[str]] = Field(default=None, description="Values for 'in' / 'not_in'.")


class ScenarioChange(BaseModel):
    model_config = ConfigDict(extra="ignore")
    column: str
    change_type: Literal["percent", "absolute", "set"] = "percent"
    value: float = Field(description="percent: +10 means +10%; absolute: amount added; set: new constant value.")


class InputValue(BaseModel):
    model_config = ConfigDict(extra="ignore")
    column: str
    value: str


class AnalysisPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intent: str = Field(default="", description="One-line statement of what the user wants.")
    operation: Operation
    needs_clarification: bool = False
    clarification_question: Optional[str] = None

    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    group_by: Optional[List[str]] = None
    time_column: Optional[str] = None
    aggregation: Optional[Aggregation] = None
    sort_order: Optional[Literal["asc", "desc"]] = None
    top_n: Optional[int] = Field(default=None, ge=1, le=200)

    frequency: Optional[Frequency] = None
    forecast_horizon: Optional[int] = Field(default=None, ge=1, le=366, description="Number of periods (of `frequency`) to forecast.")

    model_type: Optional[Literal["regression", "classification", "time_series", "anomaly", "correlation", "none"]] = None
    anomaly_method: Optional[Literal["auto", "iqr", "zscore", "isolation_forest"]] = None
    correlation_method: Optional[Literal["auto", "pearson", "spearman"]] = None

    filters: Optional[List[Filter]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    scenario_changes: Optional[List[ScenarioChange]] = None
    input_values: Optional[List[InputValue]] = None

    # ---- Phase 2 fields
    pivot_column: Optional[str] = Field(default=None, description="pivot_table: the column whose values become the table columns (group_by gives the rows).")
    business_function: Optional[BusinessFunction] = Field(default=None, description="generate_report: the business purpose of the report.")
    chart_type: Optional[ChartType] = Field(default=None, description="Chart the user explicitly asked for (pie, line, ...); auto otherwise.")
    n_clusters: Optional[int] = Field(default=None, ge=2, le=10, description="segment_records: number of segments if the user names one.")
    comparison_start: Optional[str] = Field(default=None, description="explain_change: start (ISO date) of the later/comparison period.")
    comparison_end: Optional[str] = Field(default=None, description="explain_change: end of the comparison period.")
    baseline_start: Optional[str] = Field(default=None, description="explain_change: start of the earlier/baseline period.")
    baseline_end: Optional[str] = Field(default=None, description="explain_change: end of the baseline period.")

    uses_previous_context: bool = False
    reasoning: str = ""
