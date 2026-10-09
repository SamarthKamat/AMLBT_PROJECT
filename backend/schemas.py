"""The structured analysis plan.

Gemini fills this in; Python validates it against the real dataset before anything runs.
Gemini never returns results, only intent + column references.
"""
from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

OPERATIONS = [
    "get_summary", "get_current_state", "compare_groups", "analyze_trend", "detect_anomalies",
    "calculate_correlation", "predict_class", "predict_value", "forecast", "run_scenario",
]
Operation = Literal[
    "get_summary", "get_current_state", "compare_groups", "analyze_trend", "detect_anomalies",
    "calculate_correlation", "predict_class", "predict_value", "forecast", "run_scenario", "unsupported",
]
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

    uses_previous_context: bool = False
    reasoning: str = ""
