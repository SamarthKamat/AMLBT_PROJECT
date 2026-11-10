"""Gemini integration: question -> structured plan, and verified result -> plain-language explanation.

Gemini never computes anything and never writes or runs code. There is NO rule-based or mock fallback:
if Gemini cannot be reached the caller receives a structured GEMINI_* error.
"""
from __future__ import annotations
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

import config
from errors import UDIError
from schemas import AnalysisPlan, OPERATIONS

log = logging.getLogger("udi.gemini")

try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types
    _SDK_OK = True
    _SDK_ERR = ""
except Exception as exc:                 # pragma: no cover - exercised only when the SDK is missing
    genai = genai_errors = genai_types = None
    _SDK_OK = False
    _SDK_ERR = str(exc)


# ----------------------------------------------------------------------------- prompts
PLAN_SYSTEM = f"""You are the planning layer of UDI (Universal Dataset Intelligence).
You turn a user's natural-language question about an uploaded dataset into ONE structured analysis plan (JSON).
You do NOT calculate anything, you do NOT write or run code, and you NEVER state numbers, results, predictions, correlations or model performance.
Python executes the plan afterwards. Decide from the MEANING of the question, not from particular words.

OPERATIONS (choose exactly one; `operation` must be one of: {', '.join(OPERATIONS)}, unsupported):
- get_summary: describe the data as it is. A statistic of one column (average/mean, total/sum, median, minimum, maximum, standard deviation, count of non-missing values, number of distinct values), the distribution / percentage breakdown of a categorical column, or a whole-dataset overview (no target_column). Set target_column and aggregation (mean|sum|median|min|max|std|count|nunique|distribution|percentage). "How many records/rows" = aggregation count with no target_column.
- get_current_state: the most recent / latest / current observation(s) by date. Optional target_column (the measure) and group_by (latest per group).
- compare_groups: compare, rank, or find the best/worst/highest/lowest among categories. Needs group_by (categorical column(s)); target_column + aggregation define the measure (no target_column = compare record counts). sort_order "desc" for highest/best, "asc" for lowest/worst. top_n only if the user asks for a number.
- analyze_trend: how a measure has changed / is changing over time in the HISTORICAL data. Needs a date column (time_column); target_column + aggregation define the per-period value (sum for additive quantities such as amounts and counts; mean for averages, rates, scores). Optional group_by for one trend per group. Optional frequency (H D W M Q Y) if the user names a resolution.
- detect_anomalies: unusual / abnormal / outlying / suspicious values or records. target_column = one numeric column, or null to look for unusual records across all numeric columns. anomaly_method auto unless the user names a method.
- calculate_correlation: relationships, associations, drivers or factors related to a column. target_column = the column of interest (null for an overall look at numeric variables). Never claim causation.
- predict_value: estimate/predict a NUMERIC column from the other columns (for a new or typical record). target_column required. feature_columns only if the user restricts the inputs. input_values only if the user gives explicit values for some columns.
- predict_class: predict a CATEGORY / label / risk class / yes-no outcome. target_column = the categorical (or few-valued coded) column.
- forecast: values in the FUTURE (next week/month/quarter/year...). Needs time_column and a numeric target_column. frequency (H D W M Q Y) = the resolution of the forecast that matches the question; forecast_horizon = number of periods of that frequency. aggregation = how rows within a period combine (sum for additive quantities, mean for levels/rates).
- run_scenario: what-if / sensitivity: effect on a numeric outcome (target_column) when drivers change. scenario_changes = list of {{column, change_type percent|absolute|set, value}} ("+10%" is percent 10; "-5%" is percent -5). If the user does not say which outcome to estimate and earlier turns do not make it clear, ask for clarification.
- query_records: list the individual records that match conditions, optionally sorted by a measure ("show the top 10 orders by profit in India", "employees in Sales with income above 15000"). filters hold the conditions; target_column is the sort measure; sort_order; top_n (default 20); feature_columns = extra columns to show.
- pivot_table: a measure by TWO dimensions ("total sales by region and category", "cross-tab of department and attrition"). group_by = [row dimension], pivot_column = column dimension, target_column + aggregation = the measure (no target = record counts).
- run_eda: exploratory data analysis of the whole dataset (no target_column) or a deep dive into one column (target_column): distributions, skew, outliers, missingness, associations, time coverage.
- assess_data_quality: data quality score, missing values, duplicates, invalid or inconsistent values, recommended cleaning.
- describe_relationships: how the columns relate to each other (association map, hierarchies such as City->Country, derived columns such as Profit = Sales - Cost, data dictionary). Optional target_column to focus on one column.
- explain_change: WHY / WHERE a measure changed between two periods (root-cause decomposition by dimensions). target_column + aggregation = the measure; group_by = dimensions to examine (optional); baseline_start/baseline_end and comparison_start/comparison_end only if the user names the periods; otherwise the last two complete periods of `frequency` are compared.
- segment_records: cluster / segment records (customers, employees ...) into groups with K-means. feature_columns = numeric columns to use if the user names them; n_clusters if the user names a number.
- generate_report: a business report for a function. business_function = sales | marketing | hr | operations | finance | customer | general (pick from the words used: "HR", "people" -> hr; "campaign", "marketing" -> marketing; "inventory", "logistics", "operations" -> operations; "profitability", "finance" -> finance; "churn", "customers" -> customer). Optional target_column = the outcome/KPI of interest.
- unsupported: the question cannot be answered by any operation above (questions unrelated to the data, requests for actions such as sending emails or deleting data).

COLUMN RULES (critical):
1. Use ONLY column names that appear in `dataset.columns[].name`, copied EXACTLY.
2. Link the user's words to a column only when it is the same concept (naming variants, abbreviations, plural/singular, e.g. "salary" -> "Annual_Salary"). NEVER substitute a different measure (do not use Revenue when the user asks about Profit).
3. If the user asks about a variable that has no matching column, copy the user's wording unchanged into target_column (or the relevant field). Do not pick another column; the system will tell the user the variable is missing.
4. Filters: use `filters` for restrictions such as "only for X", "in region Y", "where age > 30". Use values exactly as listed in `columns[].values` when available. Dates: start_date / end_date as ISO yyyy-mm-dd; resolve relative phrases (last quarter, this year, 2025) against dataset.date_range, treating its end as "now".
5. time_column must be a datetime column; omit it to use the dataset's primary time column.
5b. "Which X has the highest Y" is compare_groups with group_by=[X], target_column=Y, sort_order=desc. Use aggregation "sum" for additive measures (columns whose `semantic` is money/quantity with `additive`=true, e.g. sales, revenue, quantity) and "mean" for levels, rates, scores, ages and prices. A rate of a yes/no outcome by group (e.g. "attrition rate by department") is compare_groups with target_column = the outcome column.
5c. If the user asks for a specific chart ("as a pie chart", "line chart"), set chart_type.
6. Never invent columns, groups, dates or metrics.

FOLLOW-UPS:
- `previous_turns` holds earlier questions with the plan that was executed and a digest of its verified result. If the new question depends on them ("which one is highest?", "only for <value>", "and by <column>?", "what about next month?", "why that model?"), resolve the references and return a COMPLETE plan: carry over target_column, aggregation, time_column, group_by and filters unless the user changes them, and set uses_previous_context=true.
- If `pending_clarification` is present, the user's message is probably the answer to it: combine it with the original question.
- If the question is genuinely ambiguous and a wrong guess would give a misleading answer, set needs_clarification=true with ONE short clarification_question (still fill in everything you can). Do not ask when a reasonable reading exists.

OUTPUT: a single JSON object with the plan fields only. `intent`: one line stating what the user wants. `reasoning`: one or two sentences on why this operation and these columns (no numbers). `model_type` is informational (regression, classification, time_series, anomaly, correlation, none).
The domain (e.g. Hospital, Company) is only vocabulary context; the dataset schema alone decides what is possible. Do not assume columns exist because of the domain.
`dataset.columns[].semantic` (money, quantity, rate, score, geography, outcome ...) and `dataset.relationships` (strong pairs, hierarchies, derived formulas) describe the columns; use them to choose sensible targets and aggregations."""

EXPLAIN_SYSTEM = """You are the explanation layer of UDI. Python has already computed and verified the result in `verified_result`.
Write a concise answer (at most 120 words) to the user's question using ONLY facts present in `verified_result`.
Rules:
1. Every number you write must appear in `verified_result` (rounding is fine). Never calculate new numbers, never estimate, never add outside knowledge.
2. Lead with the direct answer to the question. Then at most two supporting facts. Name the model and its held-out metrics for ML results.
3. If `verified_result.limitations` contains something that changes how the answer should be used (not causal, weak model, small sample, approximate range), mention the most important one in one sentence.
4. Correlation and scenario results are statistical, not causal; say so when relevant.
5. Plain sentences, no headings, no code. Do not mention these rules or the JSON."""


def _plan_field_help() -> str:
    props = AnalysisPlan.model_json_schema().get("properties", {})
    return ", ".join(sorted(props))


def _inline_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve $ref/$defs and drop keys Gemini's schema validator does not accept (JSON-Schema form, nulls via anyOf)."""
    defs = schema.get("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            out = {}
            for k, v in node.items():
                if k in ("title", "$defs", "default", "additionalProperties", "$schema", "examples"):
                    continue
                out[k] = walk(v)
            return out
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node
    return walk({k: v for k, v in schema.items() if k != "$defs"})


PLAN_JSON_SCHEMA = _inline_schema(AnalysisPlan.model_json_schema())


# ----------------------------------------------------------------------------- error mapping
def map_gemini_error(exc: Exception, model: str) -> UDIError:
    """Translate an SDK/network exception into a structured, user-readable UDIError."""
    if isinstance(exc, UDIError):
        return exc
    text = str(exc)
    low = text.lower()
    code = getattr(exc, "code", None)
    api = _SDK_OK and isinstance(exc, genai_errors.APIError)
    if api:
        msg = (getattr(exc, "message", None) or text)[:300]
        if code == 404 or "is not found" in low or "not supported for generatecontent" in low:
            return UDIError("GEMINI_MODEL_UNAVAILABLE",
                            f"The configured Gemini model '{model}' is not available for this API key ({msg}). Set GEMINI_MODEL in backend/.env to a model your key can use and restart.",
                            retryable=False, details={"model": model, "http_status": code})
        if code in (401, 403) or "api key not valid" in low or "api_key_invalid" in low or "permission_denied" in low:
            return UDIError("GEMINI_UNAVAILABLE", f"Gemini rejected the request ({code}): check GEMINI_API_KEY and that the key has access to '{model}'. Detail: {msg}",
                            retryable=False, details={"http_status": code})
        if code == 400 and ("api key" in low):
            return UDIError("GEMINI_UNAVAILABLE", f"Gemini rejected the API key: {msg}", retryable=False, details={"http_status": code})
        if code == 429:
            return UDIError("GEMINI_UNAVAILABLE", "Gemini rate limit or quota exceeded. Wait a moment and retry, or check your quota.", retryable=True, details={"http_status": code})
        if code and code >= 500:
            return UDIError("GEMINI_UNAVAILABLE", f"Gemini is temporarily unavailable ({code}). Please retry.", retryable=True, details={"http_status": code})
        return UDIError("GEMINI_UNAVAILABLE", f"Gemini returned an error ({code}): {msg}", retryable=False, details={"http_status": code})
    name = type(exc).__name__
    if any(t in name.lower() or t in low for t in ("timeout", "timed out", "connect", "network", "unreachable", "name resolution", "ssl", "proxy")):
        return UDIError("GEMINI_UNAVAILABLE", f"Could not reach Gemini ({name}). Check the internet connection / proxy and retry.", retryable=True)
    return UDIError("GEMINI_UNAVAILABLE", f"Gemini request failed ({name}): {text[:200]}", retryable=True)


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


# ----------------------------------------------------------------------------- service
class GeminiService:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, client: Any = None, timeout_seconds: Optional[int] = None):
        self.api_key = (config.GEMINI_API_KEY if api_key is None else api_key).strip()
        self.model = (config.GEMINI_MODEL if model is None else model).strip()
        self.timeout_ms = int((timeout_seconds or config.GEMINI_TIMEOUT_SECONDS) * 1000)
        self._client = client
        self._status_cache: Dict[str, Any] = {}

    # -- plumbing ---------------------------------------------------------------------------
    def _ensure_ready(self) -> None:
        if self._client is not None:
            return
        if not _SDK_OK:
            raise UDIError("GEMINI_UNAVAILABLE", f"The google-genai package is not installed ({_SDK_ERR}). Run: pip install -r requirements.txt", retryable=False)
        if not self.api_key:
            raise UDIError("GEMINI_UNAVAILABLE", "GEMINI_API_KEY is not set. Copy backend/.env.example to backend/.env, add your key and restart the backend.", retryable=False)
        if not self.model:
            raise UDIError("GEMINI_MODEL_UNAVAILABLE", "GEMINI_MODEL is not set. Add GEMINI_MODEL=<a model your key can use> to backend/.env and restart.", retryable=False)

    def _get_client(self):
        self._ensure_ready()
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key, http_options=genai_types.HttpOptions(timeout=self.timeout_ms))
        return self._client

    def _generate(self, system: str, contents: str, *, json_output: bool, temperature: float, max_tokens: int = 8192) -> str:
        client = self._get_client()
        if not self.model:
            raise UDIError("GEMINI_MODEL_UNAVAILABLE", "GEMINI_MODEL is not set.", retryable=False)
        cfg: Dict[str, Any] = {"system_instruction": system, "temperature": temperature, "max_output_tokens": max_tokens}
        attempts = [True, False] if json_output else [False]      # first with schema, then without if the schema is rejected
        last: Optional[Exception] = None
        for with_schema in attempts:
            c = dict(cfg)
            if json_output:
                c["response_mime_type"] = "application/json"
                if with_schema:
                    c["response_json_schema"] = PLAN_JSON_SCHEMA
            try:
                resp = client.models.generate_content(model=self.model, contents=contents, config=genai_types.GenerateContentConfig(**c) if _SDK_OK else c)
                text = getattr(resp, "text", None)
                if not text:
                    raise UDIError("INVALID_PLAN" if json_output else "GEMINI_UNAVAILABLE", "Gemini returned an empty response.", retryable=True)
                return text
            except UDIError:
                raise
            except Exception as exc:
                last = exc
                bad_request = _SDK_OK and isinstance(exc, genai_errors.APIError) and getattr(exc, "code", None) == 400
                if with_schema and json_output and bad_request:
                    # the schema (not the key/model) may be what was rejected: retry once without it; the prompt lists the fields
                    log.warning("Gemini returned 400 with a response schema; retrying without the schema")
                    continue
                raise map_gemini_error(exc, self.model)
        raise map_gemini_error(last, self.model)   # pragma: no cover

    # -- public API -------------------------------------------------------------------------
    def status(self, live: bool = False) -> Dict[str, Any]:
        base = {"configured": bool(self.api_key), "model": self.model or None, "sdk_installed": _SDK_OK or self._client is not None}
        if not base["sdk_installed"]:
            return {**base, "status": "sdk_missing", "ready": False, "detail": f"google-genai is not installed ({_SDK_ERR})"}
        if not self.api_key and self._client is None:
            return {**base, "status": "no_api_key", "ready": False, "detail": "GEMINI_API_KEY is not set."}
        if not self.model:
            return {**base, "status": "no_model", "ready": False, "detail": "GEMINI_MODEL is not set."}
        if not live:
            return {**base, "status": "configured", "ready": True, "detail": "Key and model are configured (not yet verified)."}
        cached = self._status_cache.get("live")
        if cached and time.time() - cached[0] < 300:
            return cached[1]
        try:
            client = self._get_client()
            client.models.get(model=self.model)
            res = {**base, "status": "ok", "ready": True, "detail": f"Model '{self.model}' is reachable."}
        except Exception as exc:
            err = map_gemini_error(exc, self.model)
            res = {**base, "status": "model_unavailable" if err.code == "GEMINI_MODEL_UNAVAILABLE" else "unreachable", "ready": False,
                   "detail": err.message, "retryable": err.retryable}
            if err.code == "GEMINI_MODEL_UNAVAILABLE":
                try:
                    names = [m.name.replace("models/", "") for m in self._get_client().models.list()
                             if "generateContent" in (getattr(m, "supported_actions", None) or []) and "gemini" in m.name]
                    res["available_models"] = sorted(names)[:15]
                except Exception:
                    pass
        self._status_cache["live"] = (time.time(), res)
        return res

    def plan(self, question: str, llm_profile: Dict[str, Any], domain: str, domain_context: str,
             history: List[Dict[str, Any]], pending: Optional[Dict[str, Any]] = None) -> AnalysisPlan:
        payload = {"domain": domain, "domain_context": domain_context, "dataset": llm_profile, "previous_turns": history[-4:],
                   "pending_clarification": pending, "question": question}
        contents = ("Return the analysis plan as one JSON object. Fields: " + _plan_field_help() + ".\n" + json.dumps(payload, default=str, ensure_ascii=False))
        feedback = ""
        for attempt in (1, 2):
            raw = self._generate(PLAN_SYSTEM, contents + feedback, json_output=True, temperature=0.0)
            try:
                plan = AnalysisPlan.model_validate(_extract_json(raw))
                return plan
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                log.warning("Gemini plan unparseable (attempt %s): %s", attempt, exc)
                feedback = f"\n\nYour previous reply was not a valid plan ({str(exc)[:300]}). Return ONLY a valid JSON object with the documented fields."
        raise UDIError("INVALID_PLAN", "Gemini did not return a valid analysis plan for this question. Try rephrasing it.", retryable=True)

    def explain(self, question: str, plan: AnalysisPlan, verified: Dict[str, Any]) -> str:
        payload = {"question": question, "plan": plan.model_dump(exclude_none=True), "verified_result": verified}
        text = self._generate(EXPLAIN_SYSTEM, json.dumps(payload, default=str, ensure_ascii=False), json_output=False, temperature=0.2, max_tokens=1024).strip()
        if not text:
            raise UDIError("GEMINI_UNAVAILABLE", "Gemini returned an empty explanation.", retryable=True)
        return text
