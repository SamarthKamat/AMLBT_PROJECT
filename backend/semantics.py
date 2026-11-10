"""Semantic layer: what each column MEANS in business terms (Phase 2).

The profiler gives a statistical role (numeric / categorical / datetime / identifier ...). This module adds a
business semantic type (money, quantity, rate, score, age, duration, geography, person, product, outcome flag ...),
whether the measure is additive (can be summed across rows), and which business functions it is relevant to.

Everything is inferred from column names AND values; nothing is tied to one dataset. The result is used by
  * the planners (default aggregation: "highest sales by country" = SUM of an additive measure),
  * the data-quality engine (a negative age or a 140% rate is invalid),
  * the report agent (which KPIs matter for an HR vs a marketing report).
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

FUNCTIONS = ["sales", "marketing", "hr", "operations", "finance", "customer"]

# (semantic type, keywords). A keyword matches a whole name token (or a token that starts with it, for plurals).
# Order matters: first match wins. Token matching avoids accidents such as "cur-RENT" -> rent.
_NAME_RULES = [
    ("outcome", {"churn", "attrition", "default", "fraud", "converted", "subscribed", "y", "target", "outcome", "label", "risk", "flag", "late", "approved", "responded"}),
    ("rate", {"rate", "ratio", "pct", "percent", "percentage", "share", "margin", "utilization", "utilisation", "occupancy", "conversion", "ctr", "discount", "%"}),
    ("score", {"score", "rating", "satisfaction", "involvement", "engagement", "performance", "nps", "grade", "level", "rank", "index", "balance", "idx"}),
    ("money", {"sales", "revenue", "turnover", "income", "salary", "salaries", "wage", "wages", "pay", "price", "cost", "costs", "profit", "loss", "amount",
               "charge", "charges", "fee", "fees", "spend", "budget", "donation", "donations", "aid", "gmv", "earnings", "expense", "expenses",
               "cash", "payment", "rent", "bonus", "commission", "hike"}),
    ("age", {"age"}),
    ("duration", {"years", "year", "tenure", "experience", "exp", "days", "duration", "hours", "minutes", "time", "wait", "delay", "since", "months", "pdays"}),
    ("quantity", {"qty", "quantity", "units", "count", "number", "num", "volume", "orders", "customers", "patients", "students", "beneficiaries",
                  "attendance", "registrations", "requests", "complaints", "visits", "calls", "campaign", "previous", "staff", "workers", "volunteers",
                  "beds", "headcount", "employees", "demand", "inventory", "stock", "population", "items", "tickets", "seats", "capacity", "waste",
                  "resources", "companies", "employed", "times", "shortage"}),
    ("geography", {"country", "state", "city", "region", "zone", "district", "market", "territory", "area", "province", "location", "ward",
                   "postal", "zip", "continent", "branch", "pincode"}),
    ("person", {"name", "manager", "agent", "firstname", "lastname"}),
    ("product", {"product", "item", "sku", "category", "subcategory", "sub", "brand", "model", "service", "plan", "package", "department", "dept",
                 "job", "role", "segment", "channel", "ship", "mode", "type", "class", "education", "field", "contract", "method", "event", "programme", "program", "course"}),
]


def _rule_hit(tokens: List[str], words) -> bool:
    for t in tokens:
        if t in words:
            return True
        for w in words:
            if len(w) >= 4 and t.startswith(w) and len(t) - len(w) <= 2:      # plural / light inflection
                return True
    return False


_ADDITIVE = {"money", "quantity"}
_NON_ADDITIVE_TOKENS = {"price", "rate", "avg", "average", "mean", "per", "index", "score", "rating", "age", "level", "percent", "pct",
                        "ratio", "margin", "hourly", "idx", "hike", "unit",
                        "income", "salary", "salaries", "wage", "wages", "distance", "monthly", "daily", "weekly", "annual"}

_FUNCTION_TERMS = {
    "sales": r"sales|revenue|order|product|quantity|units|region|country|market|customer|price|discount|profit|segment|category|deal|store",
    "marketing": r"campaign|channel|contact|customer|segment|conversion|churn|subscribed|response|promotion|pdays|previous|poutcome|^y$|marketing|lead|click|impression|tenure|contract|age|job|education|marital|demand",
    "hr": r"employee|attrition|salary|income|job|role|department|satisfaction|performance|overtime|tenure|years|hike|training|education|gender|age|marital|travel|distance_from_home|worklife|staff|faculty|stock_option|promotion",
    "operations": r"inventory|stock|demand|ship|delivery|lead_time|wait|capacity|beds|utili|occupancy|workers|staff|resources|waste|complaints|requests|queue|downtime|priority|shipping|logistics|attendance|patients|order_date|ship_date",
    "finance": r"revenue|cost|profit|margin|price|discount|charges|income|expense|budget|donation|aid|amount|balance|payment|shipping_cost|sales|fee|loan|default|euribor|rate",
    "customer": r"customer|churn|tenure|contract|satisfaction|rating|complaint|segment|loyalty|subscription|payment|monthly_charges|total_charges|partner|dependents|senior",
}


def _norm(name: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))          # camelCase -> camel_Case
    s = re.sub(r"[^A-Za-z0-9%]+", "_", s).strip("_").lower()
    return s


def column_tokens(name: str) -> List[str]:
    return [t for t in _norm(name).split("_") if t]


_YES_NO = {"yes", "no", "y", "n", "true", "false", "1", "0", "late", "ontime", "on time", "churned", "retained"}


def _is_flag(s: pd.Series, role: str) -> bool:
    """A two-valued yes/no style column (0/1, Yes/No, True/False)."""
    nn = s.dropna()
    if nn.empty or role not in ("categorical", "binary"):
        return False
    if pd.api.types.is_numeric_dtype(s):
        return role == "binary" and set(np.unique(nn.astype(float))) <= {0.0, 1.0}
    vals = {str(v).strip().lower() for v in nn.unique()[:20]}
    return bool(vals) and vals <= _YES_NO


def infer_semantics(df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """{column: {semantic, additive, functions[], unit_hint, description}}"""
    out: Dict[str, Dict[str, Any]] = {}
    for c in profile["columns"]:
        name, role = c["name"], c["role"]
        n = _norm(name)
        sem = None
        if role == "datetime":
            sem = "date"
        elif role == "identifier":
            sem = "identifier"
        elif role == "text":
            sem = "text"
        elif role in ("constant", "empty"):
            sem = "constant"
        toks = column_tokens(name)
        flag = _is_flag(df[name], role)
        if sem is None and flag:
            sem = "outcome" if (_rule_hit(toks, _NAME_RULES[0][1]) or role == "binary" and toks[-1:] == ["y"]) else "flag"
        if sem is None:
            for st, words in _NAME_RULES:
                if st == "outcome" and role not in ("binary",):
                    continue                        # an outcome label must be two-valued (multi-class targets are 'category')
                if st == "person" and _rule_hit(toks, _NAME_RULES[-1][1]):
                    continue                        # "Product Name" is a product, not a person
                if _rule_hit(toks, words):
                    # numeric-only semantic types must sit on numeric columns and vice versa
                    numeric = pd.api.types.is_numeric_dtype(df[name]) and role in ("numeric", "binary")
                    if st in ("money", "quantity", "rate", "score", "age", "duration") and not numeric:
                        continue
                    if st in ("person", "product") and numeric:
                        continue
                    sem = st
                    break
        if sem is None:
            sem = "measure" if (pd.api.types.is_numeric_dtype(df[name]) and role in ("numeric", "binary")) else "category"
        # value checks that refine a name-based guess
        additive = sem in _ADDITIVE and not (set(toks) & _NON_ADDITIVE_TOKENS)
        funcs = [f for f, pat in _FUNCTION_TERMS.items() if re.search(pat, n)]
        out[name] = {"semantic": sem, "additive": bool(additive), "functions": funcs,
                     "description": describe_column(name, role, sem, c)}
    return out


def describe_column(name: str, role: str, sem: str, c: Dict[str, Any]) -> str:
    """One plain-language line per column for the data dictionary (computed facts only)."""
    miss = f", {c['missing_pct']:.1f}% missing" if c.get("missing_pct") else ""
    if sem == "date":
        return f"Date/time from {str(c.get('start', ''))[:10]} to {str(c.get('end', ''))[:10]} ({c.get('distinct_timestamps', '?')} distinct timestamps, ~{c.get('inferred_frequency') or '?'} spacing){miss}."
    if sem == "identifier":
        return f"Record identifier ({c['unique_values']:,} distinct values); excluded from modelling{miss}."
    if role in ("numeric", "binary") and c.get("mean") is not None:
        kind = {"money": "Monetary measure", "quantity": "Count/quantity measure", "rate": "Rate/percentage", "score": "Score/rating",
                "age": "Age", "duration": "Duration/tenure", "outcome": "Binary outcome flag", "flag": "Yes/no indicator", "measure": "Numeric measure", "geography": "Geographic code"}.get(sem, "Numeric measure")
        return f"{kind}; range {c['min']:,.4g} to {c['max']:,.4g}, mean {c['mean']:,.4g}, median {c['median']:,.4g}{miss}."
    top = (c.get("top_values") or [{}])[0]
    kind = {"geography": "Geographic dimension", "person": "Person name", "product": "Business dimension", "outcome": "Outcome/label",
            "category": "Categorical dimension", "text": "Free text", "constant": "Constant column", "flag": "Yes/no indicator"}.get(sem, "Categorical dimension")
    if top:
        return f"{kind} with {c['unique_values']:,} distinct values; most common '{top.get('value')}' ({top.get('count', 0):,} rows){miss}."
    return f"{kind} with {c['unique_values']:,} distinct values{miss}."


def kpi_columns(semantics: Dict[str, Dict[str, Any]], function: str = "general", limit: int = 4) -> List[str]:
    """Numeric KPIs, ranked for a business function: function-relevant money first, then quantities."""
    rank = {"money": 0, "quantity": 1, "rate": 2, "score": 3, "duration": 4, "measure": 5, "age": 6}
    cands = [(c, v) for c, v in semantics.items() if v["semantic"] in rank]
    def key(item):
        c, v = item
        rel = 0 if (function in v["functions"] or function == "general") else 1
        return (rel, rank[v["semantic"]], 0 if v["additive"] else 1)
    cands.sort(key=key)
    return [c for c, _ in cands[:limit]]


def outcome_columns(semantics: Dict[str, Dict[str, Any]], function: str = "general") -> List[str]:
    outs = [c for c, v in semantics.items() if v["semantic"] == "outcome"]
    rel = [c for c in outs if function in semantics[c]["functions"]]
    return rel + [c for c in outs if c not in rel]


def dimension_columns(semantics: Dict[str, Dict[str, Any]], profile: Dict[str, Any], function: str = "general",
                      max_unique: int = 30, limit: int = 6) -> List[str]:
    uniq = {c["name"]: c["unique_values"] for c in profile["columns"]}
    roles = {c["name"]: c["role"] for c in profile["columns"]}
    cands = [c for c, v in semantics.items() if v["semantic"] in ("geography", "product", "category", "outcome", "flag")
             and roles.get(c) in ("categorical", "binary") and 2 <= uniq.get(c, 0) <= max_unique]
    cands.sort(key=lambda c: (0 if function in semantics[c]["functions"] else 1, 0 if semantics[c]["semantic"] != "outcome" else 1, uniq[c]))
    return cands[:limit]


def default_aggregation(semantics: Dict[str, Dict[str, Any]], column: Optional[str]) -> str:
    """'Highest sales by country' means total sales; 'highest satisfaction by department' means average."""
    if column and semantics.get(column, {}).get("additive"):
        return "sum"
    return "mean"
