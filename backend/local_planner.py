"""Local planner (Phase 2): natural-language question -> AnalysisPlan, without any external LLM.

Pipeline
  1. Schema linking   find column mentions (exact name, token, light inflection, curated same-concept synonyms,
                      fuzzy match) and category-value mentions ("Goa", "Sales Representative") in the question.
  2. Masking          replace them by role tokens (numcol, catcol, datecol, valtok, numtok) so the question
                      exposes its structure: "which catcol has the highest numcol".
  3. Intent model     TF-IDF (word 1-3 grams + char 2-5 grams) + multinomial logistic regression trained on the
                      templated corpus in planner_corpus.py; 19 classes (18 operations + unsupported).
  4. Slot filling     deterministic rules fill target / group_by / aggregation / filters / horizon / scenario
                      changes from the linked mentions and cue words.
  5. Confidence       a low-probability intent becomes a clarification with the top interpretations as options.

The plan then goes through exactly the same validation gate as a Gemini plan; the local planner never computes
results. A request for a column that does not exist is passed on with the user's wording so the validator rejects
it with the list of available columns (never substituted).
"""
from __future__ import annotations
import difflib
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from config import MODEL_DIR
from planner_corpus import T as TEMPLATES, build_corpus
from schemas import AnalysisPlan
from semantics import column_tokens, default_aggregation

log = logging.getLogger("udi.local_planner")
MODEL_PATH = MODEL_DIR / "planner" / "intent_model.joblib"
MIN_CONFIDENCE = 0.30
OP_DESCRIPTIONS = {
    "get_summary": "a summary statistic", "get_current_state": "the latest value", "compare_groups": "a comparison across groups",
    "analyze_trend": "a trend over time", "detect_anomalies": "unusual records / outliers", "calculate_correlation": "what is correlated with a column",
    "predict_class": "a classification model", "predict_value": "a prediction (regression)", "forecast": "a forecast of future values",
    "run_scenario": "a what-if scenario", "query_records": "a list of matching records", "pivot_table": "a pivot table",
    "run_eda": "an exploratory data analysis", "assess_data_quality": "a data-quality check", "describe_relationships": "how the columns relate",
    "explain_change": "why a measure changed", "segment_records": "a segmentation", "generate_report": "a business report",
}
# Same-concept synonym groups (naming variants only - never different measures such as revenue vs profit).
SYNONYMS = [
    {"income", "salary", "salaries", "pay", "wage", "wages", "compensation", "earnings"},
    {"sales", "revenue", "turnover"},
    {"employees", "employee", "staff", "headcount", "workers", "personnel"},
    {"customers", "customer", "clients", "client"},
    {"country", "countries", "nation", "nations"},
    {"attrition", "leaving", "leave", "left", "resignation", "resignations", "quit", "quitting", "leavers"},
    {"churn", "churned", "churning", "cancellation", "cancellations", "leave", "leaving", "left"},
    {"quantity", "units", "qty", "volume"},
    {"cost", "costs", "expense", "expenses", "spend", "spending"},
    {"department", "departments", "dept", "division"},
    {"tenure", "seniority"},
    {"age", "ages", "old"},
    {"gender", "sex"},
    {"profit", "profits"},
    {"discount", "discounts"},
    {"category", "categories"},
    {"product", "products", "item", "items"},
    {"region", "regions"},
    {"city", "cities"},
    {"state", "states"},
    {"date", "day", "time"},
]
STOP = set("the a an of for in on to by and or is are was were be what which who how many much do does did with from per each show me give list find all any this that my our there their it its as at than".split())
FUNCTION_WORDS = {"marketing": "marketing", "sales": "sales", "hr": "hr", "human resources": "hr", "people": "hr", "operations": "operations", "operational": "operations",
                  "ops": "operations", "logistics": "operations", "supply chain": "operations", "finance": "finance", "financial": "finance", "customer": "customer",
                  "customers": "customer", "business": "general", "executive": "general", "management": "general", "board": "general"}
CHART_WORDS = [("pie", "pie"), ("donut", "pie"), ("doughnut", "pie"), ("line chart", "line"), ("line graph", "line"), ("bar chart", "bar"), ("bar graph", "bar"),
               ("column chart", "bar"), ("histogram", "histogram"), ("scatter", "scatter"), ("heatmap", "heatmap"), ("heat map", "heatmap"), ("box plot", "box"),
               ("boxplot", "box"), ("area chart", "area"), ("as a table", "table")]
AGG_CUES = [("standard deviation", "std"), ("std", "std"), ("spread", "std"), ("variation", "std"), ("median", "median"), ("average", "mean"), ("avg", "mean"),
            ("mean", "mean"), ("typical", "mean"), ("per record", "mean"), ("total", "sum"), ("sum", "sum"), ("overall amount", "sum"), ("combined", "sum"),
            ("how many unique", "nunique"), ("how many distinct", "nunique"), ("number of unique", "nunique"), ("number of distinct", "nunique"),
            ("unique", "nunique"), ("distinct", "nunique"), ("distribution", "distribution"), ("breakdown", "distribution"), ("composition", "distribution"),
            ("mix", "distribution"), ("split", "distribution"), ("percentage", "percentage"), ("share", "percentage"), ("proportion", "percentage"),
            ("how many", "count"), ("number of", "count"), ("count", "count"),
            ("maximum", "max"), ("highest", "max"), ("largest", "max"), ("biggest", "max"), ("max", "max"), ("peak", "max"),
            ("minimum", "min"), ("lowest", "min"), ("smallest", "min"), ("min", "min")]
DESC = ("highest", "most", "top", "best", "largest", "biggest", "maximum", "greatest", "leading", "max", "more", "higher", "sells the most", "earns the most")
ASC = ("lowest", "least", "worst", "smallest", "bottom", "minimum", "fewest", "min", "lower", "less", "weakest", "underperforming")
FREQ_WORDS = [("hourly", "H"), ("hour", "H"), ("daily", "D"), ("day", "D"), ("weekly", "W"), ("week", "W"), ("monthly", "M"), ("month", "M"),
              ("quarterly", "Q"), ("quarter", "Q"), ("yearly", "Y"), ("annual", "Y"), ("annually", "Y"), ("year", "Y")]
FOLLOW_UP = re.compile(r"^(and|also|now|what about|how about|same|only|just|instead|but|then|ok|okay|break it down|split it|for|in)\b|\b(instead|as well|too|same for|the same|that|those|it)\b")
NUM = r"(-?\d+(?:[.,]\d+)?)"
LEAD_IN = {"please", "can", "could", "you", "would", "kindly", "hey", "i", "want", "to", "now", "then", "so", "and", "ok", "okay"}
VERB_LIKE = {"segment", "cluster", "group", "compare", "show", "list", "rank", "count", "forecast", "predict", "sort", "filter", "plot", "chart",
             "explore", "summarise", "summarize", "describe", "report", "classify", "estimate", "model", "explain", "break"}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%.\-+ ]+", " ", str(text).lower().replace("_", " "))).strip()


def _sing(t: str) -> str:
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


@dataclass
class Mention:
    kind: str                 # column | value
    column: str
    start: int
    end: int
    text: str
    value: Optional[str] = None
    score: float = 1.0


@dataclass
class Linked:
    question: str
    tokens: List[str]
    mentions: List[Mention] = field(default_factory=list)
    unknown_terms: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------------- intent model
def _build_pipeline() -> Pipeline:
    feats = FeatureUnion([("w", TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=1, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
                          ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True))])
    return Pipeline([("f", feats), ("clf", LogisticRegression(C=8.0, max_iter=3000, class_weight="balanced"))])


def corpus_fingerprint() -> str:
    return hashlib.sha1(repr(sorted((k, tuple(v)) for k, v in TEMPLATES.items())).encode()).hexdigest()[:12]


def train_intent_model(save: bool = True) -> Dict[str, Any]:
    X, y = build_corpus()
    pipe = _build_pipeline().fit(X, y)
    bundle = {"pipeline": pipe, "fingerprint": corpus_fingerprint(), "classes": list(pipe.classes_), "n_train": len(X)}
    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH, compress=3)
    return bundle


_MODEL: Optional[Dict[str, Any]] = None


def intent_model() -> Dict[str, Any]:
    global _MODEL
    if _MODEL is None:
        b = None
        if MODEL_PATH.exists():
            try:
                b = joblib.load(MODEL_PATH)
                if b.get("fingerprint") != corpus_fingerprint():
                    b = None
            except Exception:
                b = None
        _MODEL = b or train_intent_model(save=True)
    return _MODEL


# ----------------------------------------------------------------------------- schema linking
class SchemaLinker:
    def __init__(self, df: pd.DataFrame, profile: Dict[str, Any]):
        self.profile = profile
        self.roles = {c["name"]: c["role"] for c in profile["columns"]}
        self.phrases: List[Tuple[str, str, float]] = []        # (normalised phrase, column, score)
        token_owner: Dict[str, List[str]] = {}
        for c in profile["columns"]:
            name = c["name"]
            toks = column_tokens(name)
            full = " ".join(toks)
            self.phrases.append((_norm(name), name, 1.0))
            self.phrases.append((full, name, 1.0))
            self.phrases.append((" ".join(_sing(t) for t in toks), name, 0.98))
            if len(toks) > 1:
                self.phrases.append(("".join(toks), name, 0.97))
            for t in set(toks):
                token_owner.setdefault(_sing(t), []).append(name)
        # a distinctive token ("income" in MonthlyIncome) links on its own when only one column has it
        for t, owners in token_owner.items():
            if len(owners) == 1 and len(t) >= 5 and t not in STOP and t not in ("id", "no", "num", "number", "code", "name", "date", "type", "level", "rate", "count", "total"):
                self.phrases.append((t, owners[0], 0.85))
        for group in SYNONYMS:
            owners = {self.phrases[i][1] for i in range(len(self.phrases)) if self.phrases[i][0] in group or _sing(self.phrases[i][0]) in group}
            if len(owners) == 1:
                owner = owners.pop()
                for w in group:
                    self.phrases.append((w, owner, 0.9))
        self.phrases = sorted(set(self.phrases), key=lambda p: (-len(p[0].split()), -p[2]))
        self.values: List[Tuple[str, str, str]] = []             # (normalised value, column, raw value)
        for c in profile["columns"]:
            if c["role"] in ("categorical", "binary") and c["unique_values"] <= 400 and not pd.api.types.is_numeric_dtype(df[c["name"]]):
                for v in df[c["name"]].dropna().astype(str).unique()[:400]:
                    nv = _norm(v)
                    if nv and len(nv) >= 2 and nv not in STOP and not re.fullmatch(r"[\d.\-]+", nv):
                        self.values.append((nv, c["name"], v))
        self.values.sort(key=lambda v: -len(v[0]))
        self.column_phrase_set = {p for p, _, _ in self.phrases}

    def link(self, question: str) -> Linked:
        q = _norm(question)
        tokens = q.split()
        taken = [False] * len(tokens)
        mentions: List[Mention] = []

        def span_free(i, j):
            return not any(taken[i:j])

        def mark(i, j):
            for k in range(i, j):
                taken[k] = True
        # category values first for multi-word values, then columns, then single-word values
        for n in range(5, 0, -1):
            for i in range(0, len(tokens) - n + 1):
                j = i + n
                if not span_free(i, j):
                    continue
                phrase = " ".join(tokens[i:j])
                hit_c = next(((c, s) for p, c, s in self.phrases if p == phrase or (n == 1 and _sing(phrase) == p)), None)
                if hit_c and n == 1 and phrase in VERB_LIKE and all(t in LEAD_IN for t in tokens[:i]) and j < len(tokens) and tokens[j] in (
                        "the", "my", "our", "all", "these", "this", "data", "customers", "employees", "records", "rows", "orders", "by", "it"):
                    hit_c = None                    # "Segment the orders": an instruction verb, not the Segment column
                hit_v = [(c, raw) for nv, c, raw in self.values if nv == phrase]
                if hit_v and (n >= 2 or not hit_c) and not (n == 1 and phrase in ("yes", "no", "y", "n", "true", "false") and not self._flag_context(tokens, i)):
                    col, raw = self._pick_value(hit_v, tokens, i, mentions)
                    mentions.append(Mention("value", col, i, j, phrase, raw))
                    mark(i, j)
                elif hit_c:
                    mentions.append(Mention("column", hit_c[0], i, j, phrase, score=hit_c[1]))
                    mark(i, j)
        # fuzzy single-token matches for remaining long words (typos: "revnue", "attritoin")
        unknown = []
        for i, t in enumerate(tokens):
            if taken[i] or len(t) < 5 or t in STOP or re.fullmatch(r"[\d.%\-+]+", t):
                continue
            if t in VERB_LIKE and all(x in LEAD_IN for x in tokens[:i]):
                continue
            single = [p for p, _, _ in self.phrases if " " not in p]
            stem = next((p for p in single if len(p) >= 5 and t.startswith(p) and len(t) - len(p) <= 4), None)   # profitable -> profit
            cands = [stem] if stem else difflib.get_close_matches(t, single, n=1, cutoff=0.86)
            if cands:
                col = next(c for p, c, _ in self.phrases if p == cands[0])
                mentions.append(Mention("column", col, i, i + 1, t, score=0.8))
                taken[i] = True
            else:
                unknown.append(t)
        mentions.sort(key=lambda m: m.start)
        return Linked(question=question, tokens=tokens, mentions=mentions, unknown_terms=unknown)

    def _flag_context(self, tokens: List[str], i: int) -> bool:
        return i > 0 and tokens[i - 1] in ("=", "is", "equals", "with", "where")

    def _pick_value(self, hits, tokens, i, mentions) -> Tuple[str, str]:
        if len(hits) == 1:
            return hits[0]
        # prefer the column mentioned just before the value ("department sales")
        prev_cols = [m.column for m in mentions if m.kind == "column" and m.end <= i]
        for col, raw in hits:
            if col in prev_cols:
                return col, raw
        return hits[0]

    def role(self, col: str) -> str:
        return self.roles.get(col, "categorical")


def mask(linked: Linked, roles: Dict[str, str]) -> str:
    toks = list(linked.tokens)
    out: List[str] = []
    i = 0
    starts = {m.start: m for m in linked.mentions}
    while i < len(toks):
        m = starts.get(i)
        if m:
            if m.kind == "value":
                out.append("valtok")
            else:
                r = roles.get(m.column, "categorical")
                out.append("numcol" if r == "numeric" else "datecol" if r == "datetime" else "idcol" if r == "identifier" else "catcol")
            i = m.end
            continue
        t = toks[i]
        if re.fullmatch(r"-?\d+(?:[.,]\d+)?%?", t):
            out.append("numtok" + (" percent" if t.endswith("%") else ""))
        else:
            out.append(t)
        i += 1
    return re.sub(r"\s+", " ", " ".join(out)).strip()


# ----------------------------------------------------------------------------- the planner
class LocalPlanner:
    name = "local-intent-model"

    def __init__(self):
        self._linkers: Dict[str, SchemaLinker] = {}

    def status(self, live: bool = False) -> Dict[str, Any]:
        b = intent_model()
        return {"configured": True, "model": "local TF-IDF + logistic regression intent model", "ready": True, "status": "ok",
                "detail": f"Local planner trained on {b['n_train']:,} templated questions ({len(b['classes'])} intents)."}

    def linker(self, df: pd.DataFrame, profile: Dict[str, Any], key: str) -> SchemaLinker:
        if key not in self._linkers:
            if len(self._linkers) > 32:
                self._linkers.clear()
            self._linkers[key] = SchemaLinker(df, profile)
        return self._linkers[key]

    def classify(self, masked: str) -> List[Tuple[str, float]]:
        b = intent_model()
        proba = b["pipeline"].predict_proba([masked])[0]
        order = np.argsort(proba)[::-1]
        return [(b["classes"][i], float(proba[i])) for i in order[:4]]

    def plan_question(self, question: str, df: pd.DataFrame, profile: Dict[str, Any], semantics: Dict[str, Dict[str, Any]],
                      history: Optional[List[Dict[str, Any]]] = None, pending: Optional[Dict[str, Any]] = None, key: str = "") -> Tuple[AnalysisPlan, Dict[str, Any]]:
        lk = self.linker(df, profile, key or profile.get("dataset_name", "") + str(profile.get("rows")))
        self._native = (profile.get("date_range") or {}).get("frequency")
        roles = lk.roles
        linked = lk.link(question)
        masked = mask(linked, roles)
        ranked = self.classify(masked)
        op, conf = ranked[0]
        info = {"masked": masked, "intent_ranking": [{"operation": o, "probability": round(p, 3)} for o, p in ranked], "mentions":
                [{"kind": m.kind, "column": m.column, "value": m.value, "text": m.text} for m in linked.mentions]}
        prev = (history or [])[-1] if history else None
        cols = [m for m in linked.mentions if m.kind == "column"]
        vals = [m for m in linked.mentions if m.kind == "value"]
        q = _norm(question)
        # ---- pending clarification: a short answer completes the previous question
        if pending and (len(linked.tokens) <= 5 or not cols):
            base = dict((prev or {}).get("plan") or {})
            if base.get("operation"):
                filled = self._fill_from_answer(base, cols, vals, roles)
                if filled:
                    filled["uses_previous_context"] = True
                    info["mode"] = "clarification-answer"
                    return AnalysisPlan(**filled), info
        # ---- follow-ups that only change a filter / grouping / horizon
        if prev and prev.get("plan", {}).get("operation") and self._is_follow_up(q, cols, vals, op, conf):
            plan = self._follow_up(dict(prev["plan"]), q, cols, vals, roles, semantics, df, profile)
            info["mode"] = "follow-up"
            return AnalysisPlan(**plan), info
        if conf < MIN_CONFIDENCE and op != "unsupported":
            opts = [f"I want {OP_DESCRIPTIONS.get(o, o)}" for o, _ in ranked[:3] if o != "unsupported"]
            plan = {"operation": op, "needs_clarification": True, "intent": question,
                    "clarification_question": "I am not sure what analysis you want. Did you mean " + "; or ".join(OP_DESCRIPTIONS.get(o, o) for o, _ in ranked[:3] if o != "unsupported") + "?",
                    "reasoning": f"Local intent model confidence {conf:.2f} is below {MIN_CONFIDENCE}."}
            info["options"] = opts
            return AnalysisPlan(**plan), info
        plan = self._fill(op, q, linked, cols, vals, roles, semantics, df, profile)
        plan["intent"] = question
        plan["reasoning"] = f"Local intent model: {op} (p={conf:.2f}); masked form '{masked}'."
        return AnalysisPlan(**plan), info

    # ------------------------------------------------------------------ slot filling
    def _fill(self, op, q, linked, cols, vals, roles, sem, df, profile) -> Dict[str, Any]:
        nums = [m.column for m in cols if roles.get(m.column) == "numeric" or (roles.get(m.column) == "binary" and pd.api.types.is_numeric_dtype(df[m.column]))]
        cats = [m.column for m in cols if roles.get(m.column) in ("categorical", "binary") and m.column not in nums]
        dates = [m.column for m in cols if roles.get(m.column) == "datetime"]
        nums, cats = list(dict.fromkeys(nums)), list(dict.fromkeys(cats))
        plan: Dict[str, Any] = {"operation": op}
        agg = self._agg(q)
        if any(re.search(rf"\b{w}\b", q) for w in DESC):
            plan["sort_order"] = "desc"
        if any(re.search(rf"\b{w}\b", q) for w in ASC):
            plan["sort_order"] = "asc"
        m = re.search(r"\b(?:top|bottom|first|best|worst|largest|smallest|highest|lowest)\s+(\d{1,3})\b|\b(\d{1,3})\s+(?:best|worst|top|largest|biggest|highest|lowest|smallest)\b", q)
        if m:
            plan["top_n"] = int(m.group(1) or m.group(2))
        chart = next((c for w, c in CHART_WORDS if w in q), None)
        if chart:
            plan["chart_type"] = chart
        filters, driver_cols, input_vals = self._numeric_conditions(q, linked, roles, op)
        scen = self._scenario(q, linked, roles) if op == "run_scenario" else []
        driver_cols += [s["column"] for s in scen]
        if dates:
            plan["time_column"] = dates[0]
        self._dates(q, plan, profile, op)
        target_nums = [c for c in nums if c not in driver_cols]
        val_cols = {v.column for v in vals}

        if op in ("predict_class",):
            outcome = (next((c for c in cats if sem.get(c, {}).get("semantic") == "outcome"), None)
                       or next((c for c in cats if sem.get(c, {}).get("semantic") == "flag"), None) or (cats[0] if cats else None))
            if outcome is None:
                outcome = next((v.column for v in vals), None)
            if outcome is None and target_nums:
                outcome = target_nums[0]
            if outcome is None:
                only = [c for c, v in sem.items() if v.get("semantic") == "outcome"]
                if len(only) == 1:
                    outcome = only[0]               # the dataset has exactly one yes/no outcome: that is what is predicted
            plan["target_column"] = outcome or self._unknown(linked)
            inputs = [{"column": v.column, "value": v.value} for v in vals if v.column != plan["target_column"]] + input_vals
            if inputs:
                plan["input_values"] = inputs
            return plan
        if op in ("predict_value",):
            plan["target_column"] = (target_nums[0] if target_nums else None) or self._unknown(linked)
            inputs = [{"column": v.column, "value": v.value} for v in vals] + input_vals
            if inputs:
                plan["input_values"] = inputs
            return plan
        if op == "run_scenario":
            plan["target_column"] = target_nums[0] if target_nums else None
            if scen:
                plan["scenario_changes"] = scen
            return plan
        # ---- columns after "by/per/across" are dimensions; columns before it are measures
        by_pos = next((i for i, t in enumerate(linked.tokens) if t in ("by", "per", "across", "vs", "versus")), None)
        if by_pos is not None and op in ("compare_groups", "pivot_table"):
            after = [m.column for m in cols if m.start > by_pos and roles.get(m.column) in ("categorical", "binary")]
            before = [m.column for m in cols if m.start < by_pos and roles.get(m.column) in ("categorical", "binary") and m.column not in after]
            if after:
                cats = after                           # words before "by" describe the measure, not a grouping
                if before and not target_nums and sem.get(before[0], {}).get("semantic") in ("outcome", "flag"):
                    plan["target_column"] = before[0]
        # ---- filters from category values (unless the value's column is the grouping dimension)
        group = [c for c in cats if c not in val_cols and c != plan.get("target_column")]
        if op in ("compare_groups", "pivot_table") and not group and cats:
            group = cats
        fl = self._value_filters(vals, exclude=set(group) if op in ("compare_groups", "pivot_table") else set())
        fl += filters
        if fl:
            plan["filters"] = fl
        if op == "compare_groups":
            rate_words = re.search(r"\b(rate|rates|share|percentage|percent|proportion|ratio|likelihood)\b", q)
            outcome_cats = [c for c in group if sem.get(c, {}).get("semantic") in ("outcome", "flag")]
            if rate_words and outcome_cats and len(group) >= 2:
                # "attrition rate by department": the outcome is the measure, not a second grouping
                plan["target_column"] = outcome_cats[0]
                group = [c for c in group if c != outcome_cats[0]]
            if rate_words and not plan.get("target_column") and not target_nums:
                only = [c for c, v in sem.items() if v.get("semantic") == "outcome" and c not in group]
                if len(only) == 1:
                    plan["target_column"] = only[0]
            if not group and vals and len({v.column for v in vals}) == 1 and ("rate" in q or "share" in q or "percentage" in q):
                # "rate of Yes by department" -> outcome column as the measure
                plan["target_column"] = vals[0].column
                plan.pop("filters", None)
            plan["group_by"] = group[:2] or None
            tgt = plan.get("target_column") or (target_nums[0] if target_nums else (cats[1] if len(cats) > 1 and len(group) < len(cats) else None))
            if tgt is None and not re.search(r"how many|number of|count|most (records|rows|orders|entries)|most common|largest|biggest", q):
                unk = self._unknown(linked, allow_none=True)
                if unk:
                    tgt = unk
            plan["target_column"] = tgt
            if tgt in nums or (tgt and tgt in df.columns and pd.api.types.is_numeric_dtype(df[tgt])):
                plan["aggregation"] = agg if agg in ("mean", "sum", "median", "min", "max", "count", "nunique", "std") and agg not in ("max", "min") else default_aggregation(sem, tgt)
                if agg in ("max", "min") and not any(w in q for w in ("highest", "lowest", "most", "least")):
                    plan["aggregation"] = agg
            plan.setdefault("sort_order", "desc")
            return plan
        if op == "pivot_table":
            plan["group_by"] = group[:1] or None
            if len(group) > 1:
                plan["pivot_column"] = group[1]
            if target_nums:
                plan["target_column"] = target_nums[0]
            if plan.get("target_column") in target_nums:
                plan["aggregation"] = agg if agg in ("mean", "sum", "median", "count") else default_aggregation(sem, plan["target_column"])
            return plan
        if op == "get_summary":
            if target_nums:
                plan["target_column"] = target_nums[0]
                plan["aggregation"] = agg if agg not in ("distribution", "percentage") else "mean"
                if agg is None:
                    plan["aggregation"] = None
            elif cats:
                plan["target_column"] = cats[0]
                plan["aggregation"] = "nunique" if agg == "nunique" else ("percentage" if agg == "percentage" else "distribution")
            elif vals and not filters:
                plan["target_column"] = vals[0].column
                plan["aggregation"] = "distribution"
                plan.pop("filters", None)
            else:
                unk = self._unknown(linked, allow_none=True)
                if unk and agg in ("mean", "sum", "median", "max", "min", "std"):
                    plan["target_column"] = unk
                    plan["aggregation"] = agg
                else:
                    plan["aggregation"] = "count" if agg == "count" else None
            return plan
        if op in ("analyze_trend", "forecast", "explain_change", "get_current_state", "detect_anomalies", "query_records", "run_eda", "calculate_correlation", "describe_relationships"):
            tgt = target_nums[0] if target_nums else None
            if op in ("run_eda", "describe_relationships", "calculate_correlation") and tgt is None and cats:
                tgt = cats[0] if op != "calculate_correlation" or not vals else None
            if op in ("analyze_trend", "forecast", "explain_change") and tgt is None and not re.search(r"how many|number of|count|records|orders|rows", q):
                tgt = self._unknown(linked, allow_none=True)
            plan["target_column"] = tgt
            if op in ("analyze_trend", "forecast", "explain_change") and tgt and tgt in df.columns:
                plan["aggregation"] = agg if agg in ("mean", "sum", "median", "count") else default_aggregation(sem, tgt)
            if op == "calculate_correlation" and len(target_nums) > 1:
                plan["feature_columns"] = target_nums[1:]
            if op in ("analyze_trend", "get_current_state", "explain_change") and cats and not (op == "explain_change" and not group):
                plan["group_by"] = [c for c in cats if c not in val_cols][:1] or None
                if op == "explain_change":
                    plan["group_by"] = [c for c in cats if c not in val_cols][:4] or None
            freq = self._freq(q)
            if freq and op in ("analyze_trend", "forecast", "explain_change"):
                plan["frequency"] = freq
            if op == "forecast":
                h = self._horizon(q)
                if h:
                    plan["frequency"], plan["forecast_horizon"] = h
            if op == "query_records" and target_nums:
                plan.setdefault("sort_order", "desc")
            return plan
        if op == "segment_records":
            if len(target_nums) >= 2:
                plan["feature_columns"] = target_nums
            m = re.search(r"\b(\d{1,2})\s+(?:segments|clusters|groups|personas)", q)
            if m and 2 <= int(m.group(1)) <= 10:
                plan["n_clusters"] = int(m.group(1))
            return plan
        if op == "generate_report":
            fn = next((v for k, v in sorted(FUNCTION_WORDS.items(), key=lambda kv: -len(kv[0])) if re.search(rf"\b{k}\b", q)), "general")
            plan["business_function"] = fn
            if cats or target_nums:
                plan["target_column"] = (target_nums or cats)[0] if fn in ("general",) else None
            return plan
        if op == "assess_data_quality":
            return plan
        if op == "unsupported":
            plan["reasoning"] = "The question is not about analysing the uploaded data."
        return plan

    def _agg(self, q: str) -> Optional[str]:
        for w, a in AGG_CUES:
            if re.search(rf"\b{re.escape(w)}\b", q):
                return a
        return None

    def _freq(self, q: str) -> Optional[str]:
        for w, f in FREQ_WORDS:
            if re.search(rf"\b(per|by|each|every|a|over the|over|next|coming)?\s*{w}s?\b", q) and re.search(rf"\b{w}", q):
                return f
        return None

    def _horizon(self, q: str) -> Optional[Tuple[str, int]]:
        m = re.search(r"\b(?:next|coming|following|upcoming)\s+(\d{1,3})\s+(hour|day|week|month|quarter|year)s?\b", q)
        if m:
            unit = {"hour": "H", "day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}[m.group(2)]
            n = int(m.group(1))
            return ("M", 12 * n) if unit == "Y" else (unit, n)
        m = re.search(r"\b(?:next|coming)\s+(day|week|month|quarter|year)\b", q)
        if m:
            native = getattr(self, "_native", None) or "D"
            unit = m.group(1)
            if native == "D":
                return {"day": ("D", 1), "week": ("D", 7), "month": ("D", 30), "quarter": ("W", 13), "year": ("M", 12)}[unit]
            if native == "W":
                return {"day": ("W", 1), "week": ("W", 1), "month": ("W", 4), "quarter": ("W", 13), "year": ("M", 12)}[unit]
            return {"day": ("M", 1), "week": ("M", 1), "month": ("M", 1), "quarter": ("M", 3), "year": ("M", 12)}[unit]
        return None

    def _dates(self, q: str, plan: Dict[str, Any], profile: Dict[str, Any], op: str) -> None:
        if not profile.get("primary_time_column"):
            return
        years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", q)]
        if op == "explain_change" and len(years) >= 2:
            a, b = sorted(years[:2])
            plan.update(baseline_start=f"{a}-01-01", baseline_end=f"{a}-12-31", comparison_start=f"{b}-01-01", comparison_end=f"{b}-12-31")
            return
        if len(years) == 1 and op not in ("forecast",):
            plan["start_date"], plan["end_date"] = f"{years[0]}-01-01", f"{years[0]}-12-31"
        elif len(years) >= 2 and "between" in q and op not in ("forecast",):
            a, b = sorted(years[:2])
            plan["start_date"], plan["end_date"] = f"{a}-01-01", f"{b}-12-31"
        end = pd.to_datetime((profile.get("date_range") or {}).get("end"), errors="coerce")
        if pd.notna(end) and op not in ("forecast", "explain_change"):
            m = re.search(r"\blast\s+(\d{1,3})?\s*(day|week|month|quarter|year)s?\b", q)
            if m:
                n = int(m.group(1) or 1)
                days = {"day": 1, "week": 7, "month": 30.44, "quarter": 91.3, "year": 365.25}[m.group(2)] * n
                plan["start_date"] = (end - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
                plan["end_date"] = end.strftime("%Y-%m-%d")

    def _value_filters(self, vals: List[Mention], exclude: set) -> List[Dict[str, Any]]:
        by_col: Dict[str, List[str]] = {}
        for v in vals:
            if v.column in exclude:
                continue
            by_col.setdefault(v.column, []).append(v.value)
        out = []
        for c, vs in by_col.items():
            vs = list(dict.fromkeys(vs))
            out.append({"column": c, "operator": "eq", "value": vs[0]} if len(vs) == 1 else {"column": c, "operator": "in", "values": vs})
        return out

    def _numeric_conditions(self, q, linked, roles, op) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
        """'sales above 5000', 'age between 30 and 40', 'for distance 12' (prediction input)."""
        filters, used, inputs = [], [], []
        toks = linked.tokens
        for m in [m for m in linked.mentions if m.kind == "column" and roles.get(m.column) == "numeric"]:
            tail = " ".join(toks[m.end:m.end + 6])
            head = " ".join(toks[max(0, m.start - 4):m.start])
            mm = re.match(rf"(?:is |was |are |of |=\s*)?(?:between|from)\s+{NUM}\s+(?:and|to)\s+{NUM}", tail)
            if mm:
                filters.append({"column": m.column, "operator": "between", "value": mm.group(1).replace(",", ""), "value2": mm.group(2).replace(",", "")})
                used.append(m.column)
                continue
            mm = re.match(rf"(?:is |was |are |of )?(above|over|greater than|more than|higher than|exceeding|at least|>=|>|below|under|less than|lower than|at most|<=|<|equal to|=)\s*{NUM}", tail)
            if mm:
                opw = mm.group(1)
                opmap = {"above": "gt", "over": "gt", "greater than": "gt", "more than": "gt", "higher than": "gt", "exceeding": "gt", ">": "gt", "at least": "gte", ">=": "gte",
                         "below": "lt", "under": "lt", "less than": "lt", "lower than": "lt", "<": "lt", "at most": "lte", "<=": "lte", "equal to": "eq", "=": "eq"}
                if op in ("predict_value", "predict_class") and opmap[opw] == "eq":
                    inputs.append({"column": m.column, "value": mm.group(2).replace(",", "")})
                else:
                    filters.append({"column": m.column, "operator": opmap[opw], "value": mm.group(2).replace(",", "")})
                used.append(m.column)
                continue
            if op in ("predict_value", "predict_class"):
                mm = re.match(rf"(?:of |is |= |was |at )?{NUM}\b", tail)
                hb = re.search(rf"{NUM}\s*(?:km|kms|years?|yrs?|units?|%)?\s*(?:of\s+)?$", head)
                if mm:
                    inputs.append({"column": m.column, "value": mm.group(1).replace(",", "")})
                    used.append(m.column)
                elif hb and not re.search(r"\btop\b", head):
                    inputs.append({"column": m.column, "value": hb.group(1).replace(",", "")})
                    used.append(m.column)
        return filters, used, inputs

    def _scenario(self, q, linked, roles) -> List[Dict[str, Any]]:
        out = []
        toks = linked.tokens
        up = r"increase|increases|increased|rise|rises|rose|raise|raised|grow|grows|grew|boost|go up|goes up|went up|gone up|up|higher|double|doubles"
        down = r"decrease|decreases|decreased|drop|drops|dropped|fall|falls|fell|cut|cuts|reduce|reduces|reduced|lower|lowered|go down|goes down|went down|gone down|down|decline|declines"
        for m in [m for m in linked.mentions if m.kind == "column" and roles.get(m.column) == "numeric"]:
            after = " ".join(toks[m.end:m.end + 6])                     # "cost rises by 15%"
            before = " ".join(toks[max(0, m.start - 3):m.start])        # "increase cost by 15%"
            pm = re.match(rf"(?:were to |was to |would |will |)({up}|{down})\b\D{{0,12}}?{NUM}\s*(%|percent|per cent|pct)?", after)
            if not pm and re.search(rf"\b({up}|{down})\s*(?:the\s+)?$", before):
                vb = re.search(rf"\b({up}|{down})\s*(?:the\s+)?$", before).group(1)
                nm = re.match(rf"\D{{0,12}}?{NUM}\s*(%|percent|per cent|pct)?", after)
                pm = (vb, nm.group(1), nm.group(2)) if nm else None
            elif pm:
                pm = (pm.group(1), pm.group(2), pm.group(3))
            if not pm:
                continue
            word, val, pct = pm[0], float(pm[1].replace(",", "")), pm[2]
            sign = -1 if re.fullmatch(down, word) else 1
            if "double" in word:
                out.append({"column": m.column, "change_type": "percent", "value": 100.0})
            elif pct:
                out.append({"column": m.column, "change_type": "percent", "value": sign * val})
            else:
                out.append({"column": m.column, "change_type": "absolute", "value": sign * val})
            break
        if not out:
            for m in [m for m in linked.mentions if m.kind == "column" and roles.get(m.column) == "numeric"]:
                tail = " ".join(toks[m.end:m.end + 4])
                sm = re.match(rf"(?:were|was|is|becomes|=|at)\s+{NUM}", tail)
                if sm:
                    out.append({"column": m.column, "change_type": "set", "value": float(sm.group(1).replace(",", ""))})
                    break
        if not out:
            pm = re.search(rf"([+-]\d+(?:\.\d+)?)\s*(%|percent)", q)
            nums = [m for m in linked.mentions if m.kind == "column" and roles.get(m.column) == "numeric"]
            if pm and nums:
                out.append({"column": nums[-1].column, "change_type": "percent", "value": float(pm.group(1))})
        return out

    def _unknown(self, linked: Linked, allow_none: bool = False) -> Optional[str]:
        """The user's own words for a variable that is not a column (passed on so the validator can reject it)."""
        q = " ".join(linked.tokens)
        m = re.search(r"\b(?:average|mean|total|sum of|median|maximum|minimum|predict|forecast|trend of|trend in|estimate|distribution of|anomalies in|outliers in|of|in)\s+(?:the\s+)?([a-z][a-z ]{2,30}?)(?:\s+(?:by|for|per|in|across|over|next|of|$)|\?|$)", q)
        if m:
            words = [w for w in m.group(1).split() if w not in STOP and w not in ("value", "values", "data", "dataset", "records", "rows", "next", "time")]
            if words:
                return " ".join(words[:2])
        cand = [t for t in linked.unknown_terms if t not in ("average", "total", "predict", "forecast", "highest", "lowest", "compare", "trend", "which", "outliers")]
        if cand and not allow_none:
            return cand[0]
        return None

    def _is_follow_up(self, q, cols, vals, op, conf) -> bool:
        if len(q.split()) <= 6 and (vals or cols) and (FOLLOW_UP.search(q) or (not cols and vals)):
            return True
        if not cols and not vals and FOLLOW_UP.search(q) and re.search(r"\b(next|instead|again|same|that|it|those)\b", q):
            return True
        return False

    def _follow_up(self, base: Dict[str, Any], q, cols, vals, roles, sem, df, profile) -> Dict[str, Any]:
        plan = {k: v for k, v in base.items() if k not in ("intent", "reasoning", "needs_clarification", "clarification_question")}
        new_cats = [m.column for m in cols if roles.get(m.column) in ("categorical", "binary")]
        new_nums = [m.column for m in cols if roles.get(m.column) == "numeric"]
        if re.search(r"\b(by|per|across|split|break)\b", q) and new_cats:
            plan["group_by"] = new_cats[:1] if plan.get("operation") != "compare_groups" or not plan.get("group_by") else new_cats[:2]
            if plan.get("operation") in ("get_summary", "analyze_trend") and plan.get("operation") == "get_summary":
                plan["operation"] = "compare_groups"
        if new_nums and not re.search(r"\b(by|per)\b", q):
            plan["target_column"] = new_nums[0]
        if vals:
            fl = [f for f in (plan.get("filters") or []) if f.get("column") not in {v.column for v in vals}]
            fl += self._value_filters(vals, exclude=set())
            plan["filters"] = fl
        if re.search(r"\b(all|everything|remove the filter|without filter|overall)\b", q):
            plan["filters"] = None
        h = self._horizon(q)
        if h and plan.get("operation") == "forecast":
            plan["frequency"], plan["forecast_horizon"] = h
        if re.search(r"\b(lowest|least|worst|bottom)\b", q):
            plan["sort_order"] = "asc"
        elif re.search(r"\b(highest|most|best|top)\b", q):
            plan["sort_order"] = "desc"
        chart = next((c for w, c in CHART_WORDS if w in q), None)
        if chart:
            plan["chart_type"] = chart
        plan["uses_previous_context"] = True
        plan["intent"] = q
        plan["reasoning"] = "Follow-up: carried over the previous plan and applied the new constraint."
        return plan

    def _fill_from_answer(self, base, cols, vals, roles) -> Optional[Dict[str, Any]]:
        plan = {k: v for k, v in base.items() if k not in ("needs_clarification", "clarification_question")}
        op = plan.get("operation")
        if cols:
            c = cols[0].column
            if op in ("compare_groups", "pivot_table") and not plan.get("group_by") and roles.get(c) in ("categorical", "binary"):
                plan["group_by"] = [c]
            elif op == "pivot_table" and plan.get("group_by") and not plan.get("pivot_column"):
                plan["pivot_column"] = c
            elif not plan.get("target_column"):
                plan["target_column"] = c
            else:
                plan["target_column"] = c
            return plan
        if vals:
            plan["filters"] = (plan.get("filters") or []) + self._value_filters(vals, set())
            return plan
        return None
