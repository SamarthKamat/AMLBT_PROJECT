"""Multi-agent analytics layer (Phase 2).

Agents do not compute numbers themselves and never run model-written code. Each agent calls TOOLS, and every tool
is one of UDI's controlled operations, invoked through the same validation gate as a chat question:

    plan (dict) --validate_plan--> ValidatedPlan --execute--> verified result

  DataAnalystAgent      profile, data-quality score, column relationships
  BusinessAnalystAgent  KPIs for the business function, segment comparisons, margins, trends, root cause
  MLAgent               driver models (classification / regression with SHAP), segmentation, anomalies, forecast
  ReportAgent           assembles sections, ranks findings, writes the executive summary from verified numbers

Every tool call is recorded in an AgentTrace (agent, tool, plan, status, seconds, headline) that the UI shows.
A tool that is not applicable or fails is recorded as skipped with its reason; the report never invents a section.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from artifacts import get_quality, get_relationships, get_semantics
from business_analyst import business_insights
from context import ExecContext
from data_quality import public
from errors import UDIError
from result_formatter import fmt_num, jsonsafe
from schemas import AnalysisPlan
from semantics import dimension_columns, kpi_columns, outcome_columns

FUNCTION_LABEL = {"general": "General business", "sales": "Sales", "marketing": "Marketing", "hr": "Human Resources",
                  "operations": "Operations", "finance": "Finance", "customer": "Customer"}


@dataclass
class AgentTrace:
    steps: List[Dict[str, Any]] = field(default_factory=list)
    progress: Optional[Callable[[float, str], None]] = None

    def log(self, agent: str, tool: str, status: str, seconds: float, headline: str = "", plan: Optional[Dict[str, Any]] = None):
        self.steps.append({"agent": agent, "tool": tool, "status": status, "seconds": round(seconds, 2), "headline": headline[:300],
                           "plan": {k: v for k, v in (plan or {}).items() if v not in (None, [], "")}})


class ToolBox:
    """Runs a controlled operation for an agent: validation first, then execution. Never raises to the agent."""

    def __init__(self, df: pd.DataFrame, ctx: ExecContext, trace: AgentTrace):
        self.df, self.ctx, self.trace = df, ctx, trace

    def run(self, agent: str, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        from executor import execute                 # local import: executor imports this module's users
        from validator import ClarificationNeeded, validate_plan
        t0 = time.perf_counter()
        try:
            vp = validate_plan(AnalysisPlan(**plan), self.df, self.ctx.profile, self.ctx.roles)
            res = execute(vp, self.ctx)
            self.trace.log(agent, plan["operation"], "ok", time.perf_counter() - t0, res.get("summary", ""), plan)
            return res
        except ClarificationNeeded as c:
            self.trace.log(agent, plan["operation"], "skipped", time.perf_counter() - t0, f"needs clarification: {c.question}", plan)
        except UDIError as e:
            self.trace.log(agent, plan["operation"], "skipped", time.perf_counter() - t0, f"{e.code}: {e.message}", plan)
        except Exception as e:                       # pragma: no cover - defensive
            self.trace.log(agent, plan["operation"], "failed", time.perf_counter() - t0, f"{type(e).__name__}: {e}", plan)
        return None


def _section(title: str, agent: str, res: Optional[Dict[str, Any]], max_charts: int = 2, intro: str = "") -> Optional[Dict[str, Any]]:
    if not res:
        return None
    charts = [c for c in (res.get("charts") or []) if c.get("type") != "scatter" or len(c.get("data") or []) <= 800][:max_charts]
    return {"title": title, "agent": agent, "operation": res.get("operation"), "summary": (intro + " " if intro else "") + res.get("summary", ""),
            "metrics": dict(list((res.get("metrics") or {}).items())[:6]), "charts": charts, "table": (res.get("table") or [])[:12],
            "table_title": res.get("table_title"), "insights": (res.get("insights") or [])[:4], "limitations": (res.get("limitations") or [])[:2]}


def _find(cols: List[str], words: List[str]) -> Optional[str]:
    for w in words:
        for c in cols:
            if w in str(c).lower().replace(" ", "_"):
                return c
    return None


# ----------------------------------------------------------------------------- agents
class DataAnalystAgent:
    name = "Data Analyst Agent"

    def run(self, df: pd.DataFrame, ctx: ExecContext, trace: AgentTrace) -> Dict[str, Any]:
        t0 = time.perf_counter()
        q = public(get_quality(ctx, df))
        trace.log(self.name, "assess_data_quality", "ok", time.perf_counter() - t0, q["summary"])
        t1 = time.perf_counter()
        rel = get_relationships(ctx, df)
        trace.log(self.name, "describe_relationships", "ok", time.perf_counter() - t1, (rel["findings"] or [""])[0])
        return {"quality": q, "relationships": rel}


class BusinessAnalystAgent:
    name = "Business Analyst Agent"

    def kpis(self, df: pd.DataFrame, sem: Dict[str, Any], function: str, outcomes: List[str]) -> List[Dict[str, Any]]:
        cards = [{"label": "Records", "value": int(len(df)), "note": "rows analysed"}]
        for c in kpi_columns(sem, function, limit=4):
            x = df[c].dropna().astype(float)
            if x.empty:
                continue
            if sem[c]["additive"]:
                cards.append({"label": f"Total {c}", "value": float(x.sum()), "note": f"average {fmt_num(float(x.mean()))} per record"})
            else:
                cards.append({"label": f"Average {c}", "value": float(x.mean()), "note": f"median {fmt_num(float(x.median()))}"})
        for o in outcomes[:1]:
            vc = df[o].dropna().astype(str).value_counts(normalize=True)
            pos = _positive_label(vc.index.tolist())
            if pos is not None:
                cards.append({"label": f"{o} rate ('{pos}')", "value": float(vc.get(pos, 0) * 100), "note": "% of records", "unit": "%"})
        money = [c for c in df.columns if sem.get(c, {}).get("semantic") == "money"]
        prof, sales = _find(money, ["profit"]), _find(money, ["sales", "revenue", "turnover"])
        if prof and sales and df[sales].sum():
            cards.append({"label": "Profit margin", "value": float(df[prof].sum() / df[sales].sum() * 100), "note": f"sum({prof}) / sum({sales})", "unit": "%"})
        return cards

    def margin_section(self, df: pd.DataFrame, sem: Dict[str, Any], dims: List[str], trace: AgentTrace) -> Optional[Dict[str, Any]]:
        money = [c for c in df.columns if sem.get(c, {}).get("semantic") == "money"]
        prof, sales = _find(money, ["profit"]), _find(money, ["sales", "revenue", "turnover"])
        if not (prof and sales and dims):
            return None
        t0 = time.perf_counter()
        g = dims[0]
        agg = df.groupby(df[g].astype(str)).agg(sales=(sales, "sum"), profit=(prof, "sum"), records=(sales, "size"))
        agg = agg[agg["sales"] != 0]
        agg["margin_%"] = agg["profit"] / agg["sales"] * 100
        agg = agg.sort_values("margin_%")
        overall = float(df[prof].sum() / df[sales].sum() * 100)
        med = float(agg["margin_%"].median())
        below = agg[agg["margin_%"] < med]
        uplift = float(((med - below["margin_%"]) / 100 * below["sales"]).sum())
        loss = agg[agg["profit"] < 0]
        rows = [{g: k, "sales": float(r["sales"]), "profit": float(r["profit"]), "margin_%": float(r["margin_%"]), "records": int(r["records"])} for k, r in agg.iterrows()]
        summary = (f"Overall margin is {overall:.1f}%. By {g}, margins range from {agg['margin_%'].iloc[0]:.1f}% ('{agg.index[0]}') to {agg['margin_%'].iloc[-1]:.1f}% ('{agg.index[-1]}').")
        insights = []
        if len(loss):
            insights.append(f"{len(loss)} {g} value(s) are loss-making in total: " + ", ".join(f"'{k}' ({fmt_num(float(r['profit']))})" for k, r in loss.head(4).iterrows()) + ".")
        rec = {"finding": f"{len(below)} {g} group(s) run below the median margin of {med:.1f}%.",
               "action": f"Review pricing, discounts and cost-to-serve for {', '.join(repr(k) for k in below.index[:3])}.",
               "evidence": f"Lowest margin: '{agg.index[0]}' at {agg['margin_%'].iloc[0]:.1f}% on {fmt_num(float(agg['sales'].iloc[0]))} {sales}.",
               "priority": "high", "impact_estimate": f"Illustrative: lifting these groups to the median margin on the same {sales} would add about {fmt_num(uplift)} {prof} (not a forecast; assumes volume unchanged)."}
        trace.log(self.name, "margin_analysis", "ok", time.perf_counter() - t0, summary)
        return {"section": {"title": f"Profitability by {g}", "agent": self.name, "operation": "margin_analysis", "summary": summary,
                            "metrics": {"Overall margin %": overall, "Median group margin %": med, "Loss-making groups": int(len(loss))},
                            "charts": [{"type": "bar", "title": f"Profit margin % by {g}", "x": g, "y": "margin_%", "data": rows[:15], "horizontal": len(rows) > 6}],
                            "table": rows[:12], "table_title": f"Sales, profit and margin by {g}", "insights": insights, "limitations": ["Margins are accounting ratios from the data; they do not show why margins differ."]},
                "recommendation": rec}

    def outcome_gap(self, df: pd.DataFrame, outcome: str, dims: List[str]) -> Optional[Dict[str, Any]]:
        vc = df[outcome].dropna().astype(str)
        pos = _positive_label(sorted(vc.unique().tolist()))
        if pos is None:
            return None
        best = None
        for g in dims:
            if g == outcome:
                continue
            r = (df[outcome].astype(str) == pos).groupby(df[g].astype(str)).agg(["mean", "size"])
            r = r[r["size"] >= 20]
            if len(r) < 2:
                continue
            hi, lo = r["mean"].idxmax(), r["mean"].idxmin()
            gap = r.loc[hi, "mean"] - r.loc[lo, "mean"]
            if best is None or gap > best["gap"]:
                best = {"dim": g, "hi": hi, "lo": lo, "gap": float(gap), "rate_hi": float(r.loc[hi, "mean"]), "rate_lo": float(r.loc[lo, "mean"]), "n_hi": int(r.loc[hi, "size"])}
        if not best or best["gap"] < 0.05:
            return None
        avoided = best["n_hi"] * best["gap"]
        return {"finding": f"{outcome} = '{pos}' is {best['rate_hi']:.1%} for {best['dim']} = '{best['hi']}' versus {best['rate_lo']:.1%} for '{best['lo']}'.",
                "action": f"Target retention / conversion actions at {best['dim']} = '{best['hi']}' first and test what drives the gap.",
                "evidence": f"Gap of {best['gap'] * 100:.1f} percentage points ({best['n_hi']:,} records in '{best['hi']}').", "priority": "high",
                "impact_estimate": f"Illustrative: closing the gap would change about {avoided:,.0f} outcome(s) in this dataset (association, not a causal estimate)."}


def _rank_dimensions(df: pd.DataFrame, dims: List[str], outcome: Optional[str], kpi: Optional[str]) -> List[str]:
    """Order candidate dimensions by how strongly they relate to the outcome (Cramer's V) or the main KPI (eta)."""
    from relationships import correlation_ratio, cramers_v
    scored = []
    for g in dims:
        try:
            if outcome and outcome != g:
                v = cramers_v(df[g], df[outcome])
            elif kpi:
                v = correlation_ratio(df[g], df[kpi])
            else:
                v = None
        except Exception:
            v = None
        scored.append((g, -1.0 if v is None else v))
    return [g for g, _ in sorted(scored, key=lambda t: -t[1])]


def _positive_label(labels: List[str]) -> Optional[str]:
    low = {str(l).strip().lower(): l for l in labels}
    for cand in ("yes", "1", "true", "churned", "late", "left", "y", "default", "high"):
        if cand in low:
            return low[cand]
    return labels[-1] if len(labels) == 2 else None


class MLAgent:
    name = "ML Agent"


class ReportAgent:
    name = "Report Agent"


# ----------------------------------------------------------------------------- orchestrator
def build_report(df: pd.DataFrame, ctx: ExecContext, function: str = "general", dataset_name: str = "dataset",
                 progress: Optional[Callable[[float, str], None]] = None, target: Optional[str] = None) -> Dict[str, Any]:
    t_start = time.perf_counter()
    function = function if function in FUNCTION_LABEL else "general"
    trace = AgentTrace(progress=progress)
    tools = ToolBox(df, ctx, trace)
    sem = get_semantics(ctx, df)

    def step(frac: float, msg: str):
        if progress:
            progress(frac, msg)

    step(0.05, "Data Analyst Agent: profiling and data quality")
    da = DataAnalystAgent().run(df, ctx, trace)
    ba = BusinessAnalystAgent()
    outcomes = outcome_columns(sem, function)
    kpis = kpi_columns(sem, function, limit=3)
    if target and target in df.columns:
        if sem.get(target, {}).get("semantic") in ("outcome", "flag") or ctx.roles.get(target, {}).get("role") in ("categorical", "binary"):
            outcomes = [target] + [o for o in outcomes if o != target]
        else:
            kpis = [target] + [k for k in kpis if k != target]
    dims = dimension_columns(sem, ctx.profile, function, max_unique=30, limit=8)
    dims = [d for d in dims if d not in outcomes[:1]] or dims
    dims = _rank_dimensions(df, dims, outcomes[0] if outcomes else None, kpis[0] if kpis else None)[:5]
    trace.log(ba.name, "dimension_selection", "ok", 0.0, "Dimensions ranked by association with " + (outcomes[0] if outcomes else (kpis[0] if kpis else "records")) + ": " + ", ".join(dims))
    tc = ctx.profile.get("primary_time_column")
    sections: List[Dict[str, Any]] = []
    recs: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []

    def add(title, agent, res, **kw):
        s = _section(title, agent, res, **kw)
        if s:
            sections.append(s)
            results.append(res)
            plan = {"target_column": (res.get("provenance") or {}).get("target"), "aggregation": (res.get("provenance") or {}).get("aggregation"),
                    "sort_order": "desc"}
            recs.extend(business_insights(res, plan, df, sem))

    step(0.15, "Business Analyst Agent: KPIs")
    cards = ba.kpis(df, sem, function, outcomes)
    trace.log(ba.name, "kpi_selection", "ok", 0.0, ", ".join(c["label"] for c in cards))

    step(0.25, "Business Analyst Agent: segment comparisons")
    for k in kpis[:2]:
        agg = "sum" if sem[k]["additive"] else "mean"
        for g in dims[:2]:
            add(f"{'Total' if agg == 'sum' else 'Average'} {k} by {g}", ba.name,
                tools.run(ba.name, {"operation": "compare_groups", "target_column": k, "group_by": [g], "aggregation": agg, "sort_order": "desc"}), max_charts=1)
    if function in ("sales", "finance", "general", "operations"):
        m = ba.margin_section(df, sem, dims, trace)
        if m:
            sections.append(m["section"])
            recs.append(m["recommendation"])
    if outcomes:
        o = outcomes[0]
        for g in dims[:3]:
            add(f"{o} by {g}", ba.name, tools.run(ba.name, {"operation": "compare_groups", "target_column": o, "group_by": [g]}), max_charts=1)
        gap = ba.outcome_gap(df, o, dims)
        if gap:
            recs.append(gap)
    if len(dims) >= 2 and kpis:
        add(f"{kpis[0]} by {dims[0]} and {dims[1]}", ba.name, tools.run(ba.name, {"operation": "pivot_table", "target_column": kpis[0], "group_by": [dims[0]],
                                                                                  "pivot_column": dims[1], "aggregation": "sum" if sem[kpis[0]]["additive"] else "mean"}), max_charts=1)

    if tc and kpis:
        step(0.45, "Business Analyst Agent: trends and root cause")
        k = kpis[0]
        agg = "sum" if sem[k]["additive"] else "mean"
        add(f"Trend of {k}", ba.name, tools.run(ba.name, {"operation": "analyze_trend", "target_column": k, "time_column": tc, "aggregation": agg}), max_charts=1)
        add(f"What changed in {k} recently", ba.name, tools.run(ba.name, {"operation": "explain_change", "target_column": k, "time_column": tc, "aggregation": agg,
                                                                          "group_by": dims[:4] or None}), max_charts=1)

    ml = MLAgent()
    step(0.6, "ML Agent: driver model and explanations")
    if outcomes:
        add(f"What predicts {outcomes[0]} (classification + SHAP)", ml.name, tools.run(ml.name, {"operation": "predict_class", "target_column": outcomes[0]}), max_charts=3)
    elif kpis:
        add(f"What drives {kpis[0]} (regression + SHAP)", ml.name, tools.run(ml.name, {"operation": "predict_value", "target_column": kpis[0]}), max_charts=3)
    if function in ("marketing", "customer", "hr", "general"):
        step(0.72, "ML Agent: segmentation")
        add("Segments", ml.name, tools.run(ml.name, {"operation": "segment_records"}), max_charts=2)
    if kpis:
        step(0.8, "ML Agent: anomalies")
        add(f"Unusual {kpis[0]} values", ml.name, tools.run(ml.name, {"operation": "detect_anomalies", "target_column": kpis[0]}), max_charts=1)
    if tc and kpis and function in ("sales", "operations", "finance", "general"):
        step(0.86, "ML Agent: forecast")
        k = kpis[0]
        add(f"Forecast of {k}", ml.name, tools.run(ml.name, {"operation": "forecast", "target_column": k, "time_column": tc, "aggregation": "sum" if sem[k]["additive"] else "mean"}), max_charts=1)

    step(0.93, "Report Agent: assembling")
    q = da["quality"]
    rel = da["relationships"]
    if q["score"] < 85:
        recs.append({"finding": f"Data quality score is {q['score']:.0f}/100 (grade {q['grade']}).", "action": "Fix the high-severity data issues before acting on the findings.",
                     "evidence": q["summary"], "priority": "high"})
    rank = {"high": 0, "medium": 1, "low": 2}
    seen, uniq_recs = set(), []
    for r in sorted(recs, key=lambda r: rank.get(r.get("priority", "medium"), 1)):
        if r["finding"] not in seen:
            seen.add(r["finding"])
            uniq_recs.append(r)
    exec_summary = [f"{FUNCTION_LABEL[function]} report on {len(df):,} records and {df.shape[1]} columns; data quality {q['score']:.0f}/100 (grade {q['grade']})."]
    for c in cards[1:4]:
        exec_summary.append(f"{c['label']}: {fmt_num(c['value'])}{c.get('unit', '')}.")
    exec_summary += [r["finding"] for r in uniq_recs[:4]]
    report = {"function": function, "function_label": FUNCTION_LABEL[function], "title": f"{FUNCTION_LABEL[function]} report — {dataset_name}",
              "generated_at": time.strftime("%Y-%m-%d %H:%M"), "dataset": {"name": dataset_name, "rows": int(len(df)), "columns": int(df.shape[1])},
              "executive_summary": exec_summary, "kpis": cards, "sections": sections, "recommendations": uniq_recs[:12],
              "data_quality": {"score": q["score"], "grade": q["grade"], "dimensions": q["dimensions"], "summary": q["summary"],
                               "top_issues": [{"severity": i["severity"], "issue": i["issue"], "recommendation": i["recommendation"]} for i in q["issues"][:6]]},
              "relationships": rel["findings"][:8], "kpi_columns": kpis, "dimensions": dims, "outcome": outcomes[0] if outcomes else None,
              "limitations": ["All figures are computed by UDI's Python tools on the uploaded data; recommendations are prompts for investigation, not causal conclusions.",
                              "Impact estimates are illustrative arithmetic on historical data (volumes held constant)."],
              "agent_trace": trace.steps, "seconds": round(time.perf_counter() - t_start, 2)}
    step(1.0, "done")
    return jsonsafe(report)
