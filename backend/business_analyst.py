"""Business Analyst agent (Phase 2): turns a VERIFIED analytical result into business findings and suggested actions.

Rules only read numbers that are already in the result (or recompute them from the same filtered data); every
finding carries its evidence, and every action is phrased as something to investigate or consider, because the
underlying analyses are descriptive or predictive, not causal.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from result_formatter import fmt_num

NON_CAUSAL = "Evidence is associational; validate with a controlled test before committing budget."


def _f(finding: str, action: str, evidence: str, priority: str = "medium", impact: Optional[str] = None) -> Dict[str, Any]:
    out = {"finding": finding, "action": action, "evidence": evidence, "priority": priority}
    if impact:
        out["impact_estimate"] = impact
    return out


def _compare(result, plan, data, sem) -> List[Dict[str, Any]]:
    out = []
    t = result.get("table") or []
    m = result.get("metrics") or {}
    if len(t) < 2 or "Top value" not in m:
        return out
    tgt = plan.get("target_column")
    agg = plan.get("aggregation")
    key = next(iter(t[0]))
    measure = [k for k in t[0] if k not in (key, "records", "vs_overall_%", "share_%")][0]
    top, bot = t[0], t[-1]
    if plan.get("sort_order") == "asc":
        top, bot = bot, top
    tv, bv = top[measure], bot[measure]
    if agg in ("sum", "count") and top.get("share_%") is not None:
        # a total mostly reflects how many records a group has: report shares and the per-record average instead of a ratio
        if data is not None and tgt in getattr(data, "columns", []) and key in data.columns:
            per = data.groupby(data[key].astype(str))[tgt].mean()
            pt, pb = per.get(str(top[key])), per.get(str(bot[key]))
            avg_txt = f" Per record the averages are {fmt_num(pt)} vs {fmt_num(pb)}." if pt is not None and pb is not None else ""
        else:
            avg_txt = ""
        out.append(_f(f"'{top[key]}' contributes {top['share_%']:.1f}% of total {tgt or 'records'} versus {bot['share_%']:.1f}% for '{bot[key]}'.",
                      f"Protect and grow '{top[key]}'; check whether '{bot[key]}' is under-served or simply small.",
                      f"{measure}: {fmt_num(tv)} vs {fmt_num(bv)}.{avg_txt}", "medium"))
    elif bv not in (0, None) and tv is not None and np.sign(tv) == np.sign(bv) and abs(tv / bv) >= 1.5:
        out.append(_f(f"'{top[key]}' outperforms '{bot[key]}' on {measure} by {abs(tv / bv):.1f}×.",
                      f"Study what '{top[key]}' does differently and test whether it transfers to '{bot[key]}'.",
                      f"{measure}: {fmt_num(tv)} vs {fmt_num(bv)}.", "high"))
    shares = [r.get("share_%") for r in t if r.get("share_%") is not None]
    if shares:
        top3 = sum(sorted(shares, reverse=True)[:3])
        if top3 >= 60 and len(shares) > 4:
            out.append(_f(f"Concentration risk: the top 3 of {len(shares)} groups account for {top3:.0f}% of {measure}.",
                          "Reduce dependence on a few groups (diversify markets/products) or protect them with retention plans.",
                          f"Shares of the top 3 groups sum to {top3:.1f}%.", "medium"))
    if tgt and sem.get(tgt, {}).get("semantic") == "money" and agg in ("sum", "mean"):
        neg = [r for r in t if (r.get(measure) or 0) < 0]
        if neg:
            names = ", ".join(f"'{r[key]}' ({fmt_num(r[measure])})" for r in neg[:4])
            out.append(_f(f"{len(neg)} group(s) are loss-making on {tgt}: {names}.",
                          "Review pricing, discounting and cost-to-serve for these groups first.",
                          f"Negative {measure} in {len(neg)} of {len(t)} groups.", "high"))
    below = [r for r in t if r.get("vs_overall_%") is not None and r["vs_overall_%"] < -20]
    if below and agg != "sum":
        out.append(_f(f"{len(below)} group(s) are more than 20% below the overall level of {measure}.",
                      f"Prioritise a diagnostic review of {', '.join(repr(r[key]) for r in below[:3])}.",
                      "; ".join(f"{r[key]}: {r['vs_overall_%']:+.0f}% vs overall" for r in below[:3]), "medium"))
    if m.get("Kruskal–Wallis p-value") is not None and m["Kruskal–Wallis p-value"] >= 0.05:
        out.append(_f(f"Differences in {measure} across {key} are not statistically significant.",
                      "Avoid re-allocating resources on the basis of this ranking alone.",
                      f"Kruskal–Wallis p = {m['Kruskal–Wallis p-value']:.3g}.", "low"))
    return out


def _trend(result, plan, sem) -> List[Dict[str, Any]]:
    m = result.get("metrics") or {}
    d = m.get("Direction")
    tgt = plan.get("target_column") or "records"
    if d == "decreasing":
        good_down = sem.get(tgt, {}).get("semantic") in ("duration",) or any(w in str(tgt).lower() for w in ("cost", "complaint", "wait", "delay", "churn", "attrition", "defect"))
        return [_f(f"'{tgt}' shows a statistically significant downward trend ({fmt_num(m.get('Fitted change %'))}% over the period).",
                   "Keep the current course and monitor." if good_down else "Treat as an early warning: investigate drivers by segment (ask 'why did it change?').",
                   f"Mann–Kendall p = {m.get('p-value', 0):.3g}, slope {fmt_num(m.get(next((k for k in m if k.startswith('Slope')), ''), None))}.", "high" if not good_down else "low")]
    if d == "increasing":
        bad_up = any(w in str(tgt).lower() for w in ("cost", "complaint", "wait", "delay", "churn", "attrition", "defect", "expense"))
        return [_f(f"'{tgt}' is rising significantly ({fmt_num(m.get('Fitted change %'))}% over the period).",
                   "Investigate the cost/risk drivers behind the increase." if bad_up else "Plan capacity and inventory for continued growth; confirm with a forecast.",
                   f"Mann–Kendall p = {m.get('p-value', 0):.3g}.", "high" if bad_up else "medium")]
    return []


def _model(result, plan) -> List[Dict[str, Any]]:
    out = []
    ex = result.get("explainability") or {}
    items = (ex.get("shap") or {}).get("items") or ex.get("items") or []
    tgt = plan.get("target_column")
    if items:
        top = items[:3]
        out.append(_f(f"The strongest predictors of '{tgt}' are " + ", ".join(f"{i['feature']} ({i['share_%']:.0f}%)" for i in top) + ".",
                      "Focus monitoring and interventions on these levers; " + NON_CAUSAL,
                      "; ".join(f"{i['feature']}: {i.get('direction', '')}" for i in top if i.get("direction")) or "SHAP / permutation importance on held-out rows.", "high"))
    ev = result.get("evaluation") or {}
    mt = ev.get("metrics") or {}
    if ev.get("task") == "classification" and mt.get("F1") is not None and mt["F1"] < 0.6:
        out.append(_f(f"The model separates the classes of '{tgt}' only moderately (macro F1 {mt['F1']:.2f}).",
                      "Use predictions to rank cases for review, not for automatic decisions.", f"Held-out macro F1 {mt['F1']:.3f}.", "medium"))
    if ev.get("task") == "regression" and mt.get("R2") is not None and mt["R2"] < 0.3:
        out.append(_f(f"'{tgt}' is hard to predict from the available columns (R² {mt['R2']:.2f}).",
                      "Collect additional drivers before using the model for planning.", f"Held-out R² {mt['R2']:.3f}.", "medium"))
    return out


def _anomalies(result) -> List[Dict[str, Any]]:
    m = result.get("metrics") or {}
    k = m.get("Flagged")
    if k:
        return [_f(f"{k:,} records ({fmt_num(m.get('Flagged %'))}%) are statistically unusual.",
                   "Route the flagged records to an owner for review (possible errors, fraud or one-off events).", f"Rule: {m.get('Rule', m.get('Method', ''))}.", "medium")]
    return []


def _forecast(result) -> List[Dict[str, Any]]:
    fc = result.get("forecast") or {}
    periods = fc.get("periods") or []
    ev = result.get("evaluation") or {}
    if not periods:
        return []
    first, last = periods[0]["forecast"], periods[-1]["forecast"]
    chg = (last - first) / abs(first) * 100 if first else None
    out = [_f(f"The forecast moves from {fmt_num(first)} to {fmt_num(last)} over {len(periods)} period(s)" + (f" ({chg:+.1f}%)." if chg is not None else "."),
              "Use the 80% range (not the point forecast) for inventory, staffing and budget buffers.",
              f"Model {ev.get('selected_model')}, backtest RMSE {fmt_num((ev.get('metrics') or {}).get('RMSE'))}.", "medium")]
    sk = ev.get("skill_vs_baseline_%")
    if sk is not None and sk <= 0:
        out.append(_f("The forecasting model does not beat a naive baseline.", "Plan with simple run-rate assumptions and review again with more history.",
                      f"Skill vs baseline {sk:.1f}%.", "medium"))
    return out


def _change(result) -> List[Dict[str, Any]]:
    rc = (result.get("business") or {}).get("root_cause") or []
    m = result.get("metrics") or {}
    if not rc:
        return []
    b = rc[0]
    direction = "decline" if (m.get("Change") or 0) < 0 else "increase"
    return [_f(f"The {direction} is concentrated in {b['dimension']} = '{b['top_group']}' (contribution {fmt_num(b['top_contribution'])}).",
               f"Start the investigation with {b['dimension']} = '{b['top_group']}' (owners, pricing, supply or staffing changes in that period).",
               f"{b['concentration']:.0%} of the gross movement across {b['groups']} groups.", "high")]


def _correlation(result) -> List[Dict[str, Any]]:
    t = result.get("table") or []
    if t and "factor" in t[0]:
        r = t[0]
        return [_f(f"'{r['factor']}' is the column most associated with the outcome ({r['measure']} {fmt_num(r['strength'])}).",
                   "Treat it as a hypothesis to test (A/B or pilot), not a proven lever. " + NON_CAUSAL, f"p = {fmt_num(r.get('p_value'))}, n = {r.get('n')}.", "medium")]
    return []


def _quality(result) -> List[Dict[str, Any]]:
    m = result.get("metrics") or {}
    if m.get("Quality score") is not None and m["Quality score"] < 85:
        return [_f(f"Data quality is {m['Quality score']:.0f}/100; decisions on this data carry extra risk.", "Apply the recommended cleaning and fix the high-severity issues at source.",
                   f"{m.get('High severity', 0)} high-severity issue(s).", "high")]
    return []


def business_insights(result: Dict[str, Any], plan: Dict[str, Any], data: Optional[pd.DataFrame], sem: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    op = result.get("operation")
    try:
        if op == "compare_groups":
            return _compare(result, plan, data, sem)
        if op == "analyze_trend":
            return _trend(result, plan, sem)
        if op in ("predict_class", "predict_value", "run_scenario"):
            return _model(result, plan)
        if op == "detect_anomalies":
            return _anomalies(result)
        if op == "forecast":
            return _forecast(result)
        if op == "explain_change":
            return _change(result)
        if op == "calculate_correlation":
            return _correlation(result)
        if op == "assess_data_quality":
            return _quality(result)
    except Exception:            # insights are an add-on: never fail the verified answer because of them
        return []
    return []
