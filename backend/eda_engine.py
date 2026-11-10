"""Exploratory data analysis, data-quality and relationship operations (Phase 2; deterministic, no ML).

run_eda                 whole-dataset EDA (or a focused EDA of one column): distributions, skewness, outliers,
                        category mixes, missingness, association heatmap, time coverage and auto-written insights
assess_data_quality     the 0-100 quality score with issues and recommended treatments (data_quality.py)
describe_relationships  the column relationship map and data dictionary (relationships.py)
"""
from __future__ import annotations
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from artifacts import get_quality, get_relationships, get_semantics
from context import ExecContext
from data_profiler import profile_dataframe
from data_quality import assess_quality, public
from relationships import analyse_relationships, association_matrix, correlation_ratio, cramers_v
from result_formatter import fmt_num, make_result, records
from validator import ValidatedPlan


def _hist(x: pd.Series, bins: int = 20) -> List[Dict[str, Any]]:
    x = x.dropna().astype(float)
    if x.nunique() < 2:
        return []
    counts, edges = np.histogram(x, bins=min(bins, max(5, int(np.sqrt(len(x))))))
    return [{"range": f"{fmt_num(edges[i])}–{fmt_num(edges[i + 1])}", "records": int(counts[i])} for i in range(len(counts))]


def box_stats(x: pd.Series, name: str) -> Dict[str, Any]:
    x = x.dropna().astype(float)
    q1, med, q3 = x.quantile([.25, .5, .75])
    iqr = q3 - q1
    lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = x[(x >= lo_f) & (x <= hi_f)]
    return {"name": name, "min": float(inside.min() if len(inside) else x.min()), "q1": float(q1), "median": float(med), "q3": float(q3),
            "max": float(inside.max() if len(inside) else x.max()), "outliers": int(((x < lo_f) | (x > hi_f)).sum()), "mean": float(x.mean())}


def heatmap_chart(assoc: Dict[str, Any], title: str, limit: int = 14) -> Dict[str, Any]:
    cols = assoc["columns"][:limit]
    idx = [assoc["columns"].index(c) for c in cols]
    M = [[assoc["signed"][i][j] if assoc["signed"][i][j] is not None else assoc["matrix"][i][j] for j in idx] for i in idx]
    return {"type": "heatmap", "title": title, "columns": cols, "matrix": M,
            "note": "Numeric pairs: signed Pearson/Spearman correlation. Pairs with a category: strength 0–1 (η or Cramér's V)."}


def _num_cols(d: pd.DataFrame, ctx: ExecContext) -> List[str]:
    return [c for c in d.columns if ctx.roles.get(c, {}).get("role") == "numeric" and d[c].nunique(dropna=True) > 1]


def _cat_cols(d: pd.DataFrame, ctx: ExecContext) -> List[str]:
    return [c for c in d.columns if ctx.roles.get(c, {}).get("role") in ("categorical", "binary") and 2 <= d[c].nunique(dropna=True) <= 60]


def run_eda(vp: ValidatedPlan, ctx: ExecContext) -> Dict[str, Any]:
    p, d = vp.plan, vp.data
    sem = get_semantics(ctx, d)
    if p.target_column:
        return _focused_eda(vp, ctx, sem)
    prof = profile_dataframe(d, ctx.profile.get("dataset_name", "dataset")) if len(d) != vp.total_rows else ctx.profile
    nums, cats = _num_cols(d, ctx), _cat_cols(d, ctx)
    kpis = [c for c in nums if sem.get(c, {}).get("semantic") in ("money", "quantity")] or nums
    ordered_nums = list(dict.fromkeys(kpis + nums))
    num_rows, insights, charts = [], [], []
    for c in ordered_nums[:25]:
        x = d[c].dropna().astype(float)
        if len(x) < 3:
            continue
        b = box_stats(x, c)
        sk = float(x.skew())
        num_rows.append({"column": c, "semantic": sem.get(c, {}).get("semantic"), "mean": float(x.mean()), "median": b["median"], "std": float(x.std()),
                         "min": float(x.min()), "max": float(x.max()), "skewness": sk, "outliers_1.5IQR": b["outliers"], "missing_%": float(d[c].isna().mean() * 100)})
        if abs(sk) >= 1:
            insights.append(f"'{c}' is {'right' if sk > 0 else 'left'}-skewed (skewness {sk:.2f}); its median ({fmt_num(b['median'])}) is a more typical value than its mean ({fmt_num(float(x.mean()))}).")
        if b["outliers"] / len(x) > 0.05:
            insights.append(f"{b['outliers']:,} values of '{c}' ({b['outliers'] / len(x):.1%}) lie outside the 1.5×IQR fences.")
    cat_rows = []
    for c in cats[:15]:
        vc = d[c].astype(str).value_counts()
        top_share = vc.iloc[0] / vc.sum()
        cat_rows.append({"column": c, "distinct": int(len(vc)), "most_common": vc.index[0], "most_common_%": float(top_share * 100), "missing_%": float(d[c].isna().mean() * 100)})
        if top_share >= 0.7 and len(vc) > 1:
            insights.append(f"'{c}' is dominated by '{vc.index[0]}' ({top_share:.0%} of records): comparisons across its other values rest on few rows.")
    for c in kpis[:4]:
        h = _hist(d[c])
        if h:
            charts.append({"type": "bar", "title": f"Distribution of {c}", "x": "range", "y": "records", "data": h})
    if ordered_nums:
        boxes = [box_stats(d[c], c) for c in ordered_nums[:8] if d[c].notna().sum() > 3]
        charts.append({"type": "box", "title": "Spread of numeric columns (box plots, 1.5×IQR whiskers)", "data": boxes, "independent": True})
    for c in cats[:3]:
        vc = d[c].astype(str).value_counts().head(8)
        charts.append({"type": "pie" if len(vc) <= 6 else "bar", "title": f"Mix of {c}", "x": c, "y": "records",
                       "data": [{c: k, "records": int(v)} for k, v in vc.items()], "horizontal": len(vc) > 6})
    miss = [{"column": c, "missing_%": float(d[c].isna().mean() * 100)} for c in d.columns if d[c].isna().any()]
    if miss:
        charts.append({"type": "bar", "title": "Missing values by column (%)", "x": "column", "y": "missing_%", "data": sorted(miss, key=lambda r: -r["missing_%"])[:15], "horizontal": True})
        worst = max(miss, key=lambda r: r["missing_%"])
        insights.append(f"{len(miss)} column(s) have missing values; the worst is '{worst['column']}' ({worst['missing_%']:.1f}%).")
    assoc = association_matrix(d, prof)
    if len(assoc["columns"]) >= 2:
        charts.append(heatmap_chart(assoc, "Association heatmap (all usable columns)"))
        for pr in [x for x in assoc["pairs"] if x["strength"] >= 0.5][:3]:
            sign = "" if pr["signed"] is None else (" positively" if pr["signed"] > 0 else " negatively")
            insights.append(f"'{pr['column_1']}' and '{pr['column_2']}' are{sign} associated ({pr['measure'].replace('_', ' ')} {pr['strength']:.2f}).")
    tc = ctx.profile.get("primary_time_column")
    time_info = None
    if tc and tc in d.columns and kpis:
        k = kpis[0]
        w = d.dropna(subset=[tc]).copy()
        span = (w[tc].max() - w[tc].min()).days
        freq = "M" if span > 180 else "W" if span > 30 else "D"
        agg = "sum" if sem.get(k, {}).get("additive") else "mean"
        ser = getattr(w.groupby(w[tc].dt.to_period(freq))[k], agg)()
        if len(ser) >= 3:
            charts.append({"type": "line", "title": f"{'Total' if agg == 'sum' else 'Average'} {k} per {'month' if freq == 'M' else 'week' if freq == 'W' else 'day'}", "x": "period", "y": [k],
                           "data": [{"period": str(i), k: float(v)} for i, v in ser.items()]})
            time_info = {"column": tc, "start": str(w[tc].min())[:10], "end": str(w[tc].max())[:10], "periods": int(len(ser))}
            first, last = float(ser.iloc[0]), float(ser.iloc[-1])
            insights.append(f"Data covers {time_info['start']} to {time_info['end']}; {agg} of '{k}' went from {fmt_num(first)} in the first period to {fmt_num(last)} in the latest.")
    dup = int(d.duplicated().sum())
    if dup:
        insights.append(f"{dup:,} exact duplicate rows are present.")
    summary = (f"EDA of {len(d):,} records and {d.shape[1]} columns ({len(nums)} numeric, {len(cats)} categorical"
               f"{', 1 date column' if tc else ''}). " + (insights[0] if insights else "No notable skew, dominance or missingness was found."))
    metrics = {"Records": len(d), "Columns": d.shape[1], "Numeric": len(nums), "Categorical": len(cats),
               "Missing cells": int(d.isna().sum().sum()), "Duplicate rows": dup}
    return make_result("run_eda", summary, title="Exploratory data analysis", metrics=metrics, table=num_rows, table_title="Numeric columns",
                       tables={"categorical_columns": cat_rows}, charts=charts, insights=insights[:12],
                       eda={"numeric": num_rows, "categorical": cat_rows, "time": time_info, "association": assoc},
                       limitations=["EDA describes the data as recorded; associations are not causal."], assumptions=list(vp.assumptions))


def _focused_eda(vp: ValidatedPlan, ctx: ExecContext, sem) -> Dict[str, Any]:
    p, d = vp.plan, vp.data
    tgt = p.target_column
    s = d[tgt]
    insights, charts, rel_rows = [], [], []
    numeric = ctx.roles.get(tgt, {}).get("role") == "numeric"
    metrics: Dict[str, Any] = {"Records": int(s.notna().sum()), "Missing": int(s.isna().sum()), "Distinct": int(s.nunique())}
    if numeric:
        x = s.dropna().astype(float)
        b = box_stats(x, tgt)
        metrics.update({"Mean": float(x.mean()), "Median": b["median"], "Std": float(x.std()), "Min": float(x.min()), "Max": float(x.max()), "Skewness": float(x.skew())})
        charts.append({"type": "bar", "title": f"Distribution of {tgt}", "x": "range", "y": "records", "data": _hist(x)})
        charts.append({"type": "box", "title": f"Box plot of {tgt}", "data": [b]})
        if abs(x.skew()) >= 1:
            insights.append(f"'{tgt}' is {'right' if x.skew() > 0 else 'left'}-skewed (skewness {x.skew():.2f}).")
        if b["outliers"]:
            insights.append(f"{b['outliers']:,} value(s) of '{tgt}' lie outside the 1.5×IQR fences.")
    else:
        vc = s.astype(str).value_counts()
        metrics.update({"Most common": vc.index[0], "Most common %": float(vc.iloc[0] / vc.sum() * 100)})
        charts.append({"type": "pie" if len(vc) <= 6 else "bar", "title": f"Mix of {tgt}", "x": tgt, "y": "records", "data": [{tgt: k, "records": int(v)} for k, v in vc.head(12).items()]})
    for c in d.columns:
        if c == tgt or ctx.roles.get(c, {}).get("role") in ("identifier", "text", "constant", "empty", "datetime"):
            continue
        pair = d[[c, tgt]].dropna()
        if len(pair) < 20:
            continue
        cnum = ctx.roles.get(c, {}).get("role") == "numeric"
        if numeric and cnum:
            v = float(pair[c].corr(pair[tgt]))
            rel_rows.append({"column": c, "measure": "Pearson r", "strength": abs(v), "signed": v})
        elif numeric and not cnum and pair[c].nunique() <= 60:
            v = correlation_ratio(pair[c], pair[tgt])
            if v is not None:
                rel_rows.append({"column": c, "measure": "η", "strength": v, "signed": None})
        elif not numeric and cnum:
            v = correlation_ratio(pair[tgt], pair[c])
            if v is not None:
                rel_rows.append({"column": c, "measure": "η", "strength": v, "signed": None})
        elif pair[c].nunique() <= 60:
            v = cramers_v(pair[c], pair[tgt])
            if v is not None:
                rel_rows.append({"column": c, "measure": "Cramér's V", "strength": v, "signed": None})
    rel_rows.sort(key=lambda r: -r["strength"])
    if rel_rows:
        charts.append({"type": "bar", "title": f"Columns most related to {tgt}", "x": "column", "y": "strength", "data": rel_rows[:12], "horizontal": True})
        top = rel_rows[0]
        insights.append(f"The column most related to '{tgt}' is '{top['column']}' ({top['measure']} {top['strength']:.2f}).")
        best_cat = next((r for r in rel_rows if ctx.roles.get(r["column"], {}).get("role") in ("categorical", "binary")), None)
        if best_cat and numeric:
            g = d.groupby(best_cat["column"])[tgt].mean().sort_values(ascending=False)
            charts.append({"type": "bar", "title": f"Average {tgt} by {best_cat['column']}", "x": best_cat["column"], "y": tgt,
                           "data": [{best_cat["column"]: str(k), tgt: float(v)} for k, v in g.head(12).items()]})
            insights.append(f"Average '{tgt}' is highest for {best_cat['column']} = '{g.index[0]}' ({fmt_num(g.iloc[0])}) and lowest for '{g.index[-1]}' ({fmt_num(g.iloc[-1])}).")
    tc = vp.time_column
    if numeric and tc and tc in d.columns:
        w = d.dropna(subset=[tc, tgt])
        span = (w[tc].max() - w[tc].min()).days if len(w) else 0
        freq = "M" if span > 180 else "W" if span > 30 else "D"
        agg = "sum" if sem.get(tgt, {}).get("additive") else "mean"
        ser = getattr(w.groupby(w[tc].dt.to_period(freq))[tgt], agg)()
        if len(ser) >= 3:
            charts.append({"type": "line", "title": f"{tgt} over time ({agg})", "x": "period", "y": [tgt], "data": [{"period": str(i), tgt: float(v)} for i, v in ser.items()]})
    summary = f"Focused EDA of '{tgt}' on {int(s.notna().sum()):,} records. " + (" ".join(insights[:2]) if insights else "")
    return make_result("run_eda", summary.strip(), title=f"Exploring {tgt}", metrics=metrics, table=rel_rows[:20], table_title=f"Relationship of other columns with {tgt}",
                       charts=charts, insights=insights, limitations=["Associations describe the recorded data and are not causal."], assumptions=list(vp.assumptions))


def assess_data_quality(vp: ValidatedPlan, ctx: ExecContext) -> Dict[str, Any]:
    d = vp.data
    sem = get_semantics(ctx, d)
    q = get_quality(ctx, d) if len(d) == vp.total_rows else assess_quality(d, profile_dataframe(d), sem)
    pq = public(q)
    issues = pq["issues"]
    charts = [{"type": "bar", "title": "Quality score by dimension (0–100)", "x": "dimension", "y": "score",
               "data": [{"dimension": k, "score": v} for k, v in pq["dimensions"].items()]}]
    by_dim: Dict[str, int] = {}
    for i in issues:
        by_dim[i["dimension"]] = by_dim.get(i["dimension"], 0) + 1
    if issues:
        charts.append({"type": "pie", "title": "Issues by dimension", "x": "dimension", "y": "issues", "data": [{"dimension": k, "issues": v} for k, v in by_dim.items()]})
    table = [{"#": i["id"], "severity": i["severity"], "dimension": i["dimension"], "issue": i["issue"], "recommendation": i["recommendation"]} for i in issues[:40]]
    metrics = {"Quality score": pq["score"], "Grade": pq["grade"], "Issues": len(issues), "High severity": sum(1 for i in issues if i["severity"] == "high"),
               "Missing cells": pq["missing_cells"], "Duplicate rows": pq["duplicate_rows"]}
    insights = [i["recommendation"] for i in issues if i["severity"] in ("high", "medium")][:6]
    return make_result("assess_data_quality", pq["summary"], title="Data quality assessment", metrics=metrics, table=table, table_title="Issues and recommended treatments",
                       tables={"column_quality": pq["columns"]}, charts=charts, insights=insights, quality=pq,
                       limitations=["The score uses fixed, documented weights and sensitivities (see methodology); it summarises measured defect rates, not fitness for every purpose.",
                                    "Use the Data Quality tab to apply the recommended cleaning and compare before/after."], assumptions=list(vp.assumptions))


def describe_relationships(vp: ValidatedPlan, ctx: ExecContext) -> Dict[str, Any]:
    d = vp.data
    sem = get_semantics(ctx, d)
    rel = get_relationships(ctx, d) if len(d) == vp.total_rows else analyse_relationships(d, profile_dataframe(d), sem)
    tgt = vp.plan.target_column
    pairs = rel["association"]["pairs"]
    if tgt:
        pairs = [p for p in pairs if tgt in (p["column_1"], p["column_2"])]
    table = [{"column_1": p["column_1"], "column_2": p["column_2"], "measure": p["measure"], "strength": p["strength"], "direction": None if p["signed"] is None else ("positive" if p["signed"] > 0 else "negative"),
              "label": p["label"]} for p in pairs[:40]]
    charts = [heatmap_chart(rel["association"], "Column relationship heatmap")] if len(rel["association"]["columns"]) >= 2 else []
    if pairs:
        charts.append({"type": "bar", "title": "Strongest column relationships" + (f" with {tgt}" if tgt else ""), "x": "pair", "y": "strength",
                       "data": [{"pair": f"{p['column_1']} × {p['column_2']}", "strength": p["strength"]} for p in pairs[:12]], "horizontal": True})
    findings = rel["findings"] if not tgt else ([f for f in rel["findings"] if tgt in f] or [f"No structural rule involves '{tgt}'."])
    if tgt and pairs:
        findings = [f"'{tgt}' is most related to '{pairs[0]['column_2'] if pairs[0]['column_1'] == tgt else pairs[0]['column_1']}' ({pairs[0]['measure'].replace('_', ' ')} {pairs[0]['strength']:.2f})."] + findings
    summary = f"Relationship map of {len(rel['association']['columns'])} columns: " + (findings[0] if findings else "no notable relationships.")
    metrics = {"Columns mapped": len(rel["association"]["columns"]), "Strong pairs (≥0.5)": sum(1 for p in rel["association"]["pairs"] if p["strength"] >= 0.5),
               "Hierarchies": len(rel["hierarchies"]), "Derived columns": len(rel["derived"]), "Candidate keys": len(rel["candidate_keys"])}
    return make_result("describe_relationships", summary, title="How the columns relate", metrics=metrics, table=table, table_title="Column pairs ranked by association strength",
                       tables={"data_dictionary": rel["dictionary"], "hierarchies": [{k: v for k, v in h.items()} for h in rel["hierarchies"]],
                               "derived_columns": rel["derived"]}, charts=charts, insights=findings[:12], relationships={k: v for k, v in rel.items() if k != "dictionary"} | {"dictionary": rel["dictionary"]},
                       limitations=["Association strength (|r|, η, Cramér's V) measures how predictable one column is from another; it is not causation.",
                                    "Hierarchies and formulas are detected from the data; confirm them against business definitions."], assumptions=list(vp.assumptions))
