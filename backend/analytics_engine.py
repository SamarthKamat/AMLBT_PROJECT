"""Deterministic pandas analytics: get_summary, get_current_state, compare_groups.

No machine learning is used here. All numbers come straight from the (filtered) data.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from context import ExecContext
from result_formatter import downsample, fmt_num, jsonsafe, make_result, records
from validator import ValidatedPlan

AGG_LABEL = {"mean": "Average", "sum": "Total", "median": "Median", "min": "Minimum", "max": "Maximum",
             "std": "Standard deviation", "count": "Count", "nunique": "Distinct count",
             "distribution": "Distribution", "percentage": "Percentage"}


def _agg_series(s: pd.Series, agg: str):
    if agg == "count":
        return int(s.notna().sum())
    if agg == "nunique":
        return int(s.nunique(dropna=True))
    return float(getattr(s.dropna(), agg)()) if s.notna().any() else None


def _describe_numeric(s: pd.Series) -> Dict[str, Any]:
    x = s.dropna().astype(float)
    if x.empty:
        return {}
    return {"count": int(x.size), "sum": float(x.sum()), "mean": float(x.mean()), "median": float(x.median()),
            "std": float(x.std()) if x.size > 1 else None, "min": float(x.min()), "q1": float(x.quantile(.25)),
            "q3": float(x.quantile(.75)), "max": float(x.max())}


def _histogram(s: pd.Series, bins: int = 20) -> List[Dict[str, Any]]:
    x = s.dropna().astype(float)
    if x.nunique() < 2:
        return []
    counts, edges = np.histogram(x, bins=min(bins, max(5, int(np.sqrt(len(x))))))
    return [{"range": f"{fmt_num(edges[i])}–{fmt_num(edges[i + 1])}", "records": int(counts[i])} for i in range(len(counts))]


# ----------------------------------------------------------------------------- get_summary
def get_summary(vp: ValidatedPlan, ctx: ExecContext) -> Dict[str, Any]:
    p, d = vp.plan, vp.data
    tgt, agg = p.target_column, p.aggregation
    role = ctx.roles.get(tgt, {}).get("role") if tgt else None

    if tgt is None:
        if agg == "count":
            return make_result("get_summary", f"The selection contains {len(d):,} records.",
                               title="Record count", metrics={"Records": len(d)})
        num_cols = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c]) and ctx.roles.get(c, {}).get("role") in ("numeric", "binary")]
        rows = []
        for c in num_cols[:30]:
            st = _describe_numeric(d[c])
            if st:
                rows.append({"column": c, "count": st["count"], "mean": st["mean"], "median": st["median"], "std": st["std"],
                             "min": st["min"], "max": st["max"], "missing": int(d[c].isna().sum())})
        cat_cols = [c for c in d.columns if ctx.roles.get(c, {}).get("role") in ("categorical",)]
        summary = (f"The selection has {len(d):,} records and {d.shape[1]} columns "
                   f"({len(num_cols)} numeric, {len(cat_cols)} categorical). ")
        if rows:
            summary += "Numeric columns are summarised in the table."
        return make_result("get_summary", summary, title="Dataset overview",
                           metrics={"Records": len(d), "Columns": d.shape[1], "Numeric columns": len(num_cols), "Missing cells": int(d.isna().sum().sum())},
                           table=rows, table_title="Numeric column statistics",
                           limitations=["No specific column was requested, so a whole-dataset overview is shown."])

    s = d[tgt]
    # ---- categorical / binary breakdowns
    if role in ("categorical", "binary") and not (agg in NUMERIC_AGGS_SET and pd.api.types.is_numeric_dtype(s)):
        if agg == "nunique":
            v = int(s.nunique(dropna=True))
            return make_result("get_summary", f"'{tgt}' has {v:,} distinct values across {int(s.notna().sum()):,} records.",
                               title=f"Distinct values of {tgt}", metrics={f"Distinct {tgt}": v, "Records": int(s.notna().sum())})
        vc = s.dropna().astype(str).value_counts()
        total = int(vc.sum())
        rows = [{tgt: k, "count": int(v), "percent": round(v / total * 100, 2)} for k, v in vc.head(50).items()]
        top = rows[0]
        summary = (f"{tgt} has {len(vc):,} distinct value(s) across {total:,} records; the most common is "
                   f"'{top[tgt]}' with {top['count']:,} records ({top['percent']:.1f}%).")
        if agg == "count":
            summary = f"There are {total:,} records with a value for '{tgt}'. " + summary
        return make_result("get_summary", summary, title=f"Breakdown of {tgt}",
                           metrics={"Records": total, "Distinct values": len(vc), "Most common": top[tgt], "Most common %": top["percent"]},
                           table=rows, table_title=f"{tgt} distribution",
                           charts=[{"type": "bar", "title": f"{tgt} distribution", "x": tgt, "y": "count", "data": rows[:15], "horizontal": len(rows) > 6}],
                           limitations=["Counts use exact text matching; spelling or casing variants are counted separately."] if len(vc) > 1 else [])

    if pd.api.types.is_datetime64_any_dtype(s):
        nn = s.dropna()
        v = {"min": nn.min(), "max": nn.max()}.get(agg or "", None)
        parts = f"'{tgt}' spans {nn.min().date()} to {nn.max().date()} across {len(nn):,} records."
        return make_result("get_summary", parts, title=f"Range of {tgt}", metrics={"Earliest": nn.min(), "Latest": nn.max(), "Records": len(nn)})

    st = _describe_numeric(s)
    if not st:
        return make_result("get_summary", f"'{tgt}' has no values in the selected rows.", title=f"Summary of {tgt}", metrics={"Records": 0})
    table = [{"statistic": k, "value": v} for k, v in
             [("count", st["count"]), ("sum", st["sum"]), ("mean", st["mean"]), ("median", st["median"]), ("std", st["std"]),
              ("min", st["min"]), ("25%", st["q1"]), ("75%", st["q3"]), ("max", st["max"])]]
    charts = []
    hist = _histogram(s)
    if hist:
        charts.append({"type": "bar", "title": f"Distribution of {tgt}", "x": "range", "y": "records", "data": hist})
    if agg and agg in NUMERIC_AGGS_SET | {"count", "nunique"}:
        val = _agg_series(s, agg)
        label = f"{AGG_LABEL[agg]} of {tgt}"
        summary = f"{label} = {fmt_num(val)} across {st['count']:,} records."
        metrics = {label: val, "Records": st["count"]}
        if agg != "median":
            metrics["Median"] = st["median"]
        if agg != "mean":
            metrics["Mean"] = st["mean"]
    else:
        summary = (f"{tgt}: mean {fmt_num(st['mean'])}, median {fmt_num(st['median'])}, range {fmt_num(st['min'])} to {fmt_num(st['max'])}, "
                   f"total {fmt_num(st['sum'])} across {st['count']:,} records.")
        metrics = {f"Average of {tgt}": st["mean"], "Median": st["median"], "Minimum": st["min"], "Maximum": st["max"]}
    missing = int(s.isna().sum())
    lim = [f"{missing:,} record(s) have no value for '{tgt}' and were excluded."] if missing else []
    return make_result("get_summary", summary, title=f"Summary of {tgt}", metrics=metrics, table=table,
                       table_title=f"Descriptive statistics for {tgt}", charts=charts, limitations=lim)


NUMERIC_AGGS_SET = {"mean", "sum", "median", "min", "max", "std"}


# ----------------------------------------------------------------------------- get_current_state
def get_current_state(vp: ValidatedPlan, ctx: ExecContext) -> Dict[str, Any]:
    p, d, tc = vp.plan, vp.data, vp.time_column
    tgt = p.target_column
    work = d.dropna(subset=[tc] + ([tgt] if tgt else [])).copy()
    if work.empty:
        return make_result("get_current_state", "No observation with a valid date" + (f" and a value for '{tgt}'" if tgt else "") + " was found.",
                           title="Latest state", metrics={"Records": 0})
    work = work.sort_values(tc)
    agg = p.aggregation if (p.aggregation in NUMERIC_AGGS_SET and tgt and pd.api.types.is_numeric_dtype(work[tgt])) else None
    assumptions = list(vp.assumptions)
    groups = p.group_by or []

    if groups:
        g = groups[0]
        rows = []
        for key, sub in work.groupby(g, dropna=True):
            latest = sub[tc].max()
            at = sub[sub[tc] == latest]
            row = {g: key, "as_of": latest, "records_at_latest": len(at)}
            if tgt:
                row[tgt] = (_agg_series(at[tgt], agg or "mean") if pd.api.types.is_numeric_dtype(at[tgt]) else str(at[tgt].mode().iloc[0]))
            rows.append(row)
        tab = pd.DataFrame(rows)
        if tgt and pd.api.types.is_numeric_dtype(tab[tgt]):
            tab = tab.sort_values(tgt, ascending=(p.sort_order == "asc"))
        charts = []
        if tgt and pd.api.types.is_numeric_dtype(tab[tgt]):
            charts.append({"type": "bar", "title": f"Latest {tgt} by {g}", "x": g, "y": tgt, "data": records(tab.head(20)), "horizontal": len(tab) > 8})
        top = tab.iloc[0]
        summary = (f"Latest observation per {g}: " + (f"{tgt} is highest for '{top[g]}' at {fmt_num(top[tgt])}. " if tgt and pd.api.types.is_numeric_dtype(tab[tgt]) else "") +
                   f"Each group's most recent date is shown (overall latest: {work[tc].max().date()}).")
        return make_result("get_current_state", summary, title=f"Latest state by {g}", table=records(tab, 50), table_title=f"Most recent record per {g}",
                           metrics={"Overall latest date": work[tc].max(), "Groups": len(tab)}, charts=charts, assumptions=assumptions)

    latest = work[tc].max()
    at = work[work[tc] == latest]
    metrics: Dict[str, Any] = {"As of": latest, "Records at that time": len(at)}
    lim: List[str] = []
    if tgt:
        if pd.api.types.is_numeric_dtype(at[tgt]):
            use = agg or ("mean" if len(at) > 1 else "mean")
            val = _agg_series(at[tgt], use)
            if len(at) > 1:
                assumptions.append(f"{len(at)} rows share the latest timestamp, so the {use} of '{tgt}' across them is reported.")
            metrics[f"{'Latest' if len(at) == 1 else AGG_LABEL[use]} {tgt}"] = val
            summary = f"As of {latest.date() if hasattr(latest, 'date') else latest}, {tgt} is {fmt_num(val)}" + (f" ({use} of {len(at)} records)." if len(at) > 1 else ".")
            prev = work[work[tc] < latest]
            if not prev.empty:
                pl = prev[tc].max()
                pv = _agg_series(prev[prev[tc] == pl][tgt], use if len(at) > 1 else "mean")
                if pv not in (None, 0) and val is not None:
                    metrics["Previous value"] = pv
                    metrics["Change vs previous"] = (val - pv) / abs(pv) * 100
                    summary += f" That is {(val - pv) / abs(pv) * 100:+.1f}% versus the previous observation on {pl.date()}."
        else:
            val = ", ".join(map(str, at[tgt].astype(str).unique()[:5]))
            metrics[f"Latest {tgt}"] = val
            summary = f"As of {latest.date()}, {tgt} is {val}."
    else:
        summary = f"The most recent record is dated {latest.date() if hasattr(latest, 'date') else latest} ({len(at)} record(s) on that date)."
    # the timeline immediately preceding, for context
    context_rows = work.tail(10)
    cols = [tc] + ([tgt] if tgt else []) + [c for c in work.columns if c not in (tc, tgt)][:6]
    if len(at) > 1 and not tgt:
        lim.append("Several rows share the latest timestamp; all are shown.")
    return make_result("get_current_state", summary, title="Latest state", metrics=metrics,
                       table=records(at[[c for c in cols if c in at.columns]], 20), table_title="Records at the latest timestamp",
                       charts=[], limitations=lim, assumptions=assumptions)


# ----------------------------------------------------------------------------- compare_groups
def compare_groups(vp: ValidatedPlan, ctx: ExecContext) -> Dict[str, Any]:
    p, d = vp.plan, vp.data
    groups, tgt, agg = p.group_by, p.target_column, p.aggregation
    desc = p.sort_order != "asc"
    top_n = p.top_n or 15
    work = d.dropna(subset=groups).copy()
    if work.empty:
        return make_result("compare_groups", f"No rows have a value for {', '.join(groups)}.", title="Group comparison", metrics={"Groups": 0})
    label_col = " · ".join(groups)
    if len(groups) == 2:
        work[label_col] = work[groups[0]].astype(str) + " · " + work[groups[1]].astype(str)
        keys = [label_col]
    else:
        keys = groups

    # ---------------- categorical measure: composition per group
    if tgt and not pd.api.types.is_numeric_dtype(work[tgt]):
        sub = work.dropna(subset=[tgt])
        ct = pd.crosstab(sub[keys[0]], sub[tgt].astype(str))
        share = ct.div(ct.sum(axis=1), axis=0) * 100
        main = sub[tgt].astype(str).value_counts().index[0]
        tab = share.copy()
        tab.insert(0, "records", ct.sum(axis=1))
        tab = tab.sort_values(main, ascending=not desc)
        rows = [{label_col: k, **{c: float(v) if c != "records" else int(v) for c, v in r.items()}} for k, r in tab.head(50).iterrows()]
        p_val = None
        try:
            if ct.shape[0] > 1 and ct.shape[1] > 1:
                p_val = float(stats.chi2_contingency(ct.values)[1])
        except Exception:
            pass
        best, worst = tab.index[0], tab.index[-1]
        summary = (f"'{main}' (the most common {tgt}) makes up {tab.loc[best, main]:.1f}% of '{best}' versus {tab.loc[worst, main]:.1f}% of '{worst}'.")
        metrics = {"Groups": int(len(tab)), f"Highest share of '{main}'": str(best), "Highest share %": float(tab.loc[best, main])}
        lim = ["Percentages are shares of each group's records; they describe association, not cause."]
        if p_val is not None:
            metrics["Chi-square p-value"] = p_val
            summary += f" Differences across groups are {'statistically significant' if p_val < 0.05 else 'not statistically significant'} (chi-square p = {p_val:.3g})."
        return make_result("compare_groups", summary, title=f"{tgt} mix by {label_col}", metrics=metrics, table=rows, table_title=f"Share of each {tgt} class by {label_col} (%)",
                           charts=[{"type": "bar", "title": f"Share of '{main}' by {label_col}", "x": label_col, "y": main, "data": rows[:top_n], "horizontal": len(rows) > 8}],
                           limitations=lim, assumptions=vp.assumptions)

    # ---------------- numeric measure or record count
    assumptions = list(vp.assumptions)
    if not tgt:
        agg_eff = "count"
        assumptions.append("No measure column was specified, so groups are compared by number of records.")
    else:
        agg_eff = agg if agg in NUMERIC_AGGS_SET | {"count", "nunique"} else "mean"
    gb = work.groupby(keys[0], dropna=True)
    if not tgt:
        res = gb.size().rename("value").to_frame()
        res["records"] = res["value"]
        measure_label = "Records"
    else:
        s_by = gb[tgt]
        if agg_eff == "count":
            val = s_by.count()
        elif agg_eff == "nunique":
            val = s_by.nunique()
        else:
            val = getattr(s_by, agg_eff)()
        res = val.rename("value").to_frame()
        res["records"] = gb.size()
        measure_label = f"{AGG_LABEL[agg_eff]} of {tgt}"
    res = res.dropna(subset=["value"])
    res = res.sort_values("value", ascending=not desc)
    overall = (len(work) if not tgt else _agg_series(work[tgt], agg_eff))
    if overall not in (None, 0):
        res["vs_overall_%"] = (res["value"] - overall) / abs(overall) * 100
    if agg_eff in ("sum", "count") or not tgt:
        tot = res["value"].sum()
        if tot:
            res["share_%"] = res["value"] / tot * 100
    out = res.reset_index().rename(columns={keys[0]: label_col, "value": measure_label})
    rows = records(out.head(50))
    best, worst = out.iloc[0], out.iloc[-1]
    word_hi, word_lo = ("highest", "lowest") if desc else ("lowest", "highest")
    summary = (f"{measure_label} is {word_hi} for '{best[label_col]}' at {fmt_num(best[measure_label])}"
               + (f" and {word_lo} for '{worst[label_col]}' at {fmt_num(worst[measure_label])}" if len(out) > 1 else "")
               + f", across {len(out):,} group(s).")
    metrics: Dict[str, Any] = {"Groups": int(len(out)), f"Top group": str(best[label_col]), "Top value": float(best[measure_label]),
                               "Bottom group": str(worst[label_col]), "Bottom value": float(worst[measure_label])}
    limitations = ["Comparisons are descriptive; differences between groups do not by themselves explain why they differ."]
    # significance of the differences between groups (numeric measure only)
    if tgt and agg_eff not in ("count", "nunique") and len(out) >= 2:
        samples = [g[tgt].dropna().to_numpy(dtype=float) for _, g in work.groupby(keys[0]) if g[tgt].notna().sum() >= 3]
        if len(samples) >= 2:
            try:
                kw = stats.kruskal(*samples)
                metrics["Kruskal–Wallis p-value"] = float(kw.pvalue)
                sig = kw.pvalue < 0.05
                summary += f" The differences between groups are {'statistically distinguishable' if sig else 'not statistically significant'} (Kruskal–Wallis p = {kw.pvalue:.3g})."
                if not sig:
                    limitations.append("The ranking may reflect random variation: group differences are not statistically significant at the 5% level.")
            except Exception:
                pass
    small = int((out["records"] < 5).sum())
    if small:
        limitations.append(f"{small} group(s) have fewer than 5 records, so their values are unstable.")
    chart_df = out.head(top_n)
    horizontal = len(chart_df) > 8 or chart_df[label_col].astype(str).str.len().max() > 12
    charts = [{"type": "bar", "title": f"{measure_label} by {label_col}", "x": label_col, "y": measure_label,
               "data": records(chart_df), "horizontal": bool(horizontal)}]
    return make_result("compare_groups", summary, title=f"{measure_label} by {label_col}", metrics=metrics, table=rows,
                       table_title=f"{measure_label} by {label_col}", charts=charts, limitations=limitations, assumptions=assumptions)
