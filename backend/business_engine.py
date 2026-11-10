"""Business Decision Intelligence operations (Phase 2).

query_records    list the records that match conditions, sorted by a measure ("top 10 orders by profit in India")
pivot_table      a measure by two dimensions ("total sales by region and category") with a heatmap
explain_change   root-cause decomposition of a KPI change between two periods
segment_records  K-means segmentation with silhouette-selected k and readable segment profiles

Root-cause decomposition (explain_change). For an additive KPI (a sum) the change splits exactly over the
groups g of a dimension:   Delta = sum_g (M_g,B - M_g,A).
For an average KPI the change splits into a MIX effect (group shares moved) and a RATE effect (group averages moved):
    Delta mean = sum_g (w_g,B - w_g,A) * m_g,A   +   sum_g w_g,B * (m_g,B - m_g,A)
Each dimension is scored by how concentrated its contributions are; the most concentrated dimension "explains"
the change best. All numbers are computed from the data; the decomposition is descriptive, not causal.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from artifacts import get_semantics
from context import ExecContext
from diagnostics_engine import FREQ_NAME, PERIOD_ALIAS, native_frequency, period_label, to_period
from errors import UDIError
from result_formatter import fmt_num, make_result, records
from semantics import default_aggregation, dimension_columns
from validator import ValidatedPlan

AGG_WORD = {"sum": "Total", "mean": "Average", "median": "Median", "count": "Count", "min": "Minimum", "max": "Maximum", "nunique": "Distinct"}


# ----------------------------------------------------------------------------- query_records
def query_records(vp: ValidatedPlan, ctx: ExecContext) -> Dict[str, Any]:
    p, d = vp.plan, vp.data
    tgt = p.target_column
    n = p.top_n or 20
    work = d.copy()
    if tgt:
        work = work.dropna(subset=[tgt]).sort_values(tgt, ascending=(p.sort_order == "asc"))
    elif vp.time_column and vp.time_column in work.columns:
        work = work.sort_values(vp.time_column, ascending=(p.sort_order == "asc"))
    show = [c for c in (p.feature_columns or []) if c in work.columns]
    cols = list(dict.fromkeys(([tgt] if tgt else []) + show + [c for c in work.columns if c not in show and c != tgt]))[:12]
    out = work.head(n)[cols]
    out.insert(0, "row", out.index)
    filt = "; ".join(vp.filters_applied) if vp.filters_applied else "no filter"
    summary = f"{len(work):,} record(s) match ({filt})."
    if tgt and len(out):
        word = "lowest" if p.sort_order == "asc" else "highest"
        summary += f" Showing the {min(n, len(work))} with the {word} '{tgt}': the first has {fmt_num(out.iloc[0][tgt])}."
    metrics = {"Matching records": int(len(work)), "Shown": int(len(out))}
    if tgt and len(work) and pd.api.types.is_numeric_dtype(work[tgt]):
        metrics.update({f"Total {tgt} (matching)": float(work[tgt].sum()), f"Average {tgt} (matching)": float(work[tgt].mean())})
    charts = []
    if tgt and len(out) and pd.api.types.is_numeric_dtype(out[tgt]):
        lab = next((c for c in cols if c != tgt and ctx.roles.get(c, {}).get("role") in ("identifier", "categorical", "text")), None)
        charts.append({"type": "bar", "title": f"{tgt} of the listed records", "x": "label", "y": tgt,
                       "data": [{"label": f"{r[lab]}" if lab else f"row {r['row']}", tgt: float(r[tgt])} for _, r in out.head(15).iterrows()], "horizontal": True})
    return make_result("query_records", summary, title="Matching records", metrics=metrics, table=records(out), table_title=f"Records ({filt})",
                       charts=charts, assumptions=list(vp.assumptions))


# ----------------------------------------------------------------------------- pivot_table
def pivot_table(vp: ValidatedPlan, ctx: ExecContext) -> Dict[str, Any]:
    p, d = vp.plan, vp.data
    rows_dim = p.group_by[0]
    col_dim = p.pivot_column or (p.group_by[1] if len(p.group_by) > 1 else None)
    if not col_dim:
        raise UDIError("INVALID_PLAN", "A pivot table needs two dimensions (rows and columns).")
    tgt = p.target_column
    sem = get_semantics(ctx, d)
    agg = p.aggregation or (default_aggregation(sem, tgt) if tgt else "count")
    if tgt is None:
        agg = "count"
    work = d.dropna(subset=[rows_dim, col_dim])
    top_r = work[rows_dim].astype(str).value_counts().head(20).index
    top_c = work[col_dim].astype(str).value_counts().head(12).index
    work = work[work[rows_dim].astype(str).isin(top_r) & work[col_dim].astype(str).isin(top_c)]
    rate_label = None
    if tgt is not None and not pd.api.types.is_numeric_dtype(work[tgt]):
        vals = sorted(work[tgt].dropna().astype(str).unique())
        rate_label = next((v for v in vals if v.strip().lower() in ("yes", "1", "true", "churned", "late", "left", "default", "y")), vals[-1])
        work = work.assign(_rate=(work[tgt].astype(str) == rate_label) * 100.0)
        agg = "mean"
    if tgt is None:
        pt = pd.crosstab(work[rows_dim].astype(str), work[col_dim].astype(str))
    elif rate_label is not None:
        pt = work.pivot_table(index=work[rows_dim].astype(str), columns=work[col_dim].astype(str), values="_rate", aggfunc="mean")
    else:
        pt = work.pivot_table(index=work[rows_dim].astype(str), columns=work[col_dim].astype(str), values=tgt, aggfunc=agg)
    pt = pt.loc[pt.sum(axis=1).sort_values(ascending=False).index]
    measure = (f"% {tgt} = '{rate_label}'" if rate_label is not None else f"{AGG_WORD.get(agg, agg)} of {tgt}") if tgt else "Record count"
    cell = pt.stack()
    best = cell.idxmax()
    worst = cell.idxmin()
    summary = (f"{measure} by {rows_dim} × {col_dim}: the highest cell is {rows_dim}='{best[0]}', {col_dim}='{best[1]}' ({fmt_num(cell.max())}); "
               f"the lowest is '{worst[0]}' × '{worst[1]}' ({fmt_num(cell.min())}).")
    table = [{rows_dim: idx, **{c: (None if pd.isna(v) else float(v)) for c, v in row.items()}} for idx, row in pt.iterrows()]
    if agg in ("sum", "count"):
        for r, idx in zip(table, pt.index):
            r["row_total"] = float(pt.loc[idx].sum())
    heat = {"type": "heatmap", "title": f"{measure} by {rows_dim} and {col_dim}", "columns": list(pt.columns), "rows": list(pt.index),
            "matrix": [[None if pd.isna(v) else float(v) for v in row] for row in pt.to_numpy()], "absolute": True}
    stacked = {"type": "bar", "title": f"{measure} by {rows_dim}, split by {col_dim}", "x": rows_dim, "y": list(pt.columns), "stacked": agg in ("sum", "count"),
               "data": [{rows_dim: idx, **{c: (None if pd.isna(v) else float(v)) for c, v in row.items()}} for idx, row in pt.head(12).iterrows()]}
    lim = []
    if len(top_r) == 20 or len(top_c) == 12:
        lim.append("Only the most frequent 20 row values and 12 column values are shown.")
    return make_result("pivot_table", summary, title=f"{measure}: {rows_dim} × {col_dim}", metrics={"Rows": len(pt), "Columns": len(pt.columns), "Highest cell": float(cell.max()), "Lowest cell": float(cell.min())},
                       table=table, table_title=f"{measure} by {rows_dim} (rows) and {col_dim} (columns)", charts=[heat, stacked], limitations=lim, assumptions=list(vp.assumptions))


# ----------------------------------------------------------------------------- explain_change
def _periods(vp: ValidatedPlan) -> Tuple[pd.Series, pd.Series, str, str, str]:
    p, d, tc = vp.plan, vp.data, vp.time_column
    w = d.dropna(subset=[tc])
    if p.comparison_start or p.baseline_start:
        def mask(a, b):
            lo = pd.to_datetime(a) if a else w[tc].min()
            hi = pd.to_datetime(b) if b else w[tc].max()
            if b and len(str(b)) <= 10:
                hi = hi + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            return (w[tc] >= lo) & (w[tc] <= hi)
        if not (p.comparison_start and p.baseline_start):
            raise UDIError("INVALID_PLAN", "Comparing two periods needs both a baseline period and a comparison period.")
        A, B = mask(p.baseline_start, p.baseline_end), mask(p.comparison_start, p.comparison_end)
        la = f"{p.baseline_start}…{p.baseline_end or ''}".rstrip("…")
        lb = f"{p.comparison_start}…{p.comparison_end or ''}".rstrip("…")
        return A, B, la, lb, "custom"
    freq = p.frequency
    if not freq:
        span = (w[tc].max() - w[tc].min()).days
        freq = "Q" if span > 3 * 365 else "M" if span > 120 else "W" if span > 28 else "D"
    per = to_period(w[tc], freq)
    uniq = sorted(per.unique())
    if len(uniq) < 2:
        raise UDIError("INSUFFICIENT_DATA", "Two periods are needed to explain a change; the data covers only one.")
    last, prev = uniq[-1], uniq[-2]
    # an incomplete latest period would fake a decline: use the last two complete ones when the latest is partial
    if len(uniq) >= 3 and w.loc[per == last, tc].max() < last.end_time - (last.end_time - last.start_time) * 0.5:
        last, prev = uniq[-2], uniq[-3]
    return per == prev, per == last, period_label(prev, freq), period_label(last, freq), freq


def explain_change(vp: ValidatedPlan, ctx: ExecContext) -> Dict[str, Any]:
    p, d, tc = vp.plan, vp.data, vp.time_column
    tgt = p.target_column
    sem = get_semantics(ctx, d)
    agg = p.aggregation if p.aggregation in ("sum", "mean", "count") else (default_aggregation(sem, tgt) if tgt else "count")
    w = d.dropna(subset=[tc]).copy()
    A, B, la, lb, freq = _periods(vp)
    A, B = A.reindex(w.index, fill_value=False), B.reindex(w.index, fill_value=False)
    a, b = w[A], w[B]
    if len(a) == 0 or len(b) == 0:
        raise UDIError("INSUFFICIENT_DATA", f"One of the periods has no records (baseline {len(a)}, comparison {len(b)}).")

    def total(x):
        if agg == "count":
            return float(len(x))
        return float(x[tgt].sum()) if agg == "sum" else float(x[tgt].mean())
    va, vb = total(a), total(b)
    delta = vb - va
    pct = delta / abs(va) * 100 if va else None
    dims = p.group_by or dimension_columns(sem, ctx.profile, "general", max_unique=40, limit=6)
    dims = [g for g in dims if g in w.columns and g != tgt]
    if not dims:
        raise UDIError("INSUFFICIENT_DATA", "No categorical column is available to break the change down by.")
    by_dim, best = [], None
    for g in dims:
        rows = []
        keys = sorted(set(a[g].dropna().astype(str)) | set(b[g].dropna().astype(str)))
        if agg in ("sum", "count"):
            ga = a.groupby(a[g].astype(str))[tgt].sum() if agg == "sum" else a.groupby(a[g].astype(str)).size()
            gb = b.groupby(b[g].astype(str))[tgt].sum() if agg == "sum" else b.groupby(b[g].astype(str)).size()
            for k in keys:
                x, y = float(ga.get(k, 0.0)), float(gb.get(k, 0.0))
                rows.append({g: k, "baseline": x, "comparison": y, "contribution": y - x})
        else:
            na, nb = len(a), len(b)
            ma, mb = a.groupby(a[g].astype(str))[tgt].mean(), b.groupby(b[g].astype(str))[tgt].mean()
            wa, wb = a[g].astype(str).value_counts() / na, b[g].astype(str).value_counts() / nb
            for k in keys:
                m_a = float(ma.get(k, np.nan)) if k in ma else np.nan
                m_b = float(mb.get(k, np.nan)) if k in mb else np.nan
                w_a, w_b = float(wa.get(k, 0.0)), float(wb.get(k, 0.0))
                mix = (w_b - w_a) * (m_a if not np.isnan(m_a) else (m_b if not np.isnan(m_b) else 0.0))
                rate = w_b * ((m_b - m_a) if not (np.isnan(m_a) or np.isnan(m_b)) else 0.0)
                rows.append({g: k, "baseline": m_a, "comparison": m_b, "share_baseline_%": w_a * 100, "share_comparison_%": w_b * 100,
                             "mix_effect": mix, "rate_effect": rate, "contribution": mix + rate})
        rows.sort(key=lambda r: -abs(r["contribution"]))
        tot_abs = sum(abs(r["contribution"]) for r in rows) or 1.0
        top_share = abs(rows[0]["contribution"]) / tot_abs if rows else 0
        same_dir = [r for r in rows if np.sign(r["contribution"]) == np.sign(delta) and delta != 0]
        explained = (sum(r["contribution"] for r in same_dir[:2]) / delta) if delta else 0.0
        entry = {"dimension": g, "groups": len(rows), "concentration": top_share, "top_group": rows[0][g] if rows else None,
                 "top_contribution": rows[0]["contribution"] if rows else 0.0, "top2_explain_%": explained * 100, "rows": rows}
        by_dim.append(entry)
        if best is None or entry["concentration"] > best["concentration"]:
            best = entry
    by_dim.sort(key=lambda e: -e["concentration"])
    label = f"{AGG_WORD.get(agg, agg)} {tgt}" if tgt else "Record count"
    direction = "rose" if delta > 0 else "fell" if delta < 0 else "did not change"
    top = best["rows"][:3]
    lead = ", ".join(f"{best['dimension']}='{r[best['dimension']]}' ({'+' if r['contribution'] >= 0 else ''}{fmt_num(r['contribution'])})" for r in top)
    summary = (f"{label} {direction} from {fmt_num(va)} ({la}) to {fmt_num(vb)} ({lb}), a change of {'+' if delta >= 0 else ''}{fmt_num(delta)}"
               + (f" ({pct:+.1f}%)" if pct is not None else "") + f". The change is most concentrated in {best['dimension']}: largest contributions {lead}.")
    if agg == "mean":
        mix_t = sum(r["mix_effect"] for r in best["rows"])
        rate_t = sum(r["rate_effect"] for r in best["rows"])
        summary += f" Mix effect {fmt_num(mix_t)}, rate effect {fmt_num(rate_t)} (by {best['dimension']})."
    wf, running = [{"label": f"{la}", "start": 0.0, "delta": va, "kind": "base", "value": va}], va
    shown = best["rows"][:8]
    for r in shown:
        c = r["contribution"]
        wf.append({"label": f"{r[best['dimension']]}", "start": running if c >= 0 else running + c, "delta": abs(c), "kind": "up" if c >= 0 else "down", "value": c})
        running += c
    rest = sum(r["contribution"] for r in best["rows"][8:])
    if abs(rest) > 1e-9:
        wf.append({"label": "other groups", "start": running if rest >= 0 else running + rest, "delta": abs(rest), "kind": "up" if rest >= 0 else "down", "value": rest})
    wf.append({"label": f"{lb}", "start": 0.0, "delta": vb, "kind": "total", "value": vb})
    charts = [{"type": "waterfall", "title": f"What drove the change in {label} ({la} → {lb}), by {best['dimension']}", "x": "label", "data": wf},
              {"type": "bar", "title": "How concentrated the change is in each dimension", "x": "dimension", "y": "concentration_%",
               "data": [{"dimension": e["dimension"], "concentration_%": e["concentration"] * 100} for e in by_dim], "horizontal": True}]
    insights = []
    for e in by_dim[:3]:
        r0 = e["rows"][0]
        insights.append(f"By {e['dimension']}: '{r0[e['dimension']]}' contributed {'+' if r0['contribution'] >= 0 else ''}{fmt_num(r0['contribution'])} "
                        f"({e['concentration']:.0%} of the gross movement across {e['groups']} groups).")
    metrics = {"Baseline": va, "Comparison": vb, "Change": delta, "Change %": pct, "Baseline period": la, "Comparison period": lb, "Main driver dimension": best["dimension"]}
    return make_result("explain_change", summary, title=f"Why did {label} change?", metrics=metrics, table=records(pd.DataFrame(best["rows"]).head(30)),
                       table_title=f"Contribution to the change by {best['dimension']}", tables={f"by_{e['dimension']}": e["rows"][:15] for e in by_dim[1:4]},
                       charts=charts, insights=insights, business={"root_cause": [{k: v for k, v in e.items() if k != "rows"} for e in by_dim]},
                       limitations=["The decomposition is an exact accounting of the change across groups; it shows WHERE the change happened, not WHY (it is not causal).",
                                    f"Periods compared: {la} (baseline, {len(a):,} records) and {lb} (comparison, {len(b):,} records)."],
                       assumptions=list(vp.assumptions) + ([f"No periods were named, so the last two complete {FREQ_NAME.get(freq, freq)}s were compared."] if freq != "custom" else []))


# ----------------------------------------------------------------------------- segment_records
def segment_records(vp: ValidatedPlan, ctx: ExecContext) -> Dict[str, Any]:
    p, d = vp.plan, vp.data
    roles = ctx.roles
    feats = [c for c in (p.feature_columns or []) if roles.get(c, {}).get("role") == "numeric"]
    if not feats:
        feats = [c for c in d.columns if roles.get(c, {}).get("role") == "numeric" and d[c].nunique(dropna=True) > 2 and d[c].isna().mean() < 0.5][:12]
    if len(feats) < 2:
        raise UDIError("INSUFFICIENT_DATA", "Segmentation needs at least two numeric columns with variation.")
    work = d.dropna(subset=feats, how="all")
    if len(work) < 30:
        raise UDIError("INSUFFICIENT_DATA", f"Segmentation needs at least 30 records; {len(work)} are available.")
    X = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(work[feats].astype(float)))
    sample = np.random.RandomState(ctx.seed).choice(len(X), min(len(X), 4000), replace=False)
    scores = []
    ks = [p.n_clusters] if p.n_clusters else list(range(2, 7))
    for k in ks:
        if k >= len(work):
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=ctx.seed).fit(X[sample])
        sil = float(silhouette_score(X[sample], km.labels_)) if k > 1 else 0.0
        scores.append({"k": k, "silhouette": sil, "inertia": float(km.inertia_)})
    best_k = max(scores, key=lambda s: s["silhouette"])["k"]
    km = KMeans(n_clusters=best_k, n_init=10, random_state=ctx.seed).fit(X)
    labels = km.labels_
    overall = work[feats].astype(float).mean()
    overall_sd = work[feats].astype(float).std().replace(0, 1)
    profiles, names = [], {}
    for k in range(best_k):
        sub = work[labels == k]
        means = sub[feats].astype(float).mean()
        z = ((means - overall) / overall_sd).sort_values(key=lambda s: -s.abs())
        desc = ", ".join(f"{'high' if z[c] > 0 else 'low'} {c}" for c in z.index[:3] if abs(z[c]) >= 0.25) or "close to average on all features"
        names[k] = f"Segment {k + 1}"
        row = {"segment": names[k], "records": int(len(sub)), "share_%": len(sub) / len(work) * 100, "profile": desc}
        row.update({f"avg_{c}": float(means[c]) for c in feats[:6]})
        cats = [c for c in d.columns if roles.get(c, {}).get("role") in ("categorical", "binary") and d[c].nunique() <= 20][:2]
        for c in cats:
            vc = sub[c].astype(str).value_counts(normalize=True)
            if len(vc):
                row[f"top_{c}"] = f"{vc.index[0]} ({vc.iloc[0]:.0%})"
        profiles.append(row)
    pts = PCA(n_components=2, random_state=ctx.seed).fit_transform(X)
    idx = np.random.RandomState(ctx.seed).choice(len(pts), min(len(pts), 800), replace=False)
    scatter = [{"pc1": float(pts[i, 0]), "pc2": float(pts[i, 1]), "segment": names[int(labels[i])]} for i in idx]
    sil = max(s["silhouette"] for s in scores)
    summary = (f"{best_k} segments were found among {len(work):,} records using K-means on {len(feats)} standardised numeric columns "
               f"(k chosen by silhouette = {sil:.2f}). Largest: {max(profiles, key=lambda r: r['records'])['segment']} "
               f"({max(r['share_%'] for r in profiles):.0f}% of records). " + " ".join(f"{r['segment']}: {r['profile']}." for r in profiles[:4]))
    lim = ["Segments describe similarity in the selected numeric columns; they are a starting point for targeting, not fixed customer types."]
    if sil < 0.25:
        lim.append(f"Silhouette {sil:.2f} is low: segments overlap considerably.")
    return make_result("segment_records", summary, title=f"{best_k} segments", metrics={"Segments": best_k, "Silhouette": sil, "Records": len(work), "Features": len(feats)},
                       table=profiles, table_title="Segment profiles (averages and dominant categories)", tables={"k_selection": scores},
                       charts=[{"type": "pie", "title": "Segment sizes", "x": "segment", "y": "records", "data": [{"segment": r["segment"], "records": r["records"]} for r in profiles]},
                               {"type": "scatter", "title": "Segments on the first two principal components", "x": "pc1", "y": "pc2", "group": "segment", "data": scatter},
                               {"type": "line", "title": "Choosing k: silhouette score", "x": "k", "y": ["silhouette"], "data": scores}],
                       limitations=lim, assumptions=list(vp.assumptions) + [f"Features used: {', '.join(feats)} (median-imputed and standardised)."])
