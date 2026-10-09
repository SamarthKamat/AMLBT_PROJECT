"""Deterministic diagnostics: analyze_trend and calculate_correlation (no ML models)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from context import ExecContext
from errors import UDIError
from result_formatter import fmt_num, make_result, records
from validator import ValidatedPlan, numeric_columns

PERIOD_ALIAS = {"H": "h", "D": "D", "W": "W-SUN", "M": "M", "Q": "Q", "Y": "Y"}
FREQ_ORDER = ["H", "D", "W", "M", "Q", "Y"]
FREQ_NAME = {"H": "hour", "D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}
AGG_FN = {"mean": "mean", "sum": "sum", "median": "median", "min": "min", "max": "max", "std": "std", "count": "count", "nunique": "nunique"}


# ----------------------------------------------------------------------------- time helpers (shared with forecasting)
def native_frequency(ts: pd.Series) -> str:
    u = pd.Series(pd.to_datetime(ts.dropna().unique())).sort_values()
    if len(u) < 3:
        return "D"
    gap = u.diff().dropna().dt.total_seconds().median() / 86400.0
    if gap < 0.9:
        return "H"
    if gap < 1.5:
        return "D"
    if 5.5 <= gap <= 8.5:
        return "W"
    if 26 <= gap <= 33:
        return "M"
    if 80 <= gap <= 100:
        return "Q"
    if 350 <= gap <= 380:
        return "Y"
    return "D" if gap < 26 else "M"


def to_period(ts: pd.Series, freq: str) -> pd.Series:
    return pd.to_datetime(ts).dt.to_period(PERIOD_ALIAS[freq])


def n_periods(ts: pd.Series, freq: str) -> int:
    return int(to_period(ts.dropna(), freq).nunique())


def fit_frequency(ts: pd.Series, requested: Optional[str], max_periods: int = 120) -> str:
    """Requested frequency if given, else the native one coarsened until the series is at most `max_periods` long."""
    freq = requested or native_frequency(ts)
    i = FREQ_ORDER.index(freq)
    if requested:
        return freq
    while i < len(FREQ_ORDER) - 1 and n_periods(ts, FREQ_ORDER[i]) > max_periods:
        i += 1
    return FREQ_ORDER[i]


def period_label(p: pd.Period, freq: str) -> str:
    ts = p.start_time
    return {"H": ts.strftime("%Y-%m-%d %H:00"), "D": ts.strftime("%Y-%m-%d"), "W": ts.strftime("%Y-%m-%d"),
            "M": ts.strftime("%Y-%m"), "Q": f"{ts.year}-Q{(ts.month - 1) // 3 + 1}", "Y": str(ts.year)}[freq]


def aggregate_series(d: pd.DataFrame, time_col: str, target: Optional[str], agg: str, freq: str, fill_gaps: bool = True) -> pd.Series:
    """Aggregate to one value per period. Index = Period (regular if fill_gaps). Missing periods stay NaN."""
    work = d.dropna(subset=[time_col]).copy()
    work["_p"] = to_period(work[time_col], freq)
    if target is None:
        s = work.groupby("_p").size().astype(float)
    else:
        s = getattr(work.groupby("_p")[target], AGG_FN[agg])().astype(float)
    if fill_gaps and len(s):
        full = pd.period_range(s.index.min(), s.index.max(), freq=PERIOD_ALIAS[freq])
        s = s.reindex(full)
    return s


# ----------------------------------------------------------------------------- trend
def _trend_stats(y: np.ndarray) -> Dict[str, Any]:
    x = np.arange(len(y), dtype=float)
    ok = ~np.isnan(y)
    xs, ys = x[ok], y[ok]
    lr = stats.linregress(xs, ys)
    tau, p = stats.kendalltau(xs, ys)
    fit_start, fit_end = lr.intercept, lr.intercept + lr.slope * (len(y) - 1)
    change_pct = (fit_end - fit_start) / abs(fit_start) * 100 if fit_start not in (0, 0.0) else None
    significant = bool(p < 0.05)
    direction = "no statistically significant trend" if not significant else ("increasing" if tau > 0 else "decreasing")
    return {"slope": float(lr.slope), "intercept": float(lr.intercept), "r2": float(lr.rvalue ** 2), "tau": float(tau), "p": float(p),
            "fit_start": float(fit_start), "fit_end": float(fit_end), "change_pct": change_pct, "direction": direction,
            "fitted": lr.intercept + lr.slope * x}


def analyze_trend(vp: ValidatedPlan, ctx: ExecContext) -> Dict[str, Any]:
    p, d, tc = vp.plan, vp.data, vp.time_column
    tgt, agg = p.target_column, p.aggregation or ("mean" if p.target_column else "count")
    if tgt is None:
        agg = "count"
    freq = fit_frequency(d[tc], p.frequency)
    measure = f"{agg} of {tgt}" if tgt else "record count"
    measure_title = f"{tgt}" if tgt else "Records"
    assumptions = list(vp.assumptions)
    native = native_frequency(d[tc])
    if freq != native and not p.frequency:
        assumptions.append(f"Data was aggregated by {FREQ_NAME[freq]} (native spacing is {FREQ_NAME[native]}) to keep the series readable.")

    # ---- per-group trends
    if p.group_by:
        g = p.group_by[0]
        top_groups = d[g].value_counts().head(6).index.tolist()
        series = {}
        rows = []
        for key in top_groups:
            s = aggregate_series(d[d[g] == key], tc, tgt, agg, freq)
            if s.notna().sum() >= 6:
                series[str(key)] = s
                st = _trend_stats(s.to_numpy())
                rows.append({g: str(key), "periods": int(s.notna().sum()), "direction": st["direction"],
                             "change_%": st["change_pct"], "slope_per_period": st["slope"], "kendall_tau": st["tau"], "p_value": st["p"]})
        if not series:
            raise UDIError("INSUFFICIENT_DATA", f"No group of '{g}' has at least 6 {FREQ_NAME[freq]}s of data for a trend.")
        idx = sorted(set().union(*[set(s.index) for s in series.values()]))
        chart_rows = [{"period": period_label(i, freq), **{k: (None if pd.isna(s.get(i, np.nan)) else float(s.get(i))) for k, s in series.items()}} for i in idx]
        inc = [r[g] for r in rows if r["direction"] == "increasing"]
        dec = [r[g] for r in rows if r["direction"] == "decreasing"]
        summary = (f"{measure_title} by {g} ({FREQ_NAME[freq]}ly): increasing for {', '.join(inc) or 'no group'}, decreasing for {', '.join(dec) or 'no group'}; "
                   f"the other groups show no statistically significant trend.")
        return make_result("analyze_trend", summary, title=f"{measure_title} trend by {g}", table=rows, table_title=f"Trend per {g}",
                           metrics={"Groups analysed": len(rows), "Period": FREQ_NAME[freq]},
                           charts=[{"type": "line", "title": f"{measure_title} over time by {g}", "x": "period", "y": list(series.keys()), "data": chart_rows}],
                           limitations=["Only the 6 largest groups are shown.", "Trend significance uses the Mann–Kendall test, which assumes independent observations; autocorrelation can overstate significance.",
                                        "A trend describes direction in the data and does not explain its cause."], assumptions=assumptions)

    s = aggregate_series(d, tc, tgt, agg, freq)
    valid = int(s.notna().sum())
    if valid < 6:
        raise UDIError("INSUFFICIENT_DATA", f"Only {valid} {FREQ_NAME[freq]}(s) of data are available; a trend needs at least 6.", details={"periods": valid})
    st = _trend_stats(s.to_numpy())
    labels = [period_label(i, freq) for i in s.index]
    chart_rows = [{"period": lab, "value": None if pd.isna(v) else float(v), "trend": float(f)} for lab, v, f in zip(labels, s.to_numpy(), st["fitted"])]
    first, last = float(s.dropna().iloc[0]), float(s.dropna().iloc[-1])
    gaps = int(s.isna().sum())
    unit = FREQ_NAME[freq]
    word = {"increasing": "rising", "decreasing": "falling"}.get(st["direction"])
    if word:
        summary = (f"{measure_title} is {word} over time: the fitted trend changes by {fmt_num(st['change_pct'])}% from the first to the last {unit} "
                   f"({fmt_num(st['slope'])} per {unit}); Mann–Kendall p = {st['p']:.3g}.") if st["change_pct"] is not None else \
                  (f"{measure_title} is {word} over time ({fmt_num(st['slope'])} per {unit}); Mann–Kendall p = {st['p']:.3g}.")
    else:
        summary = f"There is no statistically significant trend in {measure_title} over {valid} {unit}s (Mann–Kendall p = {st['p']:.3g})."
    summary += f" First {unit}: {fmt_num(first)}, latest {unit}: {fmt_num(last)}."
    metrics = {"Direction": st["direction"], "First period": first, "Latest period": last, "Fitted change %": st["change_pct"],
               f"Slope per {unit}": st["slope"], "Kendall tau": st["tau"], "p-value": st["p"], "R² (linear fit)": st["r2"], "Periods": valid}
    lim = ["Trend significance uses the Mann–Kendall test, which assumes independent observations; autocorrelation can overstate significance.",
           "A trend describes direction in the data and does not explain its cause."]
    if gaps:
        lim.append(f"{gaps} {unit}(s) in the range have no data and are shown as gaps.")
    tail = [{"period": lab, "value": None if pd.isna(v) else float(v)} for lab, v in zip(labels[-24:], s.to_numpy()[-24:])]
    return make_result("analyze_trend", summary, title=f"{measure_title} over time", metrics=metrics, table=tail, table_title=f"Most recent {len(tail)} {unit}s",
                       charts=[{"type": "line", "title": f"{measure_title} ({measure}) per {unit}", "x": "period", "y": ["value", "trend"], "data": chart_rows}],
                       limitations=lim, assumptions=assumptions)


# ----------------------------------------------------------------------------- correlation
def _strength_label(v: float) -> str:
    a = abs(v)
    return "negligible" if a < 0.1 else "weak" if a < 0.3 else "moderate" if a < 0.5 else "strong" if a < 0.7 else "very strong"


def _eta(cat: pd.Series, num: pd.Series) -> Optional[float]:
    df = pd.DataFrame({"c": cat.astype(str), "y": num}).dropna()
    if df["c"].nunique() < 2 or len(df) < 5:
        return None
    mu = df["y"].mean()
    sst = float(((df["y"] - mu) ** 2).sum())
    if sst == 0:
        return None
    ssb = float(sum(len(g) * (g["y"].mean() - mu) ** 2 for _, g in df.groupby("c")))
    return float(np.sqrt(max(0.0, ssb / sst)))


def _cramers_v(a: pd.Series, b: pd.Series):
    ct = pd.crosstab(a.astype(str), b.astype(str))
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return None, None
    chi2, p, _, _ = stats.chi2_contingency(ct.values)
    n = ct.values.sum()
    v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1)))) if n else None
    return v, float(p)


def _kruskal_p(cat: pd.Series, num: pd.Series) -> Optional[float]:
    df = pd.DataFrame({"c": cat.astype(str), "y": num}).dropna()
    samples = [g["y"].to_numpy() for _, g in df.groupby("c") if len(g) >= 3]
    if len(samples) < 2:
        return None
    try:
        return float(stats.kruskal(*samples).pvalue)
    except Exception:
        return None


def calculate_correlation(vp: ValidatedPlan, ctx: ExecContext) -> Dict[str, Any]:
    p, d, roles = vp.plan, vp.data, ctx.roles
    tgt = p.target_column
    primary = "spearman" if p.correlation_method == "spearman" else "pearson"
    nums = [c for c in numeric_columns(d, roles) if roles.get(c, {}).get("role") != "constant" and d[c].nunique(dropna=True) > 1]
    lim = ["Correlation does not establish causation.",
           "Associations are measured on the selected rows only and may differ for other periods or segments."]

    if tgt is None:
        cols = nums[:25]
        if len(cols) < 2:
            raise UDIError("INSUFFICIENT_DATA", f"Correlation needs at least two numeric columns with variation; found {len(cols)}.")
        pear, spear = d[cols].corr(method="pearson"), d[cols].corr(method="spearman")
        pairs = []
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                pair = d[[a, b]].dropna()
                if len(pair) < 20:
                    continue
                pr, sr = pear.loc[a, b], spear.loc[a, b]
                pairs.append((a, b, pr, sr, len(pair)))
        if not pairs:
            raise UDIError("INSUFFICIENT_DATA", "No pair of numeric columns has at least 20 rows with both values.")
        key = (lambda t: abs(t[3])) if primary == "spearman" else (lambda t: abs(t[2]))
        pairs.sort(key=key, reverse=True)
        rows = []
        for a, b, pr, sr, n in pairs[:30]:
            pair = d[[a, b]].dropna()
            pv = float((stats.spearmanr if primary == "spearman" else stats.pearsonr)(pair[a], pair[b])[1])
            main = sr if primary == "spearman" else pr
            rows.append({"variable_1": a, "variable_2": b, "pearson_r": float(pr), "spearman_rho": float(sr), "p_value": pv, "n": n, "strength": _strength_label(main)})
        top = rows[0]
        summary = (f"The strongest relationship among numeric columns is between '{top['variable_1']}' and '{top['variable_2']}' "
                   f"({primary} = {fmt_num(top['pearson_r'] if primary == 'pearson' else top['spearman_rho'], 3)}, {top['strength']}). "
                   f"{len(pairs)} pairs were examined. Correlation does not establish causation.")
        chart = [{"pair": f"{r['variable_1']} × {r['variable_2']}", "correlation": r["pearson_r"] if primary == "pearson" else r["spearman_rho"]} for r in rows[:12]]
        return make_result("calculate_correlation", summary, title="Strongest correlations", table=rows, table_title="Variable pairs ranked by strength",
                           metrics={"Pairs examined": len(pairs), "Strongest pair": f"{top['variable_1']} × {top['variable_2']}", "Strongest value": top["pearson_r"] if primary == "pearson" else top["spearman_rho"]},
                           charts=[{"type": "bar", "title": "Strongest pairwise correlations", "x": "pair", "y": "correlation", "data": chart, "horizontal": True}],
                           limitations=lim + (["More than 25 numeric columns exist; only the first 25 were analysed."] if len(numeric_columns(d, roles)) > 25 else []))

    # --- factors associated with a target
    y = d[tgt]
    y_numeric = pd.api.types.is_numeric_dtype(y)
    rows = []
    for c in d.columns:
        if c == tgt:
            continue
        role = roles.get(c, {}).get("role")
        if role in ("identifier", "text", "constant", "empty", "datetime"):
            continue
        s = d[c]
        pair = pd.DataFrame({"x": s, "y": y}).dropna()
        if len(pair) < 20:
            continue
        if y_numeric and pd.api.types.is_numeric_dtype(s):
            if pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
                continue
            pr = stats.pearsonr(pair["x"], pair["y"])
            sr = stats.spearmanr(pair["x"], pair["y"])
            main, pv = (sr[0], sr[1]) if primary == "spearman" else (pr[0], pr[1])
            rows.append({"factor": c, "type": "numeric", "measure": "Spearman ρ" if primary == "spearman" else "Pearson r", "strength": float(main),
                         "pearson_r": float(pr[0]), "spearman_rho": float(sr[0]), "p_value": float(pv), "n": len(pair), "label": _strength_label(main)})
        elif y_numeric and not pd.api.types.is_numeric_dtype(s):
            if s.nunique(dropna=True) > 50:
                continue
            e = _eta(pair["x"], pair["y"])
            if e is None:
                continue
            rows.append({"factor": c, "type": "categorical", "measure": "Correlation ratio η", "strength": e, "pearson_r": None, "spearman_rho": None,
                         "p_value": _kruskal_p(pair["x"], pair["y"]), "n": len(pair), "label": _strength_label(e)})
        elif not y_numeric and pd.api.types.is_numeric_dtype(s):
            e = _eta(pair["y"], pair["x"])
            if e is None:
                continue
            rows.append({"factor": c, "type": "numeric", "measure": "Correlation ratio η", "strength": e, "pearson_r": None, "spearman_rho": None,
                         "p_value": _kruskal_p(pair["y"], pair["x"]), "n": len(pair), "label": _strength_label(e)})
        else:
            if s.nunique(dropna=True) > 50:
                continue
            v, pv = _cramers_v(pair["x"], pair["y"])
            if v is None:
                continue
            rows.append({"factor": c, "type": "categorical", "measure": "Cramér's V", "strength": v, "pearson_r": None, "spearman_rho": None,
                         "p_value": pv, "n": len(pair), "label": _strength_label(v)})
    if not rows:
        raise UDIError("INSUFFICIENT_DATA", f"No other column has at least 20 rows overlapping with '{tgt}' to measure an association.")
    rows.sort(key=lambda r: abs(r["strength"]), reverse=True)
    top = rows[0]
    sig = [r for r in rows if r["p_value"] is not None and r["p_value"] < 0.05 and abs(r["strength"]) >= 0.1]
    lead = ", ".join(f"{r['factor']} ({r['measure']} = {fmt_num(r['strength'], 2)})" for r in rows[:3])
    summary = (f"The columns most strongly associated with '{tgt}' are {lead}. {len(sig)} of {len(rows)} columns show a statistically significant "
               f"association of at least weak strength. Association is not causation.")
    lim2 = list(lim)
    if len(rows) > 10:
        lim2.append("p-values are not adjusted for testing many columns; a few 'significant' results can arise by chance.")
    if not y_numeric:
        lim2.append("Target is categorical: association strengths are correlation ratio (η) and Cramér's V, which range 0 to 1 and are unsigned.")
    chart = [{"factor": r["factor"], "strength": r["strength"]} for r in rows[:12]]
    return make_result("calculate_correlation", summary, title=f"What is associated with {tgt}", table=rows, table_title=f"Association with {tgt}",
                       metrics={"Columns examined": len(rows), "Most associated": top["factor"], "Strongest value": top["strength"], "Significant (p<0.05)": len(sig)},
                       charts=[{"type": "bar", "title": f"Association strength with {tgt}", "x": "factor", "y": "strength", "data": chart, "horizontal": True}],
                       limitations=lim2)
