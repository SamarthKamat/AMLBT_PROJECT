"""detect_anomalies: IQR, z-score and Isolation Forest on real data (no domain assumptions)."""
from __future__ import annotations
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from context import ExecContext
from errors import UDIError
from result_formatter import fmt_num, make_result, records
from validator import ValidatedPlan, numeric_columns

CAVEAT = "An anomaly flag marks a statistically unusual record; it does not mean the value is wrong or erroneous. Review flagged records with domain knowledge."


def _context_columns(d: pd.DataFrame, time_col, exclude, roles, k: int = 5) -> List[str]:
    cols = []
    if time_col and time_col in d.columns:
        cols.append(time_col)
    for c in d.columns:
        if c in exclude or c in cols or roles.get(c, {}).get("role") in ("identifier", "text", "empty"):
            continue
        cols.append(c)
        if len(cols) >= k + (1 if time_col else 0):
            break
    return cols


def _chart_points(order: pd.Index, values: pd.Series, flagged: pd.Series, labels: pd.Series, limit: int = 600):
    df = pd.DataFrame({"row": np.arange(len(values)), "value": values.to_numpy(dtype=float), "anomaly": flagged.to_numpy(), "label": labels.to_numpy()})
    normal, anom = df[~df["anomaly"]], df[df["anomaly"]].head(300)
    if len(normal) > limit:
        normal = normal.iloc[np.unique(np.linspace(0, len(normal) - 1, limit).round().astype(int))]
    return records(pd.concat([normal, anom]).sort_values("row"))


def detect_anomalies(vp: ValidatedPlan, ctx: ExecContext) -> Dict[str, Any]:
    p, d, roles = vp.plan, vp.data, ctx.roles
    tgt = p.target_column
    top_n = p.top_n or 20
    method = p.anomaly_method or "auto"
    if tgt is None:
        cols = [c for c in numeric_columns(d, roles) if roles.get(c, {}).get("role") not in ("constant", "identifier") and d[c].nunique(dropna=True) > 1]
        if not cols:
            raise UDIError("INSUFFICIENT_DATA", "No numeric column with variation is available for anomaly detection.")
        if len(cols) == 1:
            tgt = cols[0]
            vp.assumptions.append(f"Only one numeric column ('{tgt}') exists, so anomalies are detected on it.")
    if tgt is not None:
        return _univariate(d, tgt, method, top_n, vp, ctx)
    return _multivariate(d, cols[:30], top_n, vp, ctx)


def _univariate(d: pd.DataFrame, tgt: str, method: str, top_n: int, vp: ValidatedPlan, ctx: ExecContext) -> Dict[str, Any]:
    work = d.dropna(subset=[tgt]).copy()
    x = work[tgt].astype(float)
    if len(x) < 20:
        raise UDIError("INSUFFICIENT_DATA", f"Anomaly detection needs at least 20 values of '{tgt}'; only {len(x)} are available.")
    if x.nunique() < 3:
        raise UDIError("INSUFFICIENT_DATA", f"'{tgt}' has fewer than 3 distinct values, so unusual values cannot be identified.")
    q1, q3 = x.quantile(.25), x.quantile(.75)
    iqr = q3 - q1
    if iqr == 0:                                       # constant core: fall back to MAD for a usable fence
        mad = float((x - x.median()).abs().median()) or float(x.std())
        lo, hi = x.median() - 3 * 1.4826 * mad, x.median() + 3 * 1.4826 * mad
    else:
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    iqr_flag = (x < lo) | (x > hi)
    sd = x.std()
    z = (x - x.mean()) / sd if sd > 0 else pd.Series(0.0, index=x.index)
    z_flag = z.abs() > 3
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=ctx.seed)
    iso.fit(x.to_numpy().reshape(-1, 1))
    iso_flag = pd.Series(iso.predict(x.to_numpy().reshape(-1, 1)) == -1, index=x.index)
    score = pd.Series(-iso.score_samples(x.to_numpy().reshape(-1, 1)), index=x.index)

    votes = iqr_flag.astype(int) + z_flag.astype(int) + iso_flag.astype(int)
    flag = {"iqr": iqr_flag, "zscore": z_flag, "isolation_forest": iso_flag, "auto": votes >= 2}[method]
    rule = {"iqr": "the IQR rule (1.5×IQR fences)", "zscore": "|z-score| > 3", "isolation_forest": "Isolation Forest",
            "auto": "at least two of three methods (IQR fences, |z| > 3, Isolation Forest)"}[method]
    methods_txt = pd.Series([", ".join(n for n, f in (("IQR", iqr_flag[i]), ("z-score", z_flag[i]), ("IsolationForest", iso_flag[i])) if f) for i in x.index], index=x.index)
    n_flag = int(flag.sum())
    tc = vp.time_column
    ctx_cols = _context_columns(work, tc, {tgt}, ctx.roles)
    out = work.loc[flag, [tgt] + ctx_cols].copy()
    out.insert(0, "row_index", out.index)
    out["z_score"] = z.loc[out.index]
    out["methods_flagging"] = methods_txt.loc[out.index]
    out["isolation_score"] = score.loc[out.index]
    out = out.sort_values("isolation_score", ascending=False)
    lbl = work[tc].dt.strftime("%Y-%m-%d") if tc and tc in work.columns else pd.Series(work.index.astype(str), index=work.index)
    charts = [{"type": "scatter", "title": f"{tgt}: flagged records", "x": "row", "y": "value", "highlight": "anomaly", "label": "label",
               "data": _chart_points(work.index, x, flag, lbl)}]
    metrics = {"Records analysed": int(len(x)), "Flagged": n_flag, "Flagged %": n_flag / len(x) * 100, "Rule": rule,
               "Lower fence": float(lo), "Upper fence": float(hi), "IQR flags": int(iqr_flag.sum()), "z-score flags": int(z_flag.sum()), "Isolation Forest flags": int(iso_flag.sum())}
    if n_flag:
        hi_ct, lo_ct = int((x[flag] > hi).sum()), int((x[flag] < lo).sum())
        summary = (f"{n_flag:,} of {len(x):,} values of '{tgt}' ({n_flag / len(x) * 100:.1f}%) are flagged as unusual under {rule}: "
                   f"{hi_ct} above {fmt_num(hi)} and {lo_ct} below {fmt_num(lo)}.")
    else:
        summary = f"No unusual values of '{tgt}' were flagged under {rule} among {len(x):,} records."
    lim = [CAVEAT]
    if method == "auto":
        lim.append("'auto' requires agreement between at least two methods, so it is deliberately conservative; choose a single method for wider coverage.")
    return make_result("detect_anomalies", summary, title=f"Unusual values of {tgt}", metrics=metrics, table=records(out, top_n),
                       table_title=f"Top {min(top_n, n_flag)} flagged records (most isolated first)" if n_flag else "Flagged records",
                       charts=charts, limitations=lim, assumptions=list(vp.assumptions))


def _multivariate(d: pd.DataFrame, cols: List[str], top_n: int, vp: ValidatedPlan, ctx: ExecContext) -> Dict[str, Any]:
    keep = [c for c in cols if d[c].isna().mean() <= 0.5]
    if not keep:
        raise UDIError("INSUFFICIENT_DATA", "Numeric columns are too sparse (over 50% missing) for multivariate anomaly detection.")
    X = d[keep].astype(float)
    Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=ctx.seed)
    pred = iso.fit_predict(Xs)
    score = -iso.score_samples(Xs)
    flag = pd.Series(pred == -1, index=d.index)
    n_flag = int(flag.sum())
    zdf = pd.DataFrame(Xs, columns=keep, index=d.index)
    tc = vp.time_column
    ctx_cols = [c for c in _context_columns(d, tc, set(), ctx.roles, k=4)]
    out = d.loc[flag, list(dict.fromkeys(ctx_cols + keep[:4]))].copy()
    out.insert(0, "row_index", out.index)
    out["anomaly_score"] = pd.Series(score, index=d.index).loc[out.index]
    drivers = []
    for i in out.index:
        top = zdf.loc[i].abs().sort_values(ascending=False).head(3)
        drivers.append(", ".join(f"{c} ({zdf.loc[i, c]:+.1f}σ)" for c in top.index))
    out["main_drivers"] = drivers
    out = out.sort_values("anomaly_score", ascending=False)
    # 2-D view of the two most variable columns for the chart
    lbl = d[tc].dt.strftime("%Y-%m-%d") if tc and tc in d.columns else pd.Series(d.index.astype(str), index=d.index)
    chart = [{"row": int(i), "value": float(sc), "anomaly": bool(f), "label": l}
             for i, (sc, f, l) in enumerate(zip(score, flag.to_numpy(), lbl.to_numpy()))]
    normal = [c for c in chart if not c["anomaly"]]
    if len(normal) > 600:
        normal = [normal[j] for j in np.unique(np.linspace(0, len(normal) - 1, 600).round().astype(int))]
    chart = sorted(normal + [c for c in chart if c["anomaly"]][:300], key=lambda c: c["row"])
    metrics = {"Records analysed": int(len(d)), "Columns used": len(keep), "Flagged": n_flag, "Flagged %": n_flag / len(d) * 100, "Method": "Isolation Forest (multivariate)"}
    if n_flag:
        summary = (f"{n_flag:,} of {len(d):,} records ({n_flag / len(d) * 100:.1f}%) look unusual across {len(keep)} numeric columns "
                   f"({', '.join(keep[:6])}{'…' if len(keep) > 6 else ''}) according to Isolation Forest. The most unusual record is row {int(out.iloc[0]['row_index'])}.")
    else:
        summary = f"No records were flagged as unusual across {len(keep)} numeric columns among {len(d):,} records."
    return make_result("detect_anomalies", summary, title="Unusual records", metrics=metrics, table=records(out, top_n),
                       table_title=f"Top {min(top_n, n_flag)} flagged records", charts=[{"type": "scatter", "title": "Anomaly score by record", "x": "row", "y": "value",
                                                                                      "yLabel": "anomaly score", "highlight": "anomaly", "label": "label", "data": chart}] if n_flag or chart else [],
                       limitations=[CAVEAT, "Multivariate detection uses Isolation Forest with scikit-learn's default 'auto' threshold on standardised numeric columns; categorical columns are not used."],
                       assumptions=list(vp.assumptions))
