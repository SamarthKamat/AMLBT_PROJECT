"""Data Quality Intelligence (Phase 2): a 0-100 quality score, an issue list with recommended treatments,
and an auditable automatic cleaning step with a before/after comparison.

Score. Five dimensions d, each with defect rate r_d (share of affected cells/rows) and sensitivity k_d
(completeness uses the mean of the missing-cell rate and the incomplete-row rate):
    S_d = 100 * (1 - min(1, k_d * r_d))          Q = sum_d w_d * S_d
    completeness (w .30, k 1), uniqueness (.20, 4), validity (.20, 4), consistency (.15, 4), outliers (.15, 2)
Rates are measured on the data; the weights and sensitivities are fixed, documented constants.

Nothing is changed silently: clean_dataset() applies only the actions the user selected, logs every change with
counts, and the session keeps the original so the cleaning can be reverted.
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from relationships import date_rules, derived_columns
from semantics import column_tokens

WEIGHTS = {"completeness": 0.30, "uniqueness": 0.20, "validity": 0.20, "consistency": 0.15, "outliers": 0.15}
SENSITIVITY = {"completeness": 1.0, "uniqueness": 4.0, "validity": 4.0, "consistency": 4.0, "outliers": 2.0}
SPARSE_COLUMN = 0.60
ROBUST_Z = 3.5
SIGNED_MONEY = {"profit", "loss", "balance", "margin", "change", "net", "variance", "delta", "growth", "return", "pnl", "gain"}
DEFAULT_ACTIONS = ["standardize_categories", "drop_duplicates", "invalid_to_missing", "impute_missing", "drop_sparse_columns"]
ALL_ACTIONS = DEFAULT_ACTIONS + ["cap_outliers", "drop_constant_columns", "drop_missing_rows"]


def _grade(score: float) -> str:
    return "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 60 else "E"


def _canon(v: str) -> str:
    return re.sub(r"[\s_\-.]+", " ", str(v).strip().casefold())


def category_variants(s: pd.Series) -> List[Dict[str, Any]]:
    """Groups of labels that differ only by case, spacing or punctuation ('North', 'north ', 'NORTH')."""
    nn = s.dropna()
    if nn.empty or pd.api.types.is_numeric_dtype(s):
        return []
    vc = nn.astype(str).value_counts()
    if len(vc) > 2000:
        return []
    groups: Dict[str, List[Tuple[str, int]]] = {}
    for val, cnt in vc.items():
        groups.setdefault(_canon(val), []).append((val, int(cnt)))
    out = []
    for key, members in groups.items():
        if len(members) > 1:
            members.sort(key=lambda m: -m[1])
            out.append({"canonical": members[0][0], "variants": [m[0] for m in members[1:]], "rows_affected": sum(m[1] for m in members[1:])})
    return out


def _non_negative(name: str, sem: str) -> bool:
    toks = set(column_tokens(name))
    if sem in ("quantity", "age", "duration", "score"):
        return not (toks & SIGNED_MONEY)
    if sem == "money":
        return not (toks & SIGNED_MONEY)
    return False


def _invalid_mask(df: pd.DataFrame, name: str, sem: str) -> Tuple[pd.Series, str]:
    s = df[name]
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return pd.Series(False, index=df.index), ""
    x = s.astype(float)
    reasons = []
    mask = pd.Series(False, index=df.index)
    if _non_negative(name, sem):
        neg = x < 0
        if neg.any() and neg.mean() < 0.2:            # a column that is mostly negative is a signed measure, not an error
            mask |= neg
            reasons.append("negative values in a quantity that cannot be negative")
    if sem == "age":
        old = x > 120
        mask |= old
        if old.any():
            reasons.append("ages above 120")
    if sem == "rate":
        nn = x.dropna()
        if len(nn):
            if (nn.between(0, 1)).mean() >= 0.95:
                bad = (x < 0) | (x > 1)
                rule = "outside 0-1"
            elif (nn.between(0, 100)).mean() >= 0.95:
                bad = (x < 0) | (x > 100)
                rule = "outside 0-100"
            else:
                bad, rule = pd.Series(False, index=df.index), ""
            if bad.any():
                mask |= bad
                reasons.append(f"rates {rule}")
    return mask.fillna(False), "; ".join(reasons)


def _robust_outliers(x: pd.Series) -> pd.Series:
    nn = x.dropna().astype(float)
    if len(nn) < 20 or nn.nunique() < 5:
        return pd.Series(False, index=x.index)
    med = nn.median()
    mad = (nn - med).abs().median()
    if mad == 0:
        return pd.Series(False, index=x.index)
    z = 0.6745 * (x.astype(float) - med) / mad             # Iglewicz-Hoaglin modified z-score
    q1, q3 = nn.quantile(0.25), nn.quantile(0.75)
    iqr = q3 - q1
    extreme = (x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)      # Tukey's "far out" fences: both rules must agree
    return ((z.abs() > ROBUST_Z) & extreme).fillna(False)


def _missing_treatment(s: pd.Series, name: str, frac: float, role: str) -> Tuple[str, str]:
    if frac > SPARSE_COLUMN:
        return "drop_column", f"Drop '{name}': {frac:.0%} of values are missing, too sparse to impute reliably."
    if pd.api.types.is_numeric_dtype(s) and role in ("numeric", "binary"):
        nn = s.dropna().astype(float)
        skew = float(nn.skew()) if len(nn) > 2 else 0.0
        if role == "binary":
            return "mode", f"Impute '{name}' with its most frequent value (two-valued column)."
        method = "median" if abs(skew) > 1 else "mean"
        extra = " Add a missing-indicator column if missingness may be informative." if frac > 0.10 else ""
        return method, (f"Impute '{name}' with the {method} ({'skewed' if method == 'median' else 'roughly symmetric'} distribution, skewness {skew:.2f}).{extra}")
    if role == "datetime":
        return "leave", f"Leave missing dates in '{name}' as missing; rows without a date are excluded from time analyses."
    if frac < 0.05:
        return "mode", f"Impute '{name}' with its most frequent category (only {frac:.1%} missing)."
    return "unknown", f"Fill missing '{name}' with an explicit 'Unknown' category ({frac:.1%} missing) rather than guessing a value."


def assess_quality(df: pd.DataFrame, profile: Dict[str, Any], semantics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    n, p = df.shape
    cells = max(1, n * p)
    roles = {c["name"]: c["role"] for c in profile["columns"]}
    issues: List[Dict[str, Any]] = []
    column_rows: List[Dict[str, Any]] = []

    # ---- completeness
    miss = df.isna()
    missing_cells = int(miss.values.sum())
    for c in df.columns:
        frac = float(miss[c].mean())
        if frac > 0:
            action, rec = _missing_treatment(df[c], c, frac, roles.get(c, ""))
            issues.append({"dimension": "completeness", "severity": "high" if frac > 0.2 else "medium" if frac > 0.05 else "low", "columns": [c],
                           "affected": int(miss[c].sum()), "affected_%": frac * 100, "issue": f"{frac:.1%} of '{c}' values are missing ({int(miss[c].sum()):,} rows).",
                           "recommendation": rec, "treatment": action})

    # ---- uniqueness
    dup_rows = int(df.duplicated().sum())
    if dup_rows:
        issues.append({"dimension": "uniqueness", "severity": "high" if dup_rows / n > 0.02 else "medium", "columns": [], "affected": dup_rows,
                       "affected_%": dup_rows / n * 100, "issue": f"{dup_rows:,} exact duplicate rows ({dup_rows / n:.1%}).",
                       "recommendation": "Remove exact duplicates; they double-count records and leak between train and test splits.", "treatment": "drop_duplicates"})
    key_dups = 0
    for c in profile["columns"]:
        if c["role"] == "identifier" and c["unique_values"] < n - c["missing"] and c["unique_values"] > 0.9 * n:
            k = int(df[c["name"]].dropna().duplicated().sum())
            key_dups += k
            issues.append({"dimension": "uniqueness", "severity": "medium", "columns": [c["name"]], "affected": k, "affected_%": k / n * 100,
                           "issue": f"Identifier '{c['name']}' repeats on {k:,} rows although it is almost unique.",
                           "recommendation": f"Check whether repeated '{c['name']}' values are true duplicates or legitimate repeat records.", "treatment": None})

    # ---- validity
    invalid_cells = int(sum((df.attrs.get("coerced_invalid") or {}).values()))
    for col, k in (df.attrs.get("coerced_invalid") or {}).items():
        issues.append({"dimension": "validity", "severity": "medium", "columns": [col], "affected": int(k), "affected_%": k / n * 100,
                       "issue": f"{k:,} value(s) in '{col}' could not be parsed as {'numbers' if pd.api.types.is_numeric_dtype(df[col]) else 'dates'} and were set to missing.",
                       "recommendation": "Correct the source formatting (e.g. text such as 'n/a' or '12,3' in a numeric column).", "treatment": None})
    invalid_masks: Dict[str, pd.Series] = {}
    for c in df.columns:
        sem = semantics.get(c, {}).get("semantic", "")
        m, why = _invalid_mask(df, c, sem)
        k = int(m.sum())
        if k:
            invalid_masks[c] = m
            invalid_cells += k
            ex = df.loc[m, c].head(3).tolist()
            issues.append({"dimension": "validity", "severity": "high" if k / n > 0.02 else "medium", "columns": [c], "affected": k, "affected_%": k / n * 100,
                           "issue": f"{k:,} invalid value(s) in '{c}': {why} (e.g. {', '.join(map(str, ex))}).",
                           "recommendation": f"Treat invalid '{c}' values as missing and impute, or correct them at the source.", "treatment": "invalid_to_missing"})

    # ---- consistency
    inconsistent = 0
    variants_by_col: Dict[str, List[Dict[str, Any]]] = {}
    for c in df.columns:
        if roles.get(c) in ("categorical", "binary", "identifier", "text"):
            v = category_variants(df[c])
            if v:
                variants_by_col[c] = v
                k = sum(g["rows_affected"] for g in v)
                inconsistent += k
                ex = "; ".join(f"{g['canonical']} ← {', '.join(map(repr, g['variants'][:3]))}" for g in v[:3])
                issues.append({"dimension": "consistency", "severity": "medium" if k / n > 0.01 else "low", "columns": [c], "affected": k, "affected_%": k / n * 100,
                               "issue": f"'{c}' has {len(v)} label(s) written in several ways ({ex}).",
                               "recommendation": f"Standardise '{c}' labels to their most frequent spelling so groups are not split.", "treatment": "standardize_categories"})
    for f in derived_columns(df, profile):
        viol = int(round((1 - f["rows_matching"]) * n))
        if viol:
            inconsistent += viol
            issues.append({"dimension": "consistency", "severity": "low", "columns": [f["column"]] + f["inputs"], "affected": viol, "affected_%": viol / n * 100,
                           "issue": f"{f['formula']} holds on {f['rows_matching']:.1%} of rows; about {viol:,} rows break the rule.",
                           "recommendation": "Review the rows that break the business rule; they are often entry errors.", "treatment": None})
    for r in date_rules(df, profile):
        viol = int(round((1 - r["share_ordered"]) * df[[r["before"], r["after"]]].dropna().shape[0]))
        if viol:
            inconsistent += viol
            issues.append({"dimension": "consistency", "severity": "medium", "columns": [r["before"], r["after"]], "affected": viol, "affected_%": viol / n * 100,
                           "issue": f"'{r['after']}' is earlier than '{r['before']}' on {viol:,} rows.", "recommendation": "Correct or exclude rows with impossible date order.", "treatment": None})

    # ---- outliers
    outlier_cells = 0
    numeric_cols = [c["name"] for c in profile["columns"] if c["role"] == "numeric"]
    for c in numeric_cols:
        m = _robust_outliers(df[c])
        k = int(m.sum())
        if k:
            outlier_cells += k
            if k / n >= 0.005:
                issues.append({"dimension": "outliers", "severity": "medium" if k / n > 0.03 else "low", "columns": [c], "affected": k, "affected_%": k / n * 100,
                               "issue": f"{k:,} extreme value(s) in '{c}' (modified z-score > {ROBUST_Z} and beyond 3×IQR); range of flagged values {df.loc[m, c].min():,.4g} to {df.loc[m, c].max():,.4g}.",
                               "recommendation": f"Review extreme '{c}' values; keep genuine ones, and cap (winsorise) only if they are errors or distort averages.", "treatment": "cap_outliers"})

    # ---- column-level table
    for c in profile["columns"]:
        name = c["name"]
        column_rows.append({"column": name, "role": c["role"], "semantic": semantics.get(name, {}).get("semantic"), "missing_%": c["missing_pct"],
                            "invalid": int(invalid_masks.get(name, pd.Series(dtype=bool)).sum()) + int((df.attrs.get("coerced_invalid") or {}).get(name, 0)),
                            "label_variants": sum(g["rows_affected"] for g in variants_by_col.get(name, [])),
                            "outliers": int(_robust_outliers(df[name]).sum()) if name in numeric_cols else 0,
                            "unique": c["unique_values"]})
    for c in profile["columns"]:
        if c["role"] == "constant":
            issues.append({"dimension": "consistency", "severity": "low", "columns": [c["name"]], "affected": n, "affected_%": 100.0,
                           "issue": f"'{c['name']}' has a single value on every row and carries no information.",
                           "recommendation": f"Drop '{c['name']}' from analysis.", "treatment": "drop_constant_columns"})

    incomplete_rows = float(miss.any(axis=1).mean()) if n else 0.0
    rates = {"completeness": 0.5 * missing_cells / cells + 0.5 * incomplete_rows, "uniqueness": (dup_rows + key_dups) / max(1, n), "validity": invalid_cells / cells,
             "consistency": inconsistent / cells, "outliers": outlier_cells / max(1, n * max(1, len(numeric_cols)))}
    dims = {d: 100.0 * (1 - min(1.0, SENSITIVITY[d] * r)) for d, r in rates.items()}
    score = float(sum(WEIGHTS[d] * dims[d] for d in dims))
    sev_rank = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda i: (sev_rank[i["severity"]], -i["affected"]))
    for i, it in enumerate(issues, 1):
        it["id"] = i
    summary = (f"Data quality score {score:.1f}/100 (grade {_grade(score)}). {len(issues)} issue(s): "
               f"{sum(1 for i in issues if i['severity'] == 'high')} high, {sum(1 for i in issues if i['severity'] == 'medium')} medium, "
               f"{sum(1 for i in issues if i['severity'] == 'low')} low severity. Weakest dimension: {min(dims, key=dims.get)} ({min(dims.values()):.1f}).")
    return {"score": score, "grade": _grade(score), "dimensions": dims, "defect_rates": rates, "weights": WEIGHTS, "sensitivity": SENSITIVITY,
            "issues": issues, "columns": column_rows, "rows": n, "columns_count": p, "missing_cells": missing_cells, "duplicate_rows": dup_rows,
            "summary": summary, "_variants": variants_by_col, "_invalid": invalid_masks}


def clean_dataset(df: pd.DataFrame, profile: Dict[str, Any], semantics: Dict[str, Dict[str, Any]],
                  actions: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Apply the selected cleaning actions in a fixed, safe order. Returns (clean_df, log)."""
    actions = list(dict.fromkeys(actions or DEFAULT_ACTIONS))
    unknown = [a for a in actions if a not in ALL_ACTIONS]
    if unknown:
        raise ValueError(f"Unknown cleaning action(s): {unknown}. Valid: {ALL_ACTIONS}")
    q = assess_quality(df, profile, semantics)
    out = df.copy()
    log: List[Dict[str, Any]] = []
    roles = {c["name"]: c["role"] for c in profile["columns"]}

    if "standardize_categories" in actions:
        for c, groups in q["_variants"].items():
            mapping = {v: g["canonical"] for g in groups for v in g["variants"]}
            k = int(out[c].astype(object).isin(list(mapping)).sum())
            out[c] = out[c].map(lambda v: mapping.get(v, v) if isinstance(v, str) else v)
            log.append({"action": "standardize_categories", "column": c, "changed": k, "detail": f"{len(mapping)} variant label(s) mapped to canonical spellings"})
    if "invalid_to_missing" in actions:
        for c, m in q["_invalid"].items():
            m = m.reindex(out.index, fill_value=False)
            out.loc[m, c] = np.nan
            log.append({"action": "invalid_to_missing", "column": c, "changed": int(m.sum()), "detail": "invalid values set to missing before imputation"})
    if "drop_duplicates" in actions:
        before = len(out)
        out = out.drop_duplicates()
        if before - len(out):
            log.append({"action": "drop_duplicates", "column": None, "changed": before - len(out), "detail": "exact duplicate rows removed"})
    if "drop_constant_columns" in actions:
        const = [c for c in out.columns if out[c].nunique(dropna=True) <= 1]
        if const:
            out = out.drop(columns=const)
            log.append({"action": "drop_constant_columns", "column": ", ".join(const), "changed": len(const), "detail": "columns with a single value removed"})
    if "drop_sparse_columns" in actions:
        sparse = [c for c in out.columns if out[c].isna().mean() > SPARSE_COLUMN]
        if sparse:
            out = out.drop(columns=sparse)
            log.append({"action": "drop_sparse_columns", "column": ", ".join(sparse), "changed": len(sparse), "detail": f"columns with more than {SPARSE_COLUMN:.0%} missing removed"})
    if "drop_missing_rows" in actions:
        before = len(out)
        out = out.dropna()
        log.append({"action": "drop_missing_rows", "column": None, "changed": before - len(out), "detail": "rows with any missing value removed"})
    if "cap_outliers" in actions:
        for c in [c for c in out.columns if roles.get(c) == "numeric"]:
            x = out[c].astype(float)
            lo, hi = x.quantile(0.01), x.quantile(0.99)
            k = int(((x < lo) | (x > hi)).sum())
            if k and _robust_outliers(x).any():
                out[c] = x.clip(lo, hi)
                log.append({"action": "cap_outliers", "column": c, "changed": k, "detail": f"values capped to the 1st-99th percentile range [{lo:,.4g}, {hi:,.4g}]"})
    if "impute_missing" in actions:
        for c in out.columns:
            k = int(out[c].isna().sum())
            if not k:
                continue
            method, _ = _missing_treatment(out[c], c, k / len(out), roles.get(c, ""))
            if method in ("leave", "drop_column"):
                continue
            s = out[c]
            if method == "median":
                fill = float(s.median())
            elif method == "mean":
                fill = float(s.mean())
            elif method == "mode":
                fill = s.mode().iloc[0]
            else:
                fill = "Unknown"
            if pd.api.types.is_integer_dtype(s.dropna()) and isinstance(fill, float) and method in ("median", "mean"):
                fill = round(fill)
            out[c] = s.fillna(fill)
            log.append({"action": "impute_missing", "column": c, "changed": k, "detail": f"{method} imputation (value {fill if isinstance(fill, str) else f'{fill:,.4g}'})"})
    out = out.reset_index(drop=True)
    out.attrs = {k: v for k, v in df.attrs.items() if k != "coerced_invalid"}
    out.attrs["coerced_invalid"] = {}
    return out, log


def compare_quality(before: Dict[str, Any], after: Dict[str, Any], df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict[str, Any]:
    rows = []
    for d in WEIGHTS:
        rows.append({"dimension": d, "before": before["dimensions"][d], "after": after["dimensions"][d], "change": after["dimensions"][d] - before["dimensions"][d]})
    stats = []
    for c in df_after.columns:
        if c in df_before.columns and pd.api.types.is_numeric_dtype(df_before[c]) and pd.api.types.is_numeric_dtype(df_after[c]):
            b, a = df_before[c].astype(float), df_after[c].astype(float)
            stats.append({"column": c, "mean_before": float(b.mean()), "mean_after": float(a.mean()), "missing_before": int(b.isna().sum()),
                          "missing_after": int(a.isna().sum()), "std_before": float(b.std()), "std_after": float(a.std())})
    return {"score_before": before["score"], "score_after": after["score"], "score_change": after["score"] - before["score"],
            "rows_before": len(df_before), "rows_after": len(df_after), "columns_before": df_before.shape[1], "columns_after": df_after.shape[1],
            "missing_before": before["missing_cells"], "missing_after": after["missing_cells"],
            "duplicates_before": before["duplicate_rows"], "duplicates_after": after["duplicate_rows"],
            "issues_before": len(before["issues"]), "issues_after": len(after["issues"]), "dimensions": rows, "numeric_columns": stats[:30]}


def public(q: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in q.items() if not k.startswith("_")}
