"""Column relationship intelligence (Phase 2): how every column relates to every other column.

Produces, from the data alone:
  * a mixed-type association matrix (Pearson |r| / Spearman for numeric-numeric, correlation ratio eta for
    numeric-categorical, bias-corrected Cramer's V for categorical-categorical), all on a 0..1 strength scale
  * hierarchies / functional dependencies between categorical columns (City -> State -> Country)
  * derived numeric columns (Profit = Sales - Cost, Revenue = Units x Price, ...)
  * duplicate / redundant columns, candidate keys and date-ordering rules (Ship Date >= Order Date)
  * a data dictionary (one row per column, with its strongest partners)
Every statement is computed; nothing is inferred from names except the semantic labels in semantics.py.
"""
from __future__ import annotations
import itertools
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

SAMPLE_ROWS = 20000
MAX_MATRIX_COLS = 24
MAX_CATS = 60


def _sample(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    return df.sample(SAMPLE_ROWS, random_state=seed) if len(df) > SAMPLE_ROWS else df


def cramers_v(a: pd.Series, b: pd.Series) -> Optional[float]:
    """Bias-corrected Cramer's V (Bergsma 2013)."""
    ct = pd.crosstab(a.astype(str), b.astype(str))
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return None
    chi2 = stats.chi2_contingency(ct.values, correction=False)[0]
    n = ct.values.sum()
    phi2 = chi2 / n
    r, k = ct.shape
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / max(1, n - 1))
    rc, kc = r - (r - 1) ** 2 / max(1, n - 1), k - (k - 1) ** 2 / max(1, n - 1)
    den = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / den)) if den > 0 else None


def correlation_ratio(cat: pd.Series, num: pd.Series) -> Optional[float]:
    df = pd.DataFrame({"c": cat.astype(str), "y": num.astype(float)}).dropna()
    if df["c"].nunique() < 2 or len(df) < 5:
        return None
    mu = df["y"].mean()
    sst = float(((df["y"] - mu) ** 2).sum())
    if sst == 0:
        return None
    g = df.groupby("c")["y"]
    ssb = float((g.size() * (g.mean() - mu) ** 2).sum())
    return float(np.sqrt(max(0.0, min(1.0, ssb / sst))))


def _usable(df: pd.DataFrame, profile: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    nums, cats = [], []
    for c in profile["columns"]:
        name, role = c["name"], c["role"]
        if role in ("numeric",) or (role == "binary" and pd.api.types.is_numeric_dtype(df[name])):
            if df[name].nunique(dropna=True) > 1:
                nums.append(name)
        elif role in ("categorical", "binary") and 2 <= c["unique_values"] <= MAX_CATS:
            cats.append(name)
    return nums, cats


def association_matrix(df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:
    d = _sample(df)
    nums, cats = _usable(d, profile)
    cols = (nums + cats)[:MAX_MATRIX_COLS]
    kind = {c: ("numeric" if c in nums else "categorical") for c in cols}
    n = len(cols)
    M = np.full((n, n), np.nan)
    signed = np.full((n, n), np.nan)
    method = [[None] * n for _ in range(n)]
    pear = d[[c for c in cols if kind[c] == "numeric"]].corr(method="pearson") if nums else pd.DataFrame()
    spear = d[[c for c in cols if kind[c] == "numeric"]].corr(method="spearman") if nums else pd.DataFrame()
    for i in range(n):
        M[i, i] = 1.0
        signed[i, i] = 1.0
        method[i][i] = "self"
        for j in range(i + 1, n):
            a, b = cols[i], cols[j]
            if kind[a] == kind[b] == "numeric":
                r, rho = pear.loc[a, b], spear.loc[a, b]
                use_s = abs(rho) > abs(r) + 0.1          # clearly monotone-but-nonlinear relationship
                v, m = (rho, "spearman") if use_s else (r, "pearson")
                M[i, j] = M[j, i] = abs(v) if pd.notna(v) else np.nan
                signed[i, j] = signed[j, i] = v
                method[i][j] = method[j][i] = m
            elif kind[a] == kind[b] == "categorical":
                v = cramers_v(d[a], d[b])
                M[i, j] = M[j, i] = v if v is not None else np.nan
                method[i][j] = method[j][i] = "cramers_v"
            else:
                c_, x_ = (a, b) if kind[a] == "categorical" else (b, a)
                v = correlation_ratio(d[c_], d[x_])
                M[i, j] = M[j, i] = v if v is not None else np.nan
                method[i][j] = method[j][i] = "eta"
    pairs = []
    for i, j in itertools.combinations(range(n), 2):
        if np.isnan(M[i, j]):
            continue
        pairs.append({"column_1": cols[i], "column_2": cols[j], "strength": float(M[i, j]),
                      "signed": None if np.isnan(signed[i, j]) else float(signed[i, j]), "measure": method[i][j],
                      "label": strength_label(M[i, j])})
    pairs.sort(key=lambda p: p["strength"], reverse=True)
    return {"columns": cols, "kinds": kind, "matrix": [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in M],
            "signed": [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in signed], "pairs": pairs,
            "rows_used": int(len(d))}


def strength_label(v: float) -> str:
    a = abs(v)
    return "negligible" if a < 0.1 else "weak" if a < 0.3 else "moderate" if a < 0.5 else "strong" if a < 0.7 else "very strong"


def hierarchies(df: pd.DataFrame, profile: Dict[str, Any], min_purity: float = 0.98) -> List[Dict[str, Any]]:
    """A -> B when (almost) every value of A maps to a single value of B and A is finer than B."""
    roles = {c["name"]: c for c in profile["columns"]}
    cats = [c["name"] for c in profile["columns"] if c["role"] in ("categorical", "binary") and 2 <= c["unique_values"] <= 5000]
    out = []
    d = df[cats].dropna(how="all") if cats else df
    for a, b in itertools.permutations(cats, 2):
        ua, ub = roles[a]["unique_values"], roles[b]["unique_values"]
        if ua <= ub or ub < 2:
            continue
        pair = d[[a, b]].dropna()
        if len(pair) < 20:
            continue
        g = pair.groupby(a)[b]
        per, size = g.nunique(), g.size()
        evidence = per[size >= 2]                      # a value seen once trivially maps to one parent: not evidence
        if len(evidence) < max(5, 0.3 * len(per)):
            continue
        purity = float((evidence == 1).mean())
        if purity >= min_purity:
            out.append({"parent": b, "child": a, "child_values": int(ua), "parent_values": int(ub), "purity": purity,
                        "statement": f"Each '{a}' belongs to one '{b}' ({purity:.0%} of {ua:,} values); {a} is a finer level of {b}."})
    # keep only the direct links (drop A->C when A->B and B->C exist)
    links = {(h["child"], h["parent"]) for h in out}
    direct = [h for h in out if not any((h["child"], m) in links and (m, h["parent"]) in links for m in cats if m not in (h["child"], h["parent"]))]
    return sorted(direct, key=lambda h: (-h["purity"], h["parent"]))


def derived_columns(df: pd.DataFrame, profile: Dict[str, Any], tol: float = 0.01, min_share: float = 0.97) -> List[Dict[str, Any]]:
    """Detect target = a op b for op in (+, -, x, /) on (almost) all rows."""
    nums = [c["name"] for c in profile["columns"] if c["role"] == "numeric"][:16]
    d = _sample(df[nums].dropna(), 42) if nums else df
    if len(d) < 20 or len(nums) < 3:
        return []
    found = []
    arr = {c: d[c].to_numpy(dtype=float) for c in nums}
    ops = {"+": lambda x, y: x + y, "-": lambda x, y: x - y, "×": lambda x, y: x * y, "/": lambda x, y: np.divide(x, y, out=np.full_like(x, np.nan), where=y != 0)}
    for tgt in nums:
        t = arr[tgt]
        scale = np.nanmax(np.abs(t)) or 1.0
        for a, b in itertools.permutations([c for c in nums if c != tgt], 2):
            for sym, fn in ops.items():
                if sym in ("+", "×") and a > b:          # commutative: test each pair once
                    continue
                with np.errstate(all="ignore"):
                    v = fn(arr[a], arr[b])
                ok = np.abs(v - t) <= np.maximum(tol * np.abs(t), 1e-6 * scale)
                share = float(np.nanmean(ok))
                if share >= min_share:
                    found.append({"column": tgt, "formula": f"{tgt} = {a} {sym} {b}", "inputs": [a, b], "operator": sym, "rows_matching": share,
                                  "statement": f"'{tgt}' equals {a} {sym} {b} on {share:.1%} of rows (within 1%): it is a derived column."})
    # one formula per target column (the best)
    best: Dict[str, Dict[str, Any]] = {}
    for f in found:
        if f["column"] not in best or f["rows_matching"] > best[f["column"]]["rows_matching"]:
            best[f["column"]] = f
    return list(best.values())


def duplicate_columns(df: pd.DataFrame, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    names = [c["name"] for c in profile["columns"] if c["role"] not in ("empty",)]
    d = _sample(df)
    for a, b in itertools.combinations(names, 2):
        sa, sb = d[a], d[b]
        if pd.api.types.is_numeric_dtype(sa) and pd.api.types.is_numeric_dtype(sb):
            pair = pd.concat([sa, sb], axis=1).dropna()
            if len(pair) > 10 and pair.iloc[:, 0].std() > 0 and pair.iloc[:, 1].std() > 0 and abs(pair.iloc[:, 0].corr(pair.iloc[:, 1])) > 0.999:
                out.append({"columns": [a, b], "statement": f"'{a}' and '{b}' are perfectly correlated (|r| > 0.999): one is redundant for modelling."})
        elif sa.astype(str).equals(sb.astype(str)):
            out.append({"columns": [a, b], "statement": f"'{a}' and '{b}' contain identical values."})
    return out


def date_rules(df: pd.DataFrame, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    dts = profile.get("datetime_columns") or []
    out = []
    for a, b in itertools.combinations(dts, 2):
        pair = df[[a, b]].dropna()
        if len(pair) < 10:
            continue
        gap = (pair[b] - pair[a]).dt.total_seconds() / 86400
        if (gap >= 0).mean() >= 0.98:
            first, second = a, b
        elif (gap <= 0).mean() >= 0.98:
            first, second, gap = b, a, -gap
        else:
            continue
        out.append({"before": first, "after": second, "share_ordered": float((gap >= 0).mean()), "median_gap_days": float(gap.median()),
                    "statement": f"'{second}' is on or after '{first}' in {(gap >= 0).mean():.1%} of rows (median gap {gap.median():.1f} days)."})
    return out


def candidate_keys(df: pd.DataFrame, profile: Dict[str, Any]) -> List[str]:
    """Unique, complete columns that look like keys (identifiers, integer codes or text), never continuous measures."""
    n = len(df)
    out = []
    for c in profile["columns"]:
        if c["unique_values"] != n or c["missing"] or c["role"] == "datetime":
            continue
        s = df[c["name"]]
        if pd.api.types.is_float_dtype(s) and not c.get("integer_valued"):
            continue
        out.append(c["name"])
    return out[:5]


def analyse_relationships(df: pd.DataFrame, profile: Dict[str, Any], semantics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    assoc = association_matrix(df, profile)
    hier = hierarchies(df, profile)
    derived = derived_columns(df, profile)
    dups = duplicate_columns(df, profile)
    drules = date_rules(df, profile)
    keys = candidate_keys(df, profile)
    partners: Dict[str, List[str]] = {}
    for p in assoc["pairs"]:
        if p["strength"] < 0.1:
            continue
        for x, y in ((p["column_1"], p["column_2"]), (p["column_2"], p["column_1"])):
            lst = partners.setdefault(x, [])
            if len(lst) < 3:
                sign = "" if p["signed"] is None else ("+" if p["signed"] > 0 else "−")
                lst.append(f"{y} ({p['measure'].replace('_', ' ')} {sign}{p['strength']:.2f})")
    dictionary = []
    for c in profile["columns"]:
        name = c["name"]
        sem = semantics.get(name, {})
        rel_notes = [h["statement"] for h in hier if name in (h["child"], h["parent"])][:2]
        rel_notes += [f["statement"] for f in derived if f["column"] == name]
        dictionary.append({"column": name, "role": c["role"], "semantic_type": sem.get("semantic"), "additive": sem.get("additive"),
                           "business_functions": ", ".join(sem.get("functions", [])), "unique_values": c["unique_values"],
                           "missing_%": c["missing_pct"], "description": sem.get("description"),
                           "strongest_relationships": "; ".join(partners.get(name, [])) or "none above 0.10",
                           "structural_notes": " ".join(rel_notes)})
    findings: List[str] = []
    strong = [p for p in assoc["pairs"] if p["strength"] >= 0.5]
    for p in strong[:6]:
        dirn = "" if p["signed"] is None else (" (positive)" if p["signed"] > 0 else " (negative)")
        findings.append(f"{p['column_1']} and {p['column_2']} are {p['label']}ly associated{dirn}: {p['measure'].replace('_', ' ')} = {p['strength']:.2f}.")
    findings += [h["statement"] for h in hier[:5]] + [f["statement"] for f in derived[:5]] + [r["statement"] for r in drules[:3]] + [x["statement"] for x in dups[:3]]
    if keys:
        findings.append(f"Candidate key column(s) (unique on every row): {', '.join(keys)}.")
    if not findings:
        findings.append("No association of at least moderate strength (0.5) and no structural dependency was found between columns.")
    return {"association": assoc, "hierarchies": hier, "derived": derived, "duplicates": dups, "date_rules": drules,
            "candidate_keys": keys, "dictionary": dictionary, "findings": findings}


def relationships_digest(rel: Dict[str, Any], limit: int = 8) -> Dict[str, Any]:
    """Compact form sent to Gemini so the planner knows how columns relate (schema-level, no rows)."""
    return {"strong_pairs": [f"{p['column_1']}~{p['column_2']}:{p['strength']:.2f}" for p in rel["association"]["pairs"][:limit] if p["strength"] >= 0.3],
            "hierarchies": [f"{h['child']}->{h['parent']}" for h in rel["hierarchies"][:limit]],
            "derived": [f["formula"] for f in rel["derived"][:limit]]}
