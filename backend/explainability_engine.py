"""Explainable AI (Phase 2): Shapley-value (SHAP) explanations for any fitted UDI model.

UDI's models are scikit-learn pipelines over RAW columns (imputation + one-hot encoding happen inside the pipeline),
so explanations are computed on the original business columns, not on one-hot dummies.

Estimator: Monte-Carlo permutation sampling of Shapley values (Strumbelj & Kononenko 2014), the same quantity SHAP's
model-agnostic explainers estimate. For an instance x, background row b and a random feature order pi, features are
switched from b to x one at a time; the change in model output when feature j is switched is its marginal
contribution. Averaging over K antithetic permutations (pi and reversed pi) and background rows:

    phi_j(x) = (1/K) * sum_k [ f(z_{pi_k, <=j}) - f(z_{pi_k, <j}) ]
    efficiency:  f(x) = phi_0 + sum_j phi_j(x),   phi_0 = mean_k f(b_k)

The additivity gap |f(x) - phi_0 - sum_j phi_j| is reported for every explanation (it is ~0 by construction).
For classifiers f is the predicted probability of the explained class.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

GLOBAL_ROWS = 60
GLOBAL_PERMS = 10          # antithetic pairs -> 20 orderings per row
LOCAL_PERMS = 40
MAX_BATCH_ROWS = 60000


def _restore_types(frame: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    for c in ref.columns:
        if pd.api.types.is_numeric_dtype(ref[c]):
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
        else:
            frame[c] = frame[c].astype(object)
    return frame


def shapley_values(predict: Callable[[pd.DataFrame], np.ndarray], X: pd.DataFrame, background: pd.DataFrame,
                   n_pairs: int = GLOBAL_PERMS, seed: int = 42) -> Dict[str, Any]:
    """phi[i, j] for every row of X. predict(frame) -> 1-D array of model outputs.

    Ordering k is paired with background row b_k (the same K rows for every instance), so phi_0 = mean_k f(b_k) is
    shared by all explanations and efficiency f(x) = phi_0 + sum_j phi_j holds exactly for every row."""
    rng = np.random.default_rng(seed)
    cols = list(X.columns)
    p = len(cols)
    Xv = X[cols].to_numpy(dtype=object)
    perms = []
    for _ in range(n_pairs):
        pi = rng.permutation(p)
        perms += [pi, pi[::-1]]
    K = len(perms)
    bg = background.sample(K, replace=len(background) < K, random_state=seed)[cols]
    Bv = bg.to_numpy(dtype=object)
    phi = np.zeros((len(X), p))
    tri = np.tril(np.ones((p + 1, p), dtype=bool), -1)       # row r switches the first r features of the order
    masks = []
    for pi in perms:
        mask = np.zeros((p + 1, p), dtype=bool)
        mask[:, pi] = tri                                      # column pi[t] is switched in rows t+1..p
        masks.append(mask)
    per_batch = max(1, MAX_BATCH_ROWS // (K * (p + 1)))
    for start in range(0, len(X), per_batch):
        idx = list(range(start, min(len(X), start + per_batch)))
        blocks = [np.where(masks[k], Xv[i][None, :], Bv[k][None, :]) for i in idx for k in range(K)]
        frame = _restore_types(pd.DataFrame(np.vstack(blocks), columns=cols), X)
        out = np.asarray(predict(frame), dtype=float).reshape(len(idx), K, p + 1)
        for a, i in enumerate(idx):
            for k, pi in enumerate(perms):
                phi[i, pi] += np.diff(out[a, k]) / K
    base = float(np.mean(predict(_restore_types(bg.reset_index(drop=True).copy(), X))))
    fx = np.asarray(predict(_restore_types(X[cols].copy(), X)), dtype=float)
    gap = float(np.mean(np.abs(fx - (phi.sum(axis=1) + base)))) if len(X) else 0.0
    return {"phi": phi, "base": base, "fx": fx, "columns": cols, "additivity_gap": gap, "orderings": K, "background": bg}


def _direction(feature: pd.Series, phi: np.ndarray) -> str:
    if pd.api.types.is_numeric_dtype(feature):
        x = feature.astype(float).to_numpy()
        ok = ~np.isnan(x)
        if ok.sum() < 5 or np.std(x[ok]) == 0 or np.std(phi[ok]) == 0:
            return "no clear direction"
        r = float(np.corrcoef(x[ok], phi[ok])[0, 1])
        if abs(r) < 0.2:
            return "mixed / non-monotone effect"
        return "higher values push the prediction up" if r > 0 else "higher values push the prediction down"
    df = pd.DataFrame({"v": feature.astype(str).to_numpy(), "phi": phi})
    g = df.groupby("v")["phi"].mean().sort_values()
    if len(g) < 2:
        return "no clear direction"
    return f"'{g.index[-1]}' pushes the prediction up most, '{g.index[0]}' pushes it down most"


def global_explanation(predict, X_eval: pd.DataFrame, X_background: pd.DataFrame, seed: int = 42,
                       target_label: str = "prediction") -> Dict[str, Any]:
    n = min(GLOBAL_ROWS, len(X_eval))
    Xs = X_eval.sample(n, random_state=seed) if len(X_eval) > n else X_eval
    sv = shapley_values(predict, Xs, X_background, GLOBAL_PERMS, seed)
    mean_abs = np.abs(sv["phi"]).mean(axis=0)
    total = mean_abs.sum() or 1.0
    items = []
    for j, c in enumerate(sv["columns"]):
        items.append({"feature": c, "mean_abs_shap": float(mean_abs[j]), "share_%": float(mean_abs[j] / total * 100),
                      "direction": _direction(Xs[c], sv["phi"][:, j])})
    items.sort(key=lambda i: -i["mean_abs_shap"])
    return {"method": f"SHAP values (Monte-Carlo permutation Shapley estimator, {sv['orderings']} feature orderings per row, {n} held-out rows, {sv['orderings']} background rows)",
            "explained_output": target_label, "base_value": sv["base"], "items": items, "additivity_gap": sv["additivity_gap"], "rows_explained": n,
            "note": "Mean |SHAP| measures how much each column moves this model's output on average; it is not a causal effect."}


def local_explanation(predict, x_row: pd.DataFrame, X_background: pd.DataFrame, seed: int = 42, top: int = 10,
                      target_label: str = "prediction") -> Dict[str, Any]:
    sv = shapley_values(predict, x_row, X_background, LOCAL_PERMS, seed)
    phi = sv["phi"][0]
    order = np.argsort(-np.abs(phi))
    contrib = []
    for j in order:
        v = x_row.iloc[0][sv["columns"][j]]
        contrib.append({"feature": sv["columns"][j], "value": None if (isinstance(v, float) and np.isnan(v)) else v, "shap": float(phi[j])})
    shown, rest = contrib[:top], contrib[top:]
    steps = [{"step": "Average prediction (base)", "value": sv["base"], "kind": "base"}]
    for c in shown:
        vtxt = c["value"] if isinstance(c["value"], str) else (f"{c['value']:,.4g}" if c["value"] is not None else "missing")
        steps.append({"step": f"{c['feature']} = {vtxt}", "value": c["shap"], "kind": "contribution"})
    if rest:
        steps.append({"step": f"{len(rest)} other column(s)", "value": float(sum(c["shap"] for c in rest)), "kind": "contribution"})
    steps.append({"step": "This prediction", "value": float(sv["fx"][0]), "kind": "total"})
    up = [c for c in shown if c["shap"] > 0][:3]
    down = [c for c in shown if c["shap"] < 0][:3]
    text = []
    if up:
        text.append("Pushing the " + target_label + " up: " + ", ".join(f"{c['feature']} ({c['shap']:+.4g})" for c in up) + ".")
    if down:
        text.append("Pushing it down: " + ", ".join(f"{c['feature']} ({c['shap']:+.4g})" for c in down) + ".")
    return {"method": f"SHAP values for this record ({sv['orderings']} feature orderings paired with as many background rows)", "explained_output": target_label,
            "base_value": sv["base"], "prediction": float(sv["fx"][0]), "contributions": contrib, "waterfall": steps,
            "additivity_gap": sv["additivity_gap"], "summary": " ".join(text)}


def waterfall_chart(local: Dict[str, Any], title: str) -> Dict[str, Any]:
    """Recharts-friendly waterfall: invisible base + visible delta bars."""
    rows, running = [], 0.0
    for s in local["waterfall"]:
        if s["kind"] == "base":
            running = s["value"]
            rows.append({"label": s["step"], "start": 0.0, "delta": s["value"], "kind": "base", "value": s["value"]})
        elif s["kind"] == "total":
            rows.append({"label": s["step"], "start": 0.0, "delta": s["value"], "kind": "total", "value": s["value"]})
        else:
            lo = running if s["value"] >= 0 else running + s["value"]
            rows.append({"label": s["step"], "start": lo, "delta": abs(s["value"]), "kind": "up" if s["value"] >= 0 else "down", "value": s["value"]})
            running += s["value"]
    return {"type": "waterfall", "title": title, "x": "label", "data": rows}


def model_predictor(pipeline, task: str, class_index: Optional[int] = None) -> Callable[[pd.DataFrame], np.ndarray]:
    if task == "classification":
        ci = 1 if class_index is None else class_index
        return lambda frame: pipeline.predict_proba(frame)[:, ci]
    return lambda frame: pipeline.predict(frame)
