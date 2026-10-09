"""forecast: time-aware forecasting with rolling-origin backtesting (observations are never shuffled).

Candidates are scored on the last blocks of the real series (expanding window); the winner is refit on all history.
Naive baselines are always included so the report says honestly whether a model adds skill.
"""
from __future__ import annotations
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from context import ExecContext
from diagnostics_engine import FREQ_NAME, PERIOD_ALIAS, aggregate_series, native_frequency, period_label
from errors import UDIError
from result_formatter import fmt_num, make_result, records
from validator import ValidatedPlan

MIN_PERIODS = 24
MIN_TRAIN = 12
MAX_MISSING_FRACTION = 0.2
SEASON = {"H": 24, "D": 7, "W": 52, "M": 12, "Q": 4, "Y": 1}
DEFAULT_HORIZON = {"H": 24, "D": 14, "W": 8, "M": 6, "Q": 4, "Y": 2}


# ----------------------------------------------------------------------------- candidate forecasters
def _seasonal_m(freq: str, n: int) -> int:
    m = SEASON[freq]
    return m if (m >= 2 and n >= 2 * m) else 0


def _naive(y, h, m):
    return np.repeat(y[-1], h)


def _seasonal_naive(y, h, m):
    return np.array([y[len(y) - m + (i % m)] for i in range(h)])


def _moving_average(y, h, m):
    k = m if m >= 2 else 3
    return np.repeat(float(np.mean(y[-k:])), h)


def _linear_trend(y, h, m):
    n = len(y)
    t = np.arange(n).reshape(-1, 1)
    lr = LinearRegression().fit(t, y)
    resid = y - lr.predict(t)
    seas = np.zeros(m) if m else None
    if m:
        for k in range(m):
            seas[k] = resid[k::m].mean()
    tf = np.arange(n, n + h)
    out = lr.predict(tf.reshape(-1, 1))
    if m:
        out = out + np.array([seas[t_ % m] for t_ in tf])
    return out


def _lags(n: int, m: int) -> List[int]:
    lags = [1, 2, 3]
    if m > 3 and n - m >= MIN_TRAIN:
        lags.append(m)
    return sorted(set(lags))


def _make_ar(model_factory: Callable[[], Any]):
    def fn(y, h, m):
        n = len(y)
        lags = _lags(n, m)
        L = max(lags)
        X = np.array([[y[t - l] for l in lags] + [t] for t in range(L, n)], dtype=float)
        Y = y[L:]
        mdl = model_factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl.fit(X, Y)
        hist = list(y)
        out = []
        for _ in range(h):
            t = len(hist)
            x = np.array([[hist[t - l] for l in lags] + [t]], dtype=float)
            v = float(mdl.predict(x)[0])
            hist.append(v)
            out.append(v)
        return np.array(out)
    return fn


def _candidates(m: int, n: int) -> Dict[str, Callable]:
    c: Dict[str, Callable] = {"Naive (last value)": _naive}
    if m:
        c["Seasonal naive"] = _seasonal_naive
    c["Moving average"] = _moving_average
    c["Linear trend" + (" + seasonal profile" if m else "")] = _linear_trend
    c["Autoregressive Ridge (lags + trend)"] = _make_ar(lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)))
    c["Random Forest (lags)"] = _make_ar(lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=42))
    c["Gradient Boosting (lags)"] = _make_ar(lambda: GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.06, random_state=42))
    return c


# ----------------------------------------------------------------------------- engine
def plan_backtest(n_total: int, horizon: int) -> Tuple[int, List[int]]:
    """Expanding-window backtest: consecutive, non-overlapping test blocks taken from the END of the series.
    Each fold trains on y[:cut] and is scored on y[cut:cut+h]; training data always precedes its test block."""
    h_bt = int(min(horizon, max(3, n_total // 8)))
    folds = 3
    while folds > 1 and n_total - folds * h_bt < MIN_TRAIN:
        folds -= 1
    if n_total - folds * h_bt < MIN_TRAIN:
        h_bt = max(1, n_total - MIN_TRAIN)
        folds = 1
    return h_bt, [n_total - h_bt * (folds - k) for k in range(folds)]


def _metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, Optional[float]]:
    err = actual - pred
    out = {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2))), "MAPE": None}
    if np.all(np.abs(actual) > 1e-9):
        out["MAPE"] = float(np.mean(np.abs(err / actual)) * 100)
    return out


def forecast(vp: ValidatedPlan, ctx: ExecContext) -> Dict[str, Any]:
    p, d, tc, tgt = vp.plan, vp.data, vp.time_column, vp.plan.target_column
    freq = p.frequency or native_frequency(d[tc])
    agg = p.aggregation or "mean"
    unit = FREQ_NAME[freq]
    s = aggregate_series(d, tc, tgt, agg, freq)
    have = int(s.notna().sum())
    gaps = int(s.isna().sum())
    if have < MIN_PERIODS:
        finer = {"Y": "Q", "Q": "M", "M": "W", "W": "D"}.get(freq)
        hint = ""
        if finer:
            alt = aggregate_series(d, tc, tgt, agg, finer)
            if alt.notna().sum() >= MIN_PERIODS:
                hint = f" The data would support a forecast at {FREQ_NAME[finer]}ly resolution ({int(alt.notna().sum())} periods)."
        raise UDIError("INSUFFICIENT_DATA", f"Only {have} {unit}(s) of history are available for '{tgt}'; forecasting needs at least {MIN_PERIODS} distinct periods.{hint}",
                       details={"periods": have, "frequency": freq, "required": MIN_PERIODS})
    if len(s) and gaps / len(s) > MAX_MISSING_FRACTION:
        raise UDIError("INSUFFICIENT_DATA", f"{gaps} of {len(s)} {unit}s have no data (over {MAX_MISSING_FRACTION:.0%}); the series is too sparse to forecast at {unit}ly resolution.")
    n_total = len(s)
    horizon = p.forecast_horizon or DEFAULT_HORIZON[freq]
    if horizon > n_total // 3:
        raise UDIError("INSUFFICIENT_DATA", f"A {horizon}-{unit} horizon is too long for {n_total} {unit}s of history; at most {n_total // 3} {unit}s can be forecast reliably (one third of the history).",
                       details={"max_horizon": n_total // 3})
    y = s.interpolate(limit_direction="both").to_numpy(dtype=float)
    m = _seasonal_m(freq, n_total)
    cands = _candidates(m, n_total)

    # ---- rolling-origin backtest on the last blocks
    h_bt, cuts = plan_backtest(n_total, horizon)
    folds = len(cuts)
    scores: List[Dict[str, Any]] = []
    errors_pool: Dict[str, List[np.ndarray]] = {}
    for name, fn in cands.items():
        fold_m, errs, failed = [], [], None
        for cut in cuts:
            try:
                pred = fn(y[:cut], h_bt, _seasonal_m(freq, cut))
                actual = y[cut:cut + h_bt]
                fold_m.append(_metrics(actual, pred))
                errs.append(actual - pred)
            except Exception as exc:        # a candidate that cannot fit this fold is reported, not hidden
                failed = str(exc)[:120]
                break
        if failed or not fold_m:
            scores.append({"model": name, "error": failed or "no folds"})
            continue
        mape = [f["MAPE"] for f in fold_m if f["MAPE"] is not None]
        scores.append({"model": name, "RMSE": float(np.mean([f["RMSE"] for f in fold_m])), "MAE": float(np.mean([f["MAE"] for f in fold_m])),
                       "MAPE": float(np.mean(mape)) if len(mape) == len(fold_m) else None})
        errors_pool[name] = errs
    ok = sorted([r for r in scores if "error" not in r], key=lambda r: r["RMSE"])
    if not ok:
        raise UDIError("MODEL_TRAINING_ERROR", "No forecasting model could be fitted: " + "; ".join(r.get("error", "") for r in scores)[:300])
    best = ok[0]
    for r in scores:
        r["selected"] = r.get("model") == best["model"]
    scores = sorted(scores, key=lambda r: r.get("RMSE", float("inf")))
    naive = next(r for r in ok if r["model"] == "Naive (last value)") if any(r["model"] == "Naive (last value)" for r in ok) else None
    best_baseline = min((r for r in ok if r["model"] in ("Naive (last value)", "Seasonal naive", "Moving average")), key=lambda r: r["RMSE"], default=None)

    # ---- final fit on all history
    fn = cands[best["model"]]
    fc = np.asarray(fn(y, horizon, m), dtype=float)
    pooled = np.concatenate(errors_pool[best["model"]]) if errors_pool.get(best["model"]) else np.array([best["RMSE"]])
    sigma = float(np.sqrt(np.mean(pooled ** 2)))
    last_p = s.index[-1]
    future_idx = pd.period_range(last_p + 1, periods=horizon, freq=PERIOD_ALIAS[freq])
    f_rows = [{"period": period_label(i, freq), "forecast": float(v), "lower_80": float(v - 1.2816 * sigma), "upper_80": float(v + 1.2816 * sigma)}
              for i, v in zip(future_idx, fc)]
    hist_keep = min(n_total, max(60, horizon * 4))
    hist_rows = [{"period": period_label(i, freq), "actual": None if pd.isna(v) else float(v)} for i, v in zip(s.index[-hist_keep:], s.to_numpy()[-hist_keep:])]
    joined = hist_rows + [{"period": r["period"], "forecast": r["forecast"], "lower": r["lower_80"], "upper": r["upper_80"]} for r in f_rows]
    if hist_rows and hist_rows[-1]["actual"] is not None:      # connect the two lines
        joined[len(hist_rows) - 1].update(forecast=hist_rows[-1]["actual"], lower=hist_rows[-1]["actual"], upper=hist_rows[-1]["actual"])
    skill = None
    if best_baseline and best_baseline["RMSE"] > 0:
        skill = (1 - best["RMSE"] / best_baseline["RMSE"]) * 100
    label_measure = f"{agg} of {tgt}" if agg != "mean" else f"average {tgt}"
    summary = (f"Forecast of {label_measure} per {unit} for the next {horizon} {unit}(s): from {fmt_num(fc[0])} ({f_rows[0]['period']}) to {fmt_num(fc[-1])} ({f_rows[-1]['period']}), "
               f"averaging {fmt_num(float(np.mean(fc)))}. Selected model: {best['model']} (rolling backtest RMSE {fmt_num(best['RMSE'])}, MAE {fmt_num(best['MAE'])}"
               + (f", MAPE {best['MAPE']:.1f}%" if best["MAPE"] is not None else "") + ").")
    lim = ["Forecasts assume historical patterns continue; they cannot anticipate events, policy changes or shocks outside the data.",
           f"The 80% range is approximate: it uses the size of errors seen in the backtest ({folds} fold(s) of {h_bt} {unit}(s)), not a formal probability model.",
           "Observations were never shuffled: every backtest forecast uses only data from before its test block."]
    if skill is not None:
        if skill <= 0:
            lim.append(f"The selected model does not beat the best simple baseline ({best_baseline['model']}, RMSE {fmt_num(best_baseline['RMSE'])}); the series may be hard to predict beyond its recent level.")
        else:
            summary += f" That is {skill:.0f}% lower error than the best naive baseline ({best_baseline['model']})."
    if gaps:
        lim.append(f"{gaps} {unit}(s) without data were filled by linear interpolation.")
    assumptions = list(vp.assumptions)
    if not p.frequency:
        assumptions.append(f"The series was forecast at its natural {unit}ly resolution.")
    metrics = {"Selected model": best["model"], "Backtest RMSE": best["RMSE"], "Backtest MAE": best["MAE"], "Horizon": f"{horizon} {unit}(s)",
               "History": f"{n_total} {unit}s", "Forecast average": float(np.mean(fc))}
    if best["MAPE"] is not None:
        metrics["Backtest MAPE %"] = best["MAPE"]
    return make_result("forecast", summary, title=f"{tgt} forecast", metrics=metrics, table=f_rows, table_title=f"Forecast for the next {horizon} {unit}(s)",
                       tables={"candidate_models": scores},
                       charts=[{"type": "forecast", "title": f"{tgt}: history and {horizon}-{unit} forecast", "x": "period", "data": joined}],
                       evaluation={"task": "forecast", "selected_model": best["model"], "selection_metric": "rolling-origin backtest RMSE (lower is better)",
                                   "backtest": {"folds": folds, "test_block_periods": h_bt, "min_train_periods": MIN_TRAIN}, "metrics": {k: best[k] for k in ("MAE", "RMSE", "MAPE")},
                                   "baseline": best_baseline, "skill_vs_baseline_%": skill, "candidates": scores, "seasonal_period": m},
                       forecast={"target": tgt, "frequency": freq, "aggregation": agg, "horizon": horizon, "history_periods": n_total, "periods": f_rows,
                                 "interval": "approximate 80% range from backtest errors"},
                       limitations=lim, assumptions=assumptions)
