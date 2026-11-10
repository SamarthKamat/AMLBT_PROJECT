"""Runtime machine learning for uploaded datasets: predict_value, predict_class, run_scenario.

Nothing here is tied to a domain or a column name. For every request the engine
  1. decides which columns are usable features (preprocessing.select_features)
  2. trains the candidate models on the uploaded data
  3. selects one by cross-validation on the training split (mean +- std; one-standard-error rule prefers the
     simpler model when the difference is within fold noise), then tunes its hyperparameters (randomised search)
  4. reports honest metrics on a held-out test split the selection never saw, plus a learning curve, a
     time-ordered validation when the data has dates, leakage screening and SHAP explanations
  5. refits the winner on all labelled rows (used for predictions / scenarios)
Trained models are cached per session by (task, target, features, filters), so follow-up questions reuse them.
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import (KFold, RandomizedSearchCV, StratifiedKFold, cross_val_score, cross_validate,
                                     learning_curve, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:                                    # XGBoost is optional ("if available")
    from xgboost import XGBClassifier, XGBRegressor
    HAVE_XGB = True
except Exception:                       # pragma: no cover
    HAVE_XGB = False

from context import MODEL_CONFIG_VERSION, ExecContext
from explainability_engine import global_explanation, local_explanation, model_predictor, waterfall_chart
from errors import UDIError
from filtering import filters_signature
from preprocessing import build_preprocessor, near_duplicate_of_target, prepare_X, select_features, typical_row
from result_formatter import fmt_num, make_result, records
from validator import ValidatedPlan

SELECTION_ROWS = 20000      # model selection is run on at most this many rows
MAX_FIT_ROWS = 200000       # final refit cap
TEST_SIZE = 0.2
IMPORTANCE_ROWS = 1500
TREE_FAMILY = ("Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost")
CAUSAL_NOTE = "Model-based estimates describe statistical relationships in the uploaded data; they do not establish causal effects."


@dataclass
class TrainedModel:
    task: str
    target: str
    features: List[str]
    excluded: List[Dict[str, str]]
    final_pipeline: Any
    eval_pipeline: Any
    selected: str
    evaluation: Dict[str, Any]
    importance: List[Dict[str, Any]]
    holdout: pd.DataFrame
    X_reference: pd.DataFrame
    classes: Optional[List[str]] = None
    encoder: Optional[LabelEncoder] = None
    notes: List[str] = field(default_factory=list)
    train_ranges: Dict[str, Any] = field(default_factory=dict)
    shap: Optional[Dict[str, Any]] = None
    X_train: Optional[pd.DataFrame] = None
    explain_index: Optional[int] = None


# --------------------------------------------------------------------------- candidates
def _regressors(n: int) -> Dict[str, Any]:
    c = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.9, random_state=42),
    }
    if HAVE_XGB:
        c["XGBoost Regressor"] = XGBRegressor(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                                              objective="reg:squarederror", n_jobs=2, random_state=42, verbosity=0)
    return c


def _classifiers(n_classes: int) -> Dict[str, Any]:
    c = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42),
    }
    if HAVE_XGB:
        c["XGBoost Classifier"] = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.06, subsample=0.9, colsample_bytree=0.9,
                                                eval_metric="logloss" if n_classes == 2 else "mlogloss", n_jobs=2, random_state=42, verbosity=0)
    return c


# --------------------------------------------------------------------------- helpers
def _permutation_importance(pipe, Xte: pd.DataFrame, yte, scoring: str, seed: int) -> List[Dict[str, Any]]:
    if len(Xte) > IMPORTANCE_ROWS:
        idx = np.random.RandomState(seed).choice(len(Xte), IMPORTANCE_ROWS, replace=False)
        Xte, yte = Xte.iloc[idx], np.asarray(yte)[idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = permutation_importance(pipe, Xte, yte, scoring=scoring, n_repeats=3, random_state=seed, n_jobs=1)
    pos = np.clip(res.importances_mean, 0, None)
    total = pos.sum()
    items = [{"feature": f, "importance": float(m), "std": float(s), "share_%": float(max(m, 0) / total * 100) if total > 0 else 0.0}
             for f, m, s in zip(Xte.columns, res.importances_mean, res.importances_std)]
    return sorted(items, key=lambda i: i["importance"], reverse=True)


def _prep_training_frame(vp: ValidatedPlan, ctx: ExecContext, requested: Optional[List[str]], always: List[str]):
    p = vp.plan
    tgt = p.target_column
    d = vp.data.dropna(subset=[tgt])
    notes: List[str] = []
    dups = int(d.duplicated().sum())
    if dups:
        d = d.drop_duplicates()
        notes.append(f"{dups} exact duplicate row(s) were dropped before training so identical rows cannot leak between train and test.")
    feats, excluded = select_features(d, ctx.roles, tgt, requested, always)
    if requested:
        dropped = [e["column"] for e in excluded if e["column"] in requested]
        if dropped:
            notes.append("Requested feature(s) not used: " + "; ".join(f"{e['column']} ({e['reason']})" for e in excluded if e["column"] in requested))
    if not feats:
        raise UDIError("INSUFFICIENT_DATA", f"No usable feature columns remain to model '{tgt}' (" +
                       "; ".join(f"{e['column']}: {e['reason']}" for e in excluded[:6]) + ").")
    return d, feats, excluded, notes


def _cache_key(task: str, vp: ValidatedPlan, ctx: ExecContext, feats: List[str]):
    return (ctx.dataset_id, task, vp.plan.target_column, tuple(feats), filters_signature(vp.plan), MODEL_CONFIG_VERSION)


# --------------------------------------------------------------------------- AutoML helpers (Phase 2)
COMPLEXITY = {"Linear Regression": 1, "Logistic Regression": 1, "Decision Tree Regressor": 2, "Decision Tree Classifier": 2,
              "Random Forest Regressor": 3, "Random Forest Classifier": 3, "Gradient Boosting Regressor": 4,
              "Gradient Boosting Classifier": 4, "XGBoost Regressor": 5, "XGBoost Classifier": 5}
TUNE_MAX_ROWS = 3000
LARGE_TRAIN = 3000          # above this many training rows, selection uses 3-fold instead of 5-fold CV
CURVE_MAX_ROWS = 2000
TUNE_ITER = 5
TUNING_GRIDS = {
    "Random Forest": {"model__n_estimators": [100, 200], "model__max_depth": [None, 6, 12], "model__min_samples_leaf": [1, 2, 5]},
    "Gradient Boosting": {"model__n_estimators": [100, 200], "model__learning_rate": [0.03, 0.05, 0.1], "model__max_depth": [2, 3, 4]},
    "XGBoost": {"model__n_estimators": [150, 250], "model__learning_rate": [0.03, 0.06, 0.1], "model__max_depth": [3, 4, 6]},
    "Decision Tree": {"model__max_depth": [4, 6, 8, 12], "model__min_samples_leaf": [2, 5, 10]},
}


def _one_se_select(ok: List[Dict[str, Any]], key: str, std_key: str, folds: int, higher_better: bool) -> Tuple[Dict[str, Any], str]:
    """One-standard-error rule: among models whose CV score is within one standard error of the best,
    choose the simplest. Prevents picking a complex model because of fold-to-fold noise."""
    best = max(ok, key=lambda r: r[key]) if higher_better else min(ok, key=lambda r: r[key])
    se = best[std_key] / np.sqrt(max(1, folds))
    if higher_better:
        near = [r for r in ok if r[key] >= best[key] - se]
    else:
        near = [r for r in ok if r[key] <= best[key] + se]
    chosen = min(near, key=lambda r: (COMPLEXITY.get(r["model"], 9), -r[key] if higher_better else r[key]))
    if chosen["model"] == best["model"]:
        why = f"{best['model']} has the best cross-validated score ({best[key]:.4g} ± {best[std_key]:.3g})."
        others = sorted([r for r in near if r["model"] != best["model"]], key=lambda r: COMPLEXITY.get(r["model"], 9))
        if others:
            why += f" {len(others)} other model(s) were within one standard error, but none was simpler."
    else:
        why = (f"{best['model']} scored best ({best[key]:.4g}), but {chosen['model']} ({chosen[key]:.4g}) is within one standard error "
               f"({se:.3g}) and is simpler, so it was preferred (one-standard-error rule).")
    return chosen, why


def _paired_test(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    try:
        if len(a) >= 3 and np.std(a - b) > 0:
            return float(stats.ttest_rel(a, b).pvalue)
    except Exception:
        pass
    return None


def _cv3(cv, seed: int):
    """A cheaper 3-fold splitter of the same kind, for tuning and learning curves."""
    return StratifiedKFold(3, shuffle=True, random_state=seed) if isinstance(cv, StratifiedKFold) else KFold(3, shuffle=True, random_state=seed)


def _tune(pipe, name: str, Xtr, ytr, scoring: str, cv, seed: int) -> Optional[Dict[str, Any]]:
    """Randomised search over a small grid, scored with 3-fold CV. The tuned model is adopted only if it beats the
    default model's 3-fold score (compared on the same folds)."""
    family = next((f for f in TUNING_GRIDS if name.startswith(f)), None)
    if family is None or len(Xtr) > TUNE_MAX_ROWS or len(Xtr) < 60:
        return None
    cv3 = _cv3(cv, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default_score = float(np.mean(cross_val_score(clone(pipe), Xtr, ytr, cv=cv3, scoring=scoring)))
        search = RandomizedSearchCV(clone(pipe), TUNING_GRIDS[family], n_iter=TUNE_ITER, scoring=scoring, cv=cv3, random_state=seed, n_jobs=1, error_score=np.nan)
        search.fit(Xtr, ytr)
    return {"best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()}, "best_score": float(search.best_score_),
            "default_score": default_score, "estimator": search.best_estimator_, "candidates_tried": TUNE_ITER, "cv_folds": 3}


def _learning_curve(pipe, X, y, scoring: str, cv, seed: int, higher_better: bool) -> List[Dict[str, Any]]:
    if len(X) > CURVE_MAX_ROWS:
        idx = np.random.RandomState(seed).choice(len(X), CURVE_MAX_ROWS, replace=False)
        X, y = X.iloc[idx], np.asarray(y)[idx]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sizes, tr, va = learning_curve(clone(pipe), X, y, train_sizes=[0.25, 0.5, 0.75, 1.0], cv=_cv3(cv, seed), scoring=scoring, n_jobs=1, error_score=np.nan)
    except Exception:
        return []
    sign = 1 if higher_better else -1
    return [{"training_rows": int(s), "train_score": float(sign * np.nanmean(t)), "validation_score": float(sign * np.nanmean(v))} for s, t, v in zip(sizes, tr, va)]


def _temporal_check(pipe, d: pd.DataFrame, feats: List[str], target: str, time_col: Optional[str], task: str, enc=None) -> Optional[Dict[str, Any]]:
    """Train on the earliest 80% of rows (by time) and test on the latest 20%: detects drift that a random split hides."""
    if not time_col or time_col not in d.columns or len(d) < 100:
        return None
    w = d.dropna(subset=[time_col]).sort_values(time_col)
    if len(w) < 100:
        return None
    cut = int(len(w) * 0.8)
    tr, te = w.iloc[:cut], w.iloc[cut:]
    Xtr, Xte = prepare_X(tr, feats), prepare_X(te, feats)
    m = clone(pipe)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if task == "regression":
                m.fit(Xtr, tr[target].astype(float))
                pred = m.predict(Xte)
                yte = te[target].astype(float)
                out = {"R2": float(r2_score(yte, pred)), "RMSE": float(np.sqrt(mean_squared_error(yte, pred)))}
            else:
                ytr = enc.transform(tr[target].astype(str))
                yte = enc.transform(te[target].astype(str))
                if len(np.unique(ytr)) < 2:
                    return None
                m.fit(Xtr, ytr)
                pred = m.predict(Xte)
                out = {"Accuracy": float(accuracy_score(yte, pred)), "F1": float(f1_score(yte, pred, average="macro", zero_division=0))}
        except Exception:
            return None
    return {"time_column": time_col, "train_until": str(tr[time_col].max())[:10], "test_from": str(te[time_col].min())[:10],
            "train_rows": int(len(tr)), "test_rows": int(len(te)), "metrics": out}


def _single_feature_leaks(X: pd.DataFrame, y: np.ndarray, k: int) -> List[str]:
    """A single column that almost perfectly separates the classes usually encodes the label itself."""
    if k != 2:
        return []
    flags = []
    for c in X.columns:
        s = X[c]
        try:
            if pd.api.types.is_numeric_dtype(s):
                ok = s.notna().to_numpy()
                if ok.sum() > 20 and s[ok].nunique() > 1:
                    auc = roc_auc_score(y[ok], s[ok].astype(float))
                    if max(auc, 1 - auc) >= 0.99:
                        flags.append(c)
            elif s.nunique() <= 50:
                rate = pd.Series(y).groupby(s.astype(str).to_numpy()).mean()
                share_pure = float(pd.Series(s.astype(str).to_numpy()).map(((rate == 0) | (rate == 1)).to_dict()).mean())
                if share_pure >= 0.99 and s.nunique() > 1:
                    flags.append(c)
        except Exception:
            continue
    return flags


def _shap_global(pipe, task: str, Xte: pd.DataFrame, Xref: pd.DataFrame, seed: int, label: str, class_index: Optional[int] = None):
    try:
        return global_explanation(model_predictor(pipe, task, class_index), Xte, Xref, seed, label)
    except Exception as exc:          # explanation must never break a prediction
        return {"method": "SHAP unavailable", "items": [], "error": str(exc)[:200]}


# --------------------------------------------------------------------------- training
def train_regression(d: pd.DataFrame, feats: List[str], excluded, target: str, seed: int, notes: List[str],
                     time_col: Optional[str] = None) -> TrainedModel:
    if len(d) > MAX_FIT_ROWS:
        d = d.sample(MAX_FIT_ROWS, random_state=seed)
        notes.append(f"Training used a random sample of {MAX_FIT_ROWS:,} rows for speed.")
    y = d[target].astype(float)
    X = prepare_X(d, feats)
    leak = near_duplicate_of_target(X, y)
    if leak:
        notes.append(f"Possible leakage: column(s) {', '.join(leak)} are almost perfectly correlated with '{target}'. If the target is calculated from them, "
                     f"the model is learning that formula rather than discovering a relationship.")
    sel = d.sample(SELECTION_ROWS, random_state=seed).index if len(d) > SELECTION_ROWS else d.index
    Xs, ys = X.loc[sel], y.loc[sel]
    if len(d) > SELECTION_ROWS:
        notes.append(f"Model selection used a random sample of {SELECTION_ROWS:,} rows; the winner was refit on all {len(d):,}.")
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=TEST_SIZE, random_state=seed)
    folds = 5 if 100 <= len(Xtr) <= LARGE_TRAIN else 3
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    rows, fitted, fold_scores = [], {}, {}
    base_rmse = float(np.sqrt(mean_squared_error(yte, np.full(len(yte), ytr.mean()))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, est in _regressors(len(Xtr)).items():
            try:
                pipe = Pipeline([("prep", build_preprocessor(Xtr)), ("model", clone(est))])
                cvr = cross_validate(pipe, Xtr, ytr, cv=cv, scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}, error_score="raise")
                pipe.fit(Xtr, ytr)
                pred = pipe.predict(Xte)
                rmse_f = -cvr["test_rmse"]
                fold_scores[name] = rmse_f
                rows.append({"model": name, "cv_RMSE": float(rmse_f.mean()), "cv_RMSE_std": float(rmse_f.std(ddof=1)) if len(rmse_f) > 1 else 0.0,
                             "cv_MAE": float(-cvr["test_mae"].mean()), "cv_R2": float(cvr["test_r2"].mean()),
                             "test_MAE": float(mean_absolute_error(yte, pred)), "test_RMSE": float(np.sqrt(mean_squared_error(yte, pred))), "test_R2": float(r2_score(yte, pred))})
                fitted[name] = (pipe, pred)
            except Exception as exc:
                rows.append({"model": name, "error": str(exc)[:160]})
    ok = [r for r in rows if "error" not in r]
    if not ok:
        raise UDIError("MODEL_TRAINING_ERROR", "No regression model could be trained on this data: " + "; ".join(r.get("error", "") for r in rows)[:300])
    best, why = _one_se_select(ok, "cv_RMSE", "cv_RMSE_std", folds, higher_better=False)
    ranked = sorted(ok, key=lambda r: r["cv_RMSE"])
    runner = next((r for r in ranked if r["model"] != best["model"]), None)
    p_vs_runner = _paired_test(fold_scores[best["model"]], fold_scores[runner["model"]]) if runner else None
    eval_pipe, pred = fitted[best["model"]]
    tuning = _tune(eval_pipe, best["model"], Xtr, ytr, "neg_root_mean_squared_error", cv, seed)
    tuned_used = False
    if tuning and tuning["best_score"] > tuning["default_score"] and -tuning["best_score"] < -tuning["default_score"] * 0.995:
        eval_pipe = tuning["estimator"]
        pred = eval_pipe.predict(Xte)
        tuned_used = True
        why += f" Hyperparameter tuning (randomised search, {tuning['candidates_tried']} settings) lowered 3-fold CV RMSE from {-tuning['default_score']:.4g} to {-tuning['best_score']:.4g} and was adopted."
    elif tuning:
        why += " Hyperparameter tuning did not improve the cross-validated score, so default settings were kept."
    test_m = {"MAE": float(mean_absolute_error(yte, pred)), "RMSE": float(np.sqrt(mean_squared_error(yte, pred))), "R2": float(r2_score(yte, pred))}
    for r in rows:
        r["selected"] = r.get("model") == best["model"]
    rows = sorted(rows, key=lambda r: (r.get("cv_RMSE", float("inf"))))
    final = clone(eval_pipe)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final.fit(X, y)
    importance = _permutation_importance(eval_pipe, Xte, yte, "neg_root_mean_squared_error", seed)
    shap = _shap_global(eval_pipe, "regression", Xte, Xtr, seed, f"predicted {target}")
    curve = _learning_curve(eval_pipe, Xtr, ytr, "neg_root_mean_squared_error", cv, seed, higher_better=False)
    temporal = _temporal_check(eval_pipe, d, feats, target, time_col, "regression")
    holdout = pd.DataFrame({"actual": yte.to_numpy(), "predicted": pred}, index=yte.index)
    holdout["residual"] = holdout["actual"] - holdout["predicted"]
    evaluation = {"task": "regression", "selected_model": best["model"], "selection_metric": "cross-validated RMSE (lower is better), one-standard-error rule",
                  "selection_rationale": why, "runner_up": runner["model"] if runner else None, "p_value_vs_runner_up": p_vs_runner,
                  "cv_folds": folds, "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)), "total_labeled_rows": int(len(d)),
                  "metrics": test_m, "cv_metrics": {"RMSE": best["cv_RMSE"], "RMSE_std": best["cv_RMSE_std"], "MAE": best["cv_MAE"], "R2": best["cv_R2"]},
                  "baseline": {"description": "always predicting the training mean", "RMSE": base_rmse},
                  "skill_vs_baseline_%": (1 - test_m["RMSE"] / base_rmse) * 100 if base_rmse else None,
                  "tuning": {k: v for k, v in (tuning or {}).items() if k != "estimator"} | {"adopted": tuned_used} if tuning else None,
                  "learning_curve": curve, "temporal_validation": temporal, "leakage_suspects": leak,
                  "candidates": rows, "features_used": feats, "features_excluded": excluded}
    return TrainedModel("regression", target, feats, excluded, final, eval_pipe, best["model"], evaluation, importance, holdout,
                        X, notes=notes, train_ranges={c: (float(X[c].min()), float(X[c].max())) for c in X.columns if pd.api.types.is_numeric_dtype(X[c])},
                        shap=shap, X_train=Xtr)


def train_classification(d: pd.DataFrame, feats: List[str], excluded, target: str, seed: int, notes: List[str],
                         time_col: Optional[str] = None) -> TrainedModel:
    if len(d) > MAX_FIT_ROWS:
        d = d.sample(MAX_FIT_ROWS, random_state=seed)
        notes.append(f"Training used a random sample of {MAX_FIT_ROWS:,} rows for speed.")
    enc = LabelEncoder()
    y = pd.Series(enc.fit_transform(d[target].astype(str)), index=d.index)
    classes = [str(c) for c in enc.classes_]
    k = len(classes)
    X = prepare_X(d, feats)
    leak = _single_feature_leaks(X, y.to_numpy(), k)
    if leak:
        notes.append(f"Possible leakage: column(s) {', '.join(leak)} separate the classes of '{target}' almost perfectly on their own; "
                     f"check that they are not recorded after (or derived from) the outcome.")
    sel = d.sample(SELECTION_ROWS, random_state=seed).index if len(d) > SELECTION_ROWS else d.index
    if len(d) > SELECTION_ROWS:
        notes.append(f"Model selection used a random sample of {SELECTION_ROWS:,} rows; the winner was refit on all {len(d):,}.")
    Xs, ys = X.loc[sel], y.loc[sel]
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=TEST_SIZE, random_state=seed, stratify=ys)
    folds = max(2, min(5 if len(Xtr) <= LARGE_TRAIN else 3, int(ytr.value_counts().min())))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    majority_acc = float((yte == ytr.value_counts().idxmax()).mean())
    rows, fitted, fold_scores = [], {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, est in _classifiers(k).items():
            try:
                pipe = Pipeline([("prep", build_preprocessor(Xtr)), ("model", clone(est))])
                cvr = cross_validate(pipe, Xtr, ytr, cv=cv, scoring={"f1": "f1_macro", "acc": "accuracy"}, error_score="raise")
                pipe.fit(Xtr, ytr)
                pred = pipe.predict(Xte)
                proba = pipe.predict_proba(Xte)
                auc, ap = None, None
                try:
                    if k == 2:
                        auc = float(roc_auc_score(yte, proba[:, 1]))
                        ap = float(average_precision_score(yte, proba[:, 1]))
                    elif set(np.unique(yte)) == set(range(k)):
                        auc = float(roc_auc_score(yte, proba, multi_class="ovr", average="macro"))
                except Exception:
                    auc = None
                f1f = cvr["test_f1"]
                fold_scores[name] = f1f
                rows.append({"model": name, "cv_F1": float(f1f.mean()), "cv_F1_std": float(f1f.std(ddof=1)) if len(f1f) > 1 else 0.0, "cv_accuracy": float(cvr["test_acc"].mean()),
                             "test_accuracy": float(accuracy_score(yte, pred)), "test_precision": float(precision_score(yte, pred, average="macro", zero_division=0)),
                             "test_recall": float(recall_score(yte, pred, average="macro", zero_division=0)), "test_F1": float(f1_score(yte, pred, average="macro", zero_division=0)),
                             "test_ROC_AUC": auc, "test_PR_AUC": ap})
                fitted[name] = (pipe, pred, proba)
            except Exception as exc:
                rows.append({"model": name, "error": str(exc)[:160]})
    ok = [r for r in rows if "error" not in r]
    if not ok:
        raise UDIError("MODEL_TRAINING_ERROR", "No classification model could be trained on this data: " + "; ".join(r.get("error", "") for r in rows)[:300])
    best, why = _one_se_select(ok, "cv_F1", "cv_F1_std", folds, higher_better=True)
    ranked = sorted(ok, key=lambda r: -r["cv_F1"])
    runner = next((r for r in ranked if r["model"] != best["model"]), None)
    p_vs_runner = _paired_test(fold_scores[best["model"]], fold_scores[runner["model"]]) if runner else None
    eval_pipe, pred, proba = fitted[best["model"]]
    tuning = _tune(eval_pipe, best["model"], Xtr, ytr, "f1_macro", cv, seed)
    tuned_used = False
    if tuning and tuning["best_score"] > tuning["default_score"] * 1.005:
        eval_pipe = tuning["estimator"]
        pred, proba = eval_pipe.predict(Xte), eval_pipe.predict_proba(Xte)
        tuned_used = True
        why += f" Hyperparameter tuning (randomised search, {tuning['candidates_tried']} settings) raised 3-fold CV macro F1 from {tuning['default_score']:.3f} to {tuning['best_score']:.3f} and was adopted."
    elif tuning:
        why += " Hyperparameter tuning did not improve the cross-validated score, so default settings were kept."
    auc = ap = None
    try:
        if k == 2:
            auc, ap = float(roc_auc_score(yte, proba[:, 1])), float(average_precision_score(yte, proba[:, 1]))
        elif set(np.unique(yte)) == set(range(k)):
            auc = float(roc_auc_score(yte, proba, multi_class="ovr", average="macro"))
    except Exception:
        pass
    test_m = {"Accuracy": float(accuracy_score(yte, pred)), "Precision": float(precision_score(yte, pred, average="macro", zero_division=0)),
              "Recall": float(recall_score(yte, pred, average="macro", zero_division=0)), "F1": float(f1_score(yte, pred, average="macro", zero_division=0)),
              "ROC_AUC": auc}
    if ap is not None:
        test_m["PR_AUC"] = ap
    for r in rows:
        r["selected"] = r.get("model") == best["model"]
    rows = sorted(rows, key=lambda r: -r.get("cv_F1", -1))
    final = clone(eval_pipe)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final.fit(X, y)
    importance = _permutation_importance(eval_pipe, Xte, yte.to_numpy(), "f1_macro", seed)
    counts = y.value_counts()
    explain_idx = int(counts.idxmin()) if k == 2 else int(counts.idxmax())      # binary: explain the (usually rarer) event class
    shap = _shap_global(eval_pipe, "classification", Xte, Xtr, seed, f"probability of {target} = '{classes[explain_idx]}'", explain_idx)
    curve = _learning_curve(eval_pipe, Xtr, ytr, "f1_macro", cv, seed, higher_better=True)
    temporal = _temporal_check(eval_pipe, d, feats, target, time_col, "classification", enc)
    cm = confusion_matrix(yte, pred, labels=list(range(k)))
    prec = precision_score(yte, pred, average=None, labels=list(range(k)), zero_division=0)
    rec = recall_score(yte, pred, average=None, labels=list(range(k)), zero_division=0)
    f1c = f1_score(yte, pred, average=None, labels=list(range(k)), zero_division=0)
    per_class = [{"class": classes[i], "precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1c[i]), "support": int((yte == i).sum())} for i in range(k)]
    holdout = pd.DataFrame({"actual": [classes[i] for i in yte.to_numpy()], "predicted": [classes[i] for i in pred], "confidence": proba.max(axis=1)}, index=yte.index)
    dist = {c: int((y == i).sum()) for i, c in enumerate(classes)}
    evaluation = {"task": "classification", "selected_model": best["model"], "selection_metric": "cross-validated macro F1 (higher is better), one-standard-error rule",
                  "selection_rationale": why, "runner_up": runner["model"] if runner else None, "p_value_vs_runner_up": p_vs_runner,
                  "cv_folds": folds, "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)), "total_labeled_rows": int(len(d)), "classes": classes,
                  "metrics": test_m, "cv_metrics": {"F1": best["cv_F1"], "F1_std": best["cv_F1_std"], "Accuracy": best["cv_accuracy"]},
                  "averaging": "macro (every class weighted equally)", "baseline": {"description": "always predicting the most common class", "accuracy": majority_acc},
                  "imbalance_ratio": float(max(dist.values()) / max(1, min(dist.values()))), "class_weighting": "balanced class weights for Logistic / Tree / Random Forest",
                  "confusion_matrix": {"labels": classes, "matrix": cm.tolist()}, "per_class": per_class, "class_distribution": dist,
                  "tuning": ({k_: v for k_, v in tuning.items() if k_ != "estimator"} | {"adopted": tuned_used}) if tuning else None,
                  "learning_curve": curve, "temporal_validation": temporal, "leakage_suspects": leak, "explained_class": classes[explain_idx],
                  "candidates": rows, "features_used": feats, "features_excluded": excluded}
    return TrainedModel("classification", target, feats, excluded, final, eval_pipe, best["model"], evaluation, importance, holdout, X,
                        classes=classes, encoder=enc, notes=notes, shap=shap, X_train=Xtr, explain_index=explain_idx)


def get_model(task: str, vp: ValidatedPlan, ctx: ExecContext, always_include: Optional[List[str]] = None):
    d, feats, excluded, notes = _prep_training_frame(vp, ctx, vp.plan.feature_columns, always_include or [])
    key = _cache_key(task, vp, ctx, feats)
    trainer = train_regression if task == "regression" else train_classification
    time_col = vp.time_column if vp.time_column in d.columns else None

    def factory():
        try:
            return trainer(d, feats, excluded, vp.plan.target_column, ctx.seed, notes, time_col)
        except UDIError:
            raise
        except Exception as exc:
            raise UDIError("MODEL_TRAINING_ERROR", f"Model training failed: {exc}")
    return ctx.cache.get_or_create(key, factory)


# --------------------------------------------------------------------------- prediction input
def _build_input(model: TrainedModel, vp: ValidatedPlan):
    row = typical_row(model.X_reference)
    given: Dict[str, Any] = {}
    for iv in vp.plan.input_values or []:
        if iv.column not in model.features:
            reason = next((e["reason"] for e in model.excluded if e["column"] == iv.column), "not used by the model")
            raise UDIError("INVALID_PLAN", f"'{iv.column}' is not a model input ({reason}); inputs must be among: {', '.join(model.features)}.")
        ref = model.X_reference[iv.column]
        val = float(str(iv.value).replace(",", "")) if pd.api.types.is_numeric_dtype(ref) else str(iv.value)
        row[iv.column] = val
        given[iv.column] = val
    frame = pd.DataFrame([row], columns=model.features)
    for c in model.features:
        if pd.api.types.is_numeric_dtype(model.X_reference[c]):
            frame[c] = pd.to_numeric(frame[c])
    return frame, given, {c: row[c] for c in model.features if c not in given}


def _feature_table(model: TrainedModel, top: int = 15, local: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = {"method": "Permutation importance on held-out rows (drop in model score when a feature is shuffled)",
           "items": model.importance[:top],
           "note": "Importance shows what the model relies on, not what causes the outcome."}
    if model.shap and model.shap.get("items"):
        out["shap"] = {k: v for k, v in model.shap.items() if k != "items"} | {"items": model.shap["items"][:top]}
    if local:
        out["local"] = local
    return out


def _local_shap(model: TrainedModel, frame: pd.DataFrame, seed: int) -> Optional[Dict[str, Any]]:
    try:
        label = model.shap.get("explained_output", "prediction") if model.shap else "prediction"
        return local_explanation(model_predictor(model.final_pipeline, model.task, model.explain_index), frame, model.X_reference, seed, 10, label)
    except Exception:
        return None


def _diagnostic_charts(model: TrainedModel) -> List[Dict[str, Any]]:
    charts = []
    sh = (model.shap or {}).get("items") or []
    if sh:
        charts.append({"type": "bar", "title": "SHAP global importance (mean |SHAP value|)", "x": "feature", "y": "mean_abs_shap",
                       "data": [{"feature": i["feature"], "mean_abs_shap": i["mean_abs_shap"]} for i in sh[:12]], "horizontal": True})
    curve = model.evaluation.get("learning_curve") or []
    if len(curve) >= 3:
        metric = "RMSE" if model.task == "regression" else "macro F1"
        charts.append({"type": "line", "title": f"Learning curve ({metric}, cross-validated)", "x": "training_rows", "y": ["train_score", "validation_score"], "data": curve})
    return charts


def _shared_limits(model: TrainedModel, cached: bool) -> List[str]:
    lim = list(model.notes)
    if model.excluded:
        lim.append("Columns not used as features: " + "; ".join(f"{e['column']} ({e['reason']})" for e in model.excluded[:8]))
    ev = model.evaluation
    if ev["total_labeled_rows"] < 100:
        lim.append(f"Only {ev['total_labeled_rows']} labelled rows were available; metrics from such a small test set are noisy.")
    tv = ev.get("temporal_validation")
    if tv:
        m = tv["metrics"]
        if model.task == "regression" and m.get("R2") is not None and m["R2"] < ev["metrics"]["R2"] - 0.1:
            lim.append(f"Time-ordered validation (train to {tv['train_until']}, test from {tv['test_from']}) gives R² = {m['R2']:.2f}, lower than the random split: relationships may be drifting over time.")
        if model.task == "classification" and m.get("F1") is not None and m["F1"] < ev["metrics"]["F1"] - 0.1:
            lim.append(f"Time-ordered validation gives macro F1 = {m['F1']:.2f}, lower than the random split: the pattern may be drifting over time.")
    p = ev.get("p_value_vs_runner_up")
    if p is not None and p > 0.05 and ev.get("runner_up"):
        lim.append(f"The selected model is not significantly better than the runner-up ({ev['runner_up']}, paired fold test p = {p:.2f}); either is a reasonable choice.")
    return lim


# --------------------------------------------------------------------------- operations
def predict_value(vp: ValidatedPlan, ctx: ExecContext) -> Dict[str, Any]:
    p = vp.plan
    model, cached = get_model("regression", vp, ctx)
    ev = model.evaluation
    m = ev["metrics"]
    frame, given, assumed = _build_input(model, vp)
    pred = float(model.final_pipeline.predict(frame)[0])
    sd = float(model.holdout["residual"].std()) if len(model.holdout) > 2 else None
    lim = _shared_limits(model, cached)
    if m["R2"] < 0.3:
        lim.append(f"The best model explains little of the variation in '{p.target_column}' (held-out R² = {m['R2']:.2f}); treat predictions as rough.")
    if m["R2"] > 0.995:
        lim.append("Near-perfect accuracy usually means the target is (almost) a direct function of other columns; verify this is not circular.")
    lim.append(CAUSAL_NOTE)
    n_cand = len([r for r in ev["candidates"] if "error" not in r])
    summary = (f"{model.selected} was selected from {n_cand} candidate regression models (cross-validated RMSE {fmt_num(ev['cv_metrics']['RMSE'])} ± {fmt_num(ev['cv_metrics']['RMSE_std'])}) "
               f"to predict '{p.target_column}'. On {ev['test_rows']:,} held-out rows it reaches R² = {m['R2']:.3f}, MAE = {fmt_num(m['MAE'])}, RMSE = {fmt_num(m['RMSE'])}"
               f" (mean-prediction baseline RMSE {fmt_num(ev['baseline']['RMSE'])}).")
    if given:
        summary += f" Predicted {p.target_column} for the requested inputs ({', '.join(f'{k}={fmt_num(v) if not isinstance(v, str) else v}' for k, v in given.items())}): {fmt_num(pred)}"
        summary += f" (typical error about ±{fmt_num(m['RMSE'])})."
    else:
        summary += f" For a typical record (median/most-common values of every feature) the model predicts {fmt_num(pred)}; supply feature values to predict a specific case."
    prediction = {"target": p.target_column, "value": pred, "inputs_given": given, "inputs_assumed": assumed, "typical_error_rmse": m["RMSE"],
                  "interval_80": [pred - 1.2816 * sd, pred + 1.2816 * sd] if sd else None,
                  "basis": "user-specified values" if given else "typical record"}
    local = _local_shap(model, frame, ctx.seed)
    hold = model.holdout.reset_index(drop=True)
    scatter = records(hold[["actual", "predicted"]].head(500))
    metrics = {"Selected model": model.selected, "R²": m["R2"], "MAE": m["MAE"], "RMSE": m["RMSE"], "Prediction": pred,
               "Training rows": ev["train_rows"], "Test rows": ev["test_rows"]}
    return make_result("predict_value", summary, title=f"Predicting {p.target_column}", metrics=metrics,
                       table=records(hold.head(12)), table_title="Held-out examples (actual vs predicted)",
                       tables={"candidate_models": ev["candidates"], "feature_importance": model.importance[:15]},
                       charts=[{"type": "scatter", "title": f"Actual vs predicted {p.target_column} (held-out)", "x": "actual", "y": "predicted", "identity": True, "data": scatter}]
                       + ([waterfall_chart(local, f"Why the model predicts {fmt_num(pred)} (SHAP contributions)")] if local else []) + _diagnostic_charts(model),
                       evaluation={**ev, "cached_model": cached}, explainability=_feature_table(model, local=local), prediction=prediction,
                       limitations=lim, assumptions=list(vp.assumptions) + ([f"Features not specified for the prediction were set to typical values: " +
                                                                            ", ".join(f"{k}={fmt_num(v) if not isinstance(v, str) else v}" for k, v in list(assumed.items())[:8])] if assumed else []))


def predict_class(vp: ValidatedPlan, ctx: ExecContext) -> Dict[str, Any]:
    p = vp.plan
    model, cached = get_model("classification", vp, ctx)
    ev = model.evaluation
    m = ev["metrics"]
    frame, given, assumed = _build_input(model, vp)
    proba = model.final_pipeline.predict_proba(frame)[0]
    order = np.argsort(proba)[::-1]
    pred_class = model.classes[int(order[0])]
    probs = [{"class": model.classes[int(i)], "probability": float(proba[int(i)])} for i in order]
    lim = _shared_limits(model, cached)
    dist = ev["class_distribution"]
    minority = min(dist.values()) / sum(dist.values())
    if minority < 0.2:
        lim.append(f"Classes are imbalanced (smallest class is {minority:.0%} of rows); macro-averaged metrics and class weighting were used, but rare classes remain hard to predict.")
    if m["F1"] < 0.5:
        lim.append(f"Macro F1 is {m['F1']:.2f}: the model separates the classes poorly; treat predictions as weak signals.")
    if m["Accuracy"] > 0.995:
        lim.append("Near-perfect accuracy usually means the label is (almost) defined by another column; verify this is not circular.")
    lim.append(CAUSAL_NOTE)
    n_cand = len([r for r in ev["candidates"] if "error" not in r])
    auc_txt = f", ROC-AUC = {m['ROC_AUC']:.3f}" if m.get("ROC_AUC") is not None else ""
    summary = (f"{model.selected} was selected from {n_cand} candidate classifiers (cross-validated macro F1 {ev['cv_metrics']['F1']:.3f}) to predict '{p.target_column}' "
               f"({len(model.classes)} classes). On {ev['test_rows']:,} held-out rows: accuracy {m['Accuracy']:.3f} (majority-class baseline {ev['baseline']['accuracy']:.3f}), "
               f"macro precision {m['Precision']:.3f}, recall {m['Recall']:.3f}, F1 {m['F1']:.3f}{auc_txt}.")
    basis = "the requested inputs" if given else "a typical record"
    summary += f" For {basis} the predicted class is '{pred_class}' (probability {probs[0]['probability']:.0%})."
    local = _local_shap(model, frame, ctx.seed)
    prediction = {"target": p.target_column, "class": pred_class, "probabilities": probs, "inputs_given": given, "inputs_assumed": assumed,
                  "basis": "user-specified values" if given else "typical record"}
    metrics = {"Selected model": model.selected, "Accuracy": m["Accuracy"], "Precision (macro)": m["Precision"], "Recall (macro)": m["Recall"], "F1 (macro)": m["F1"]}
    if m.get("ROC_AUC") is not None:
        metrics["ROC-AUC"] = m["ROC_AUC"]
    metrics["Prediction"] = pred_class
    hold = model.holdout.reset_index(drop=True)
    return make_result("predict_class", summary, title=f"Classifying {p.target_column}", metrics=metrics,
                       table=records(hold.head(12)), table_title="Held-out examples (actual vs predicted class)",
                       tables={"candidate_models": ev["candidates"], "per_class": ev["per_class"], "feature_importance": model.importance[:15]},
                       charts=[{"type": "bar", "title": "Class probabilities for this prediction", "x": "class", "y": "probability", "data": probs[:10]}]
                       + ([waterfall_chart(local, f"Why: SHAP contributions to P({p.target_column} = '{ev['explained_class']}')")] if local else []) + _diagnostic_charts(model),
                       evaluation={**ev, "cached_model": cached}, explainability=_feature_table(model, local=local), prediction=prediction, limitations=lim,
                       assumptions=list(vp.assumptions) + (["Features not specified were set to typical values."] if assumed and not given else []))


def run_scenario(vp: ValidatedPlan, ctx: ExecContext) -> Dict[str, Any]:
    p = vp.plan
    changes = p.scenario_changes or []
    changed = [c.column for c in changes]
    model, cached = get_model("regression", vp, ctx, always_include=changed)
    for c in changed:
        if c not in model.features:
            reason = next((e["reason"] for e in model.excluded if e["column"] == c), "it is not a usable model feature")
            raise UDIError("INVALID_PLAN", f"'{c}' cannot be varied in a scenario: {reason}.")
    base = prepare_X(vp.data, model.features)
    for c in model.features:
        if pd.api.types.is_numeric_dtype(model.X_reference[c]):
            base[c] = pd.to_numeric(base[c], errors="coerce")

    def apply(frame: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
        f = frame.copy()
        for ch in changes:
            v = ch.value * scale
            if ch.change_type == "percent":
                f[ch.column] = f[ch.column] * (1 + v / 100.0)
            elif ch.change_type == "absolute":
                f[ch.column] = f[ch.column] + v
            else:
                f[ch.column] = ch.value
        return f

    scen = apply(base)
    base_pred = model.final_pipeline.predict(base)
    scen_pred = model.final_pipeline.predict(scen)
    b, s = float(np.mean(base_pred)), float(np.mean(scen_pred))
    delta = s - b
    pct = delta / abs(b) * 100 if b else None
    observed = float(vp.data[p.target_column].mean()) if vp.data[p.target_column].notna().any() else None
    desc = []
    for ch in changes:
        desc.append(f"{ch.column} {ch.value:+g}%" if ch.change_type == "percent" else f"{ch.column} {ch.value:+g}" if ch.change_type == "absolute" else f"{ch.column} set to {ch.value:g}")
    changed_txt = ", ".join(desc)
    summary = (f"Applying {changed_txt} changes the model-estimated average '{p.target_column}' from {fmt_num(b)} to {fmt_num(s)} "
               f"({fmt_num(delta)}{f', {pct:+.1f}%' if pct is not None else ''}), using {model.selected} (held-out R² = {model.evaluation['metrics']['R2']:.2f}). "
               f"This is a model-based scenario estimate and does not establish causal impact.")
    lim = _shared_limits(model, cached)
    lim.insert(0, "This is a model-based scenario estimate and does not establish causal impact.")
    m_r2 = model.evaluation["metrics"]["R2"]
    if m_r2 < 0.3:
        lim.append(f"The underlying model is weak (held-out R² = {m_r2:.2f}); the scenario estimate is unreliable.")
    for ch in changes:
        share = next((i["share_%"] for i in model.importance if i["feature"] == ch.column), 0.0)
        if share < 1.0:
            lim.append(f"The model treats '{ch.column}' as nearly irrelevant to '{p.target_column}' (importance share {share:.1f}%), so a near-zero effect may reflect the model, not reality.")
        lo, hi = model.train_ranges.get(ch.column, (None, None))
        if lo is not None:
            out_share = float(((scen[ch.column] > hi) | (scen[ch.column] < lo)).mean())
            if out_share > 0.05:
                lim.append(f"{out_share:.0%} of scenario values for '{ch.column}' fall outside the range seen in training ({fmt_num(lo)}–{fmt_num(hi)}); "
                           f"models extrapolate poorly beyond observed data" + (" (tree-based models hold predictions flat there)." if any(t in model.selected for t in TREE_FAMILY) else "."))
    if observed not in (None, 0) and b and abs(b - observed) / abs(observed) > 0.1:
        lim.append(f"The model's baseline average ({fmt_num(b)}) differs from the observed average ({fmt_num(observed)}) by more than 10%.")
    charts = [{"type": "bar", "title": f"Estimated average {p.target_column}: baseline vs scenario", "x": "state", "y": "value",
               "data": [{"state": "Baseline", "value": b}, {"state": "Scenario", "value": s}]}]
    if len(changes) == 1 and changes[0].change_type == "percent" and changes[0].value != 0:
        mag = max(abs(changes[0].value), 10.0)
        pts = []
        for pc in np.linspace(-mag * 1.5, mag * 1.5, 13):
            one = base.copy()
            one[changes[0].column] = one[changes[0].column] * (1 + pc / 100.0)
            pts.append({"change_%": round(float(pc), 2), f"estimated {p.target_column}": float(np.mean(model.final_pipeline.predict(one)))})
        charts.append({"type": "line", "title": f"Sensitivity of {p.target_column} to {changes[0].column}", "x": "change_%", "y": [f"estimated {p.target_column}"], "data": pts})
    scenario = {"target": p.target_column, "baseline": b, "scenario": s, "change": delta, "change_pct": pct, "observed_mean": observed,
                "changes": [{"column": c.column, "type": c.change_type, "value": c.value} for c in changes], "model": model.selected}
    metrics = {"Baseline (model)": b, "Scenario (model)": s, "Change": delta, "Change %": pct, "Model": model.selected, "Model R² (held-out)": m_r2}
    return make_result("run_scenario", summary, title=f"What if {changed_txt}?", metrics=metrics, charts=charts,
                       table=[{"state": "Baseline", "value": b}, {"state": "Scenario", "value": s}, {"state": "Difference", "value": delta}],
                       table_title="Model-estimated average outcome", tables={"candidate_models": model.evaluation["candidates"], "feature_importance": model.importance[:15]},
                       evaluation={**model.evaluation, "cached_model": cached}, explainability=_feature_table(model), scenario=scenario,
                       limitations=lim, assumptions=list(vp.assumptions) + [f"All rows in the selection were re-scored with {changed_txt}; other columns were left unchanged."])
