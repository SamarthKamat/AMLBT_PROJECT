"""Runtime machine learning for uploaded datasets: predict_value, predict_class, run_scenario.

Nothing here is tied to a domain or a column name. For every request the engine
  1. decides which columns are usable features (preprocessing.select_features)
  2. trains the candidate models on the uploaded data
  3. selects one by cross-validation on the training split
  4. reports honest metrics on a held-out test split the selection never saw
  5. refits the winner on all labelled rows (used for predictions / scenarios)
Trained models are cached per session by (task, target, features, filters), so follow-up questions reuse them.
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:                                    # XGBoost is optional ("if available")
    from xgboost import XGBClassifier, XGBRegressor
    HAVE_XGB = True
except Exception:                       # pragma: no cover
    HAVE_XGB = False

from context import MODEL_CONFIG_VERSION, ExecContext
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
        res = permutation_importance(pipe, Xte, yte, scoring=scoring, n_repeats=5, random_state=seed, n_jobs=1)
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


# --------------------------------------------------------------------------- training
def train_regression(d: pd.DataFrame, feats: List[str], excluded, target: str, seed: int, notes: List[str]) -> TrainedModel:
    if len(d) > MAX_FIT_ROWS:
        d = d.sample(MAX_FIT_ROWS, random_state=seed)
        notes.append(f"Training used a random sample of {MAX_FIT_ROWS:,} rows for speed.")
    y = d[target].astype(float)
    X = prepare_X(d, feats)
    leak = near_duplicate_of_target(X, y)
    if leak:
        notes.append(f"Column(s) {', '.join(leak)} are almost perfectly correlated with '{target}'. If the target is calculated from them, "
                     f"the model is learning that formula rather than discovering a relationship.")
    sel = d.sample(SELECTION_ROWS, random_state=seed).index if len(d) > SELECTION_ROWS else d.index
    Xs, ys = X.loc[sel], y.loc[sel]
    if len(d) > SELECTION_ROWS:
        notes.append(f"Model selection used a random sample of {SELECTION_ROWS:,} rows; the winner was refit on all {len(d):,}.")
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=TEST_SIZE, random_state=seed)
    folds = 5 if len(Xtr) >= 100 else 3
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    rows, fitted = [], {}
    base_rmse = float(np.sqrt(mean_squared_error(yte, np.full(len(yte), ytr.mean()))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, est in _regressors(len(Xtr)).items():
            try:
                pipe = Pipeline([("prep", build_preprocessor(Xtr)), ("model", clone(est))])
                cvr = cross_validate(pipe, Xtr, ytr, cv=cv, scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}, error_score="raise")
                pipe.fit(Xtr, ytr)
                pred = pipe.predict(Xte)
                rows.append({"model": name, "cv_RMSE": float(-cvr["test_rmse"].mean()), "cv_MAE": float(-cvr["test_mae"].mean()), "cv_R2": float(cvr["test_r2"].mean()),
                             "test_MAE": float(mean_absolute_error(yte, pred)), "test_RMSE": float(np.sqrt(mean_squared_error(yte, pred))), "test_R2": float(r2_score(yte, pred))})
                fitted[name] = (pipe, pred)
            except Exception as exc:
                rows.append({"model": name, "error": str(exc)[:160]})
    ok = [r for r in rows if "error" not in r]
    if not ok:
        raise UDIError("MODEL_TRAINING_ERROR", "No regression model could be trained on this data: " + "; ".join(r.get("error", "") for r in rows)[:300])
    ok.sort(key=lambda r: r["cv_RMSE"])
    best = ok[0]
    for r in rows:
        r["selected"] = r.get("model") == best["model"]
    rows = sorted(rows, key=lambda r: (r.get("cv_RMSE", float("inf"))))
    eval_pipe, pred = fitted[best["model"]]
    final = clone(eval_pipe)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final.fit(X, y)
    importance = _permutation_importance(eval_pipe, Xte, yte, "neg_root_mean_squared_error", seed)
    holdout = pd.DataFrame({"actual": yte.to_numpy(), "predicted": pred}, index=yte.index)
    holdout["residual"] = holdout["actual"] - holdout["predicted"]
    evaluation = {"task": "regression", "selected_model": best["model"], "selection_metric": "cross-validated RMSE (lower is better)",
                  "cv_folds": folds, "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)), "total_labeled_rows": int(len(d)),
                  "metrics": {"MAE": best["test_MAE"], "RMSE": best["test_RMSE"], "R2": best["test_R2"]},
                  "cv_metrics": {"RMSE": best["cv_RMSE"], "MAE": best["cv_MAE"], "R2": best["cv_R2"]},
                  "baseline": {"description": "always predicting the training mean", "RMSE": base_rmse},
                  "candidates": rows, "features_used": feats, "features_excluded": excluded}
    return TrainedModel("regression", target, feats, excluded, final, eval_pipe, best["model"], evaluation, importance, holdout,
                        X, notes=notes, train_ranges={c: (float(X[c].min()), float(X[c].max())) for c in X.columns if pd.api.types.is_numeric_dtype(X[c])})


def train_classification(d: pd.DataFrame, feats: List[str], excluded, target: str, seed: int, notes: List[str]) -> TrainedModel:
    if len(d) > MAX_FIT_ROWS:
        d = d.sample(MAX_FIT_ROWS, random_state=seed)
        notes.append(f"Training used a random sample of {MAX_FIT_ROWS:,} rows for speed.")
    enc = LabelEncoder()
    y = pd.Series(enc.fit_transform(d[target].astype(str)), index=d.index)
    classes = [str(c) for c in enc.classes_]
    k = len(classes)
    X = prepare_X(d, feats)
    sel = d.sample(SELECTION_ROWS, random_state=seed).index if len(d) > SELECTION_ROWS else d.index
    if len(d) > SELECTION_ROWS:
        notes.append(f"Model selection used a random sample of {SELECTION_ROWS:,} rows; the winner was refit on all {len(d):,}.")
    Xs, ys = X.loc[sel], y.loc[sel]
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=TEST_SIZE, random_state=seed, stratify=ys)
    folds = max(2, min(5, int(ytr.value_counts().min())))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    majority_acc = float((yte == ytr.value_counts().idxmax()).mean())
    rows, fitted = [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, est in _classifiers(k).items():
            try:
                pipe = Pipeline([("prep", build_preprocessor(Xtr)), ("model", clone(est))])
                cvr = cross_validate(pipe, Xtr, ytr, cv=cv, scoring={"f1": "f1_macro", "acc": "accuracy"}, error_score="raise")
                pipe.fit(Xtr, ytr)
                pred = pipe.predict(Xte)
                proba = pipe.predict_proba(Xte)
                auc = None
                try:
                    if k == 2:
                        auc = float(roc_auc_score(yte, proba[:, 1]))
                    elif set(np.unique(yte)) == set(range(k)):
                        auc = float(roc_auc_score(yte, proba, multi_class="ovr", average="macro"))
                except Exception:
                    auc = None
                rows.append({"model": name, "cv_F1": float(cvr["test_f1"].mean()), "cv_accuracy": float(cvr["test_acc"].mean()),
                             "test_accuracy": float(accuracy_score(yte, pred)), "test_precision": float(precision_score(yte, pred, average="macro", zero_division=0)),
                             "test_recall": float(recall_score(yte, pred, average="macro", zero_division=0)), "test_F1": float(f1_score(yte, pred, average="macro", zero_division=0)),
                             "test_ROC_AUC": auc})
                fitted[name] = (pipe, pred, proba)
            except Exception as exc:
                rows.append({"model": name, "error": str(exc)[:160]})
    ok = [r for r in rows if "error" not in r]
    if not ok:
        raise UDIError("MODEL_TRAINING_ERROR", "No classification model could be trained on this data: " + "; ".join(r.get("error", "") for r in rows)[:300])
    ok.sort(key=lambda r: r["cv_F1"], reverse=True)
    best = ok[0]
    for r in rows:
        r["selected"] = r.get("model") == best["model"]
    rows = sorted(rows, key=lambda r: -r.get("cv_F1", -1))
    eval_pipe, pred, proba = fitted[best["model"]]
    final = clone(eval_pipe)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final.fit(X, y)
    importance = _permutation_importance(eval_pipe, Xte, yte.to_numpy(), "f1_macro", seed)
    cm = confusion_matrix(yte, pred, labels=list(range(k)))
    prec = precision_score(yte, pred, average=None, labels=list(range(k)), zero_division=0)
    rec = recall_score(yte, pred, average=None, labels=list(range(k)), zero_division=0)
    f1c = f1_score(yte, pred, average=None, labels=list(range(k)), zero_division=0)
    per_class = [{"class": classes[i], "precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1c[i]), "support": int((yte == i).sum())} for i in range(k)]
    holdout = pd.DataFrame({"actual": [classes[i] for i in yte.to_numpy()], "predicted": [classes[i] for i in pred], "confidence": proba.max(axis=1)}, index=yte.index)
    evaluation = {"task": "classification", "selected_model": best["model"], "selection_metric": "cross-validated macro F1 (higher is better)",
                  "cv_folds": folds, "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)), "total_labeled_rows": int(len(d)), "classes": classes,
                  "metrics": {"Accuracy": best["test_accuracy"], "Precision": best["test_precision"], "Recall": best["test_recall"], "F1": best["test_F1"], "ROC_AUC": best["test_ROC_AUC"]},
                  "averaging": "macro (every class weighted equally)", "baseline": {"description": "always predicting the most common class", "accuracy": majority_acc},
                  "confusion_matrix": {"labels": classes, "matrix": cm.tolist()}, "per_class": per_class,
                  "class_distribution": {c: int((y == i).sum()) for i, c in enumerate(classes)},
                  "candidates": rows, "features_used": feats, "features_excluded": excluded}
    return TrainedModel("classification", target, feats, excluded, final, eval_pipe, best["model"], evaluation, importance, holdout, X,
                        classes=classes, encoder=enc, notes=notes)


def get_model(task: str, vp: ValidatedPlan, ctx: ExecContext, always_include: Optional[List[str]] = None):
    d, feats, excluded, notes = _prep_training_frame(vp, ctx, vp.plan.feature_columns, always_include or [])
    key = _cache_key(task, vp, ctx, feats)
    trainer = train_regression if task == "regression" else train_classification

    def factory():
        try:
            return trainer(d, feats, excluded, vp.plan.target_column, ctx.seed, notes)
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


def _feature_table(model: TrainedModel, top: int = 15) -> Dict[str, Any]:
    return {"method": "Permutation importance on held-out rows (drop in model score when a feature is shuffled)",
            "items": model.importance[:top],
            "note": "Importance shows what the model relies on, not what causes the outcome."}


def _shared_limits(model: TrainedModel, cached: bool) -> List[str]:
    lim = list(model.notes)
    if model.excluded:
        lim.append("Columns not used as features: " + "; ".join(f"{e['column']} ({e['reason']})" for e in model.excluded[:8]))
    ev = model.evaluation
    if ev["total_labeled_rows"] < 100:
        lim.append(f"Only {ev['total_labeled_rows']} labelled rows were available; metrics from such a small test set are noisy.")
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
    summary = (f"{model.selected} was selected from {n_cand} candidate regression models (lowest cross-validated RMSE {fmt_num(ev['cv_metrics']['RMSE'])}) "
               f"to predict '{p.target_column}'. On {ev['test_rows']:,} held-out rows it reaches R² = {m['R2']:.3f}, MAE = {fmt_num(m['MAE'])}, RMSE = {fmt_num(m['RMSE'])}.")
    if given:
        summary += f" Predicted {p.target_column} for the requested inputs ({', '.join(f'{k}={fmt_num(v) if not isinstance(v, str) else v}' for k, v in given.items())}): {fmt_num(pred)}"
        summary += f" (typical error about ±{fmt_num(m['RMSE'])})."
    else:
        summary += f" For a typical record (median/most-common values of every feature) the model predicts {fmt_num(pred)}; supply feature values to predict a specific case."
    prediction = {"target": p.target_column, "value": pred, "inputs_given": given, "inputs_assumed": assumed, "typical_error_rmse": m["RMSE"],
                  "interval_80": [pred - 1.2816 * sd, pred + 1.2816 * sd] if sd else None,
                  "basis": "user-specified values" if given else "typical record"}
    hold = model.holdout.reset_index(drop=True)
    scatter = records(hold[["actual", "predicted"]].head(500))
    metrics = {"Selected model": model.selected, "R²": m["R2"], "MAE": m["MAE"], "RMSE": m["RMSE"], "Prediction": pred,
               "Training rows": ev["train_rows"], "Test rows": ev["test_rows"]}
    return make_result("predict_value", summary, title=f"Predicting {p.target_column}", metrics=metrics,
                       table=records(hold.head(12)), table_title="Held-out examples (actual vs predicted)",
                       tables={"candidate_models": ev["candidates"], "feature_importance": model.importance[:15]},
                       charts=[{"type": "scatter", "title": f"Actual vs predicted {p.target_column} (held-out)", "x": "actual", "y": "predicted", "identity": True, "data": scatter}],
                       evaluation={**ev, "cached_model": cached}, explainability=_feature_table(model), prediction=prediction,
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
    summary = (f"{model.selected} was selected from {n_cand} candidate classifiers (highest cross-validated macro F1 {ev['candidates'][0]['cv_F1']:.3f}) to predict '{p.target_column}' "
               f"({len(model.classes)} classes). On {ev['test_rows']:,} held-out rows: accuracy {m['Accuracy']:.3f} (majority-class baseline {ev['baseline']['accuracy']:.3f}), "
               f"macro precision {m['Precision']:.3f}, recall {m['Recall']:.3f}, F1 {m['F1']:.3f}{auc_txt}.")
    basis = "the requested inputs" if given else "a typical record"
    summary += f" For {basis} the predicted class is '{pred_class}' (probability {probs[0]['probability']:.0%})."
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
                       charts=[{"type": "bar", "title": "Class probabilities for this prediction", "x": "class", "y": "probability", "data": probs[:10]}],
                       evaluation={**ev, "cached_model": cached}, explainability=_feature_table(model), prediction=prediction, limitations=lim,
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
