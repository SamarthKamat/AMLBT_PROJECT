"""Feature selection and scikit-learn preprocessing for runtime model training.

Rules (all data-driven, none name-driven except weak identifier hints in data_profiler):
  * numeric / binary / low-cardinality categorical columns are usable features
  * identifier-like, free-text, constant, empty, datetime and very-high-cardinality columns are excluded (with a reason)
  * high-cardinality *numeric* business variables (Sales, Revenue, ...) are never dropped for having many distinct values
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from errors import UDIError

MAX_CATEGORIES = 50
MAX_MISSING_FRACTION = 0.6


def select_features(df: pd.DataFrame, roles: Dict[str, Dict[str, Any]], target: str,
                    requested: Optional[Sequence[str]] = None,
                    always_include: Sequence[str] = ()) -> Tuple[List[str], List[Dict[str, str]]]:
    """Return (usable_feature_columns, excluded=[{column, reason}])."""
    excluded: List[Dict[str, str]] = []
    pool = [c for c in df.columns if c != target]
    if requested:
        req = [c for c in requested if c in df.columns and c != target]
        for c in requested:
            if c == target:
                excluded.append({"column": c, "reason": "is the prediction target"})
        pool = list(dict.fromkeys(req + [c for c in always_include if c in df.columns and c not in req and c != target]))
    feats: List[str] = []
    for c in pool:
        role = roles.get(c, {}).get("role", "categorical")
        s = df[c]
        if role == "identifier":
            excluded.append({"column": c, "reason": "identifier-like column"})
        elif role == "datetime":
            excluded.append({"column": c, "reason": "date/time columns are not used as tabular features"})
        elif role == "text":
            excluded.append({"column": c, "reason": "free-text column"})
        elif role in ("constant", "empty"):
            excluded.append({"column": c, "reason": "no variation" if role == "constant" else "no values"})
        elif s.isna().mean() > MAX_MISSING_FRACTION:
            excluded.append({"column": c, "reason": f"{s.isna().mean():.0%} of values missing"})
        elif role == "categorical" and not pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > MAX_CATEGORIES:
            excluded.append({"column": c, "reason": f"{s.nunique()} distinct categories (too many to encode reliably)"})
        else:
            feats.append(c)
    if not requested:
        # a non-requested column excluded for being a "pool" artefact needs no mention beyond the reason above
        pass
    return feats, excluded


def prepare_X(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    X = df[list(features)].copy()
    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(int)
        elif not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].map(lambda v: v if pd.isna(v) else str(v)).astype(object)
    return X


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    nums = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cats = [c for c in X.columns if c not in nums]
    steps = []
    if nums:
        steps.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), nums))
    if cats:
        steps.append(("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                                       ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cats))
    if not steps:
        raise UDIError("INSUFFICIENT_DATA", "No usable feature columns remain after excluding identifiers, text and constant columns.")
    return ColumnTransformer(steps, remainder="drop")


def typical_row(X: pd.DataFrame) -> Dict[str, Any]:
    """Median for numeric features, most frequent value for categorical ones."""
    row: Dict[str, Any] = {}
    for c in X.columns:
        s = X[c].dropna()
        if s.empty:
            row[c] = np.nan if pd.api.types.is_numeric_dtype(X[c]) else "missing"
        elif pd.api.types.is_numeric_dtype(X[c]):
            row[c] = float(s.median())
        else:
            row[c] = s.mode().iloc[0]
    return row


def near_duplicate_of_target(X: pd.DataFrame, y: pd.Series, threshold: float = 0.995) -> List[str]:
    flags = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]) and pd.api.types.is_numeric_dtype(y):
            pair = pd.concat([X[c], y], axis=1).dropna()
            if len(pair) > 10 and pair.iloc[:, 0].std() > 0 and pair.iloc[:, 1].std() > 0:
                if abs(pair.iloc[:, 0].corr(pair.iloc[:, 1])) >= threshold:
                    flags.append(c)
    return flags
