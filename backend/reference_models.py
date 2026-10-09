"""Domain REFERENCE models: reproducible, domain-labelled artifacts for validation and demonstration.

They are trained by train_models.py on the seven bundled sample datasets. They are NEVER loaded to answer questions
about an uploaded dataset: uploaded data always gets models trained on that data (ml_engine.py).

Uses:
  * `python reference_models.py --validate`  re-scores every artifact on its sample dataset and checks it matches metadata
  * GET /api/reference-models                lists them for the demo UI
  * tests/test_reference_models.py           reproducibility + a benchmark of the dynamic engine against them
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import DOMAINS, MODEL_DIR, SAMPLE_DIR


def _meta(domain: str) -> Dict[str, Any]:
    p = MODEL_DIR / domain / "metadata.json"
    return json.loads(p.read_text(encoding="utf8")) if p.exists() else {}


def list_reference_models() -> List[Dict[str, Any]]:
    out = []
    for d in DOMAINS:
        m = _meta(d)
        if not m:
            continue
        out.append({"domain": d, "label": DOMAINS[d]["label"], "training_rows": m["rows"], "regression_target": m["regression_target"],
                    "classification_target": m["classification_target"], "forecast_target": m["forecast_target"],
                    "selected_regression": m["selected_regression"], "selected_classification": m["selected_classification"],
                    "regression_candidates": m["regression_candidates"], "classification_candidates": m["classification_candidates"],
                    "artifacts": sorted(p.name for p in (MODEL_DIR / d).glob("*.joblib")),
                    "purpose": "reference / validation / demonstration only - uploaded datasets train their own models"})
    return out


def validate_domain(domain: str, tol: float = 1e-6) -> Dict[str, Any]:
    """Reload the selected artifacts, re-score them on the same deterministic split used at training time, compare to metadata."""
    m = _meta(domain)
    df = pd.read_csv(SAMPLE_DIR / f"{domain}.csv")
    reg_t, cls_t, fc_t = m["regression_target"], m["classification_target"], m["forecast_target"]
    X = df.drop(columns=[reg_t, cls_t, fc_t], errors="ignore").drop(columns=["Date"], errors="ignore")
    y = pd.to_numeric(df[reg_t], errors="coerce")
    ok = y.notna()
    X, y = X.loc[ok], y.loc[ok]
    _, Xte, _, yte = train_test_split(X, y, test_size=.2, random_state=42)
    reg_name = Path(m["selected_regression"]).stem.split("_regression_")[1]
    reg = joblib.load(MODEL_DIR / domain / m["selected_regression"])
    r2 = float(r2_score(yte, reg.predict(Xte)))
    expected_r2 = m["regression_candidates"][reg_name]["R2"]

    enc = LabelEncoder()
    ycls = enc.fit_transform(df.loc[ok, cls_t].astype(str))
    _, cxte, _, cyte = train_test_split(X, ycls, test_size=.2, random_state=42, stratify=ycls)
    bundle = joblib.load(MODEL_DIR / domain / m["selected_classification"])
    cls_name = Path(m["selected_classification"]).stem.split("_classification_")[1]
    pred = bundle["pipeline"].predict(cxte)
    acc = float(accuracy_score(cyte, pred))
    f1 = float(f1_score(cyte, pred, average="weighted"))
    exp = m["classification_candidates"][cls_name]
    return {"domain": domain, "regression_model": reg_name, "r2": r2, "expected_r2": expected_r2,
            "regression_reproduced": abs(r2 - expected_r2) <= max(tol, 1e-4),
            "classification_model": cls_name, "accuracy": acc, "f1": f1, "expected_f1": exp["F1"],
            "classification_reproduced": abs(f1 - exp["F1"]) <= max(tol, 1e-4), "artifacts_present": len(list((MODEL_DIR / domain).glob("*.joblib")))}


if __name__ == "__main__":
    if "--validate" in sys.argv:
        bad = 0
        for d in DOMAINS:
            r = validate_domain(d)
            flag = "OK " if r["regression_reproduced"] and r["classification_reproduced"] else "FAIL"
            bad += flag == "FAIL"
            print(f"{flag} {d:16s} regression {r['regression_model']:18s} R2 {r['r2']:.4f} (expected {r['expected_r2']:.4f}) | "
                  f"classification {r['classification_model']:18s} F1 {r['f1']:.4f} (expected {r['expected_f1']:.4f})")
        sys.exit(1 if bad else 0)
    print(json.dumps(list_reference_models(), indent=2)[:2000])
