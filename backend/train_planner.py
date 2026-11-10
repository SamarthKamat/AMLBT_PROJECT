"""Train (or re-train) the local intent model used by the local planner and report its in-corpus cross-validation.

    python train_planner.py          # writes models/planner/intent_model.joblib

The real routing accuracy is measured on held-out, independently written questions: evaluation/run_benchmark.py.
"""
import json
import time

from sklearn.model_selection import StratifiedKFold, cross_val_score

import local_planner as lp
from planner_corpus import build_corpus

if __name__ == "__main__":
    t0 = time.time()
    X, y = build_corpus()
    b = lp.train_intent_model(save=True)
    cv = cross_val_score(lp._build_pipeline(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    info = {"training_questions": len(X), "intents": len(set(y)), "corpus_fingerprint": b["fingerprint"],
            "in_corpus_cv_accuracy_mean": round(float(cv.mean()), 4), "in_corpus_cv_accuracy_std": round(float(cv.std()), 4),
            "seconds": round(time.time() - t0, 1), "saved_to": str(lp.MODEL_PATH)}
    (lp.MODEL_PATH.parent / "training_report.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
