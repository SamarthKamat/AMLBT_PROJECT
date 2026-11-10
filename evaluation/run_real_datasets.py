"""Real-world and robustness experiments for the UDI 2.0 report (all numbers in the paper come from this script).

    python evaluation/run_real_datasets.py

1. Four public datasets: profiling, data-quality score, relationship discovery, AutoML (CV mean +- std,
   one-standard-error selection, held-out metrics vs baseline, time-ordered check), SHAP top drivers, runtime.
2. Robustness: controlled defects are injected into IBM HR and IBM Telco (10% MCAR missing values in two numeric
   columns, 2% gross outliers, 3% duplicate rows, 5% case/spacing label variants, 1% impossible negatives). We
   measure the detection recall of the data-quality engine per defect type, the quality score (clean / corrupted /
   after auto-cleaning) and the macro-F1 of the selected classifier on each version.
3. Forecasting: monthly Global Superstore sales, rolling-origin backtest of all candidates against naive baselines.
4. Generated reference domains (kept separate): the dynamic engine on the seven bundled synthetic datasets.
Writes results/real_datasets.json and figures for the paper (results/fig_*.pdf).
"""
from __future__ import annotations
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("UDI_DATA_DIR", tempfile.mkdtemp(prefix="udi_real_"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from context import ExecContext  # noqa: E402
from data_profiler import load_dataset, profile_dataframe  # noqa: E402
from data_quality import assess_quality, clean_dataset, _robust_outliers  # noqa: E402
from executor import execute  # noqa: E402
from relationships import analyse_relationships  # noqa: E402
from schemas import AnalysisPlan  # noqa: E402
from semantics import infer_semantics  # noqa: E402
from validator import roles_from_profile, validate_plan  # noqa: E402

OUT = ROOT / "evaluation" / "results"
REAL = {
    "Global Superstore": ("backend/sample_data/real/global_superstore_sample.csv", "Profit", "predict_value"),
    "IBM HR Attrition": ("backend/sample_data/real/ibm_hr_attrition.csv", "Attrition", "predict_class"),
    "IBM Telco Churn": ("backend/sample_data/real/ibm_telco_churn.csv", "Churn", "predict_class"),
    "UCI Bank Marketing": ("backend/sample_data/real/uci_bank_marketing.csv", "y", "predict_class"),
}


def ctx_for(df, name):
    p = profile_dataframe(df, name)
    r = roles_from_profile(p)
    return p, r, ExecContext(roles=r, profile=p, dataset_id=name, semantics=infer_semantics(df, p), full_data=df)


def run(df, p, r, ctx, plan):
    return execute(validate_plan(AnalysisPlan(**plan), df, p, r), ctx)


def model_summary(res):
    ev = res["evaluation"]
    sh = (res.get("explainability") or {}).get("shap", {}).get("items", [])[:3]
    out = {"selected": ev["selected_model"], "candidates": len([c for c in ev["candidates"] if "error" not in c]), "cv": ev["cv_metrics"],
           "test": ev["metrics"], "baseline": ev["baseline"], "rationale": ev["selection_rationale"], "p_vs_runner_up": ev.get("p_value_vs_runner_up"),
           "temporal": (ev.get("temporal_validation") or {}).get("metrics"), "train_rows": ev["train_rows"], "test_rows": ev["test_rows"],
           "shap_top": [(i["feature"], round(i["share_%"], 1)) for i in sh], "tuning_adopted": (ev.get("tuning") or {}).get("adopted")}
    return out


def real_datasets():
    rows = {}
    for name, (path, target, op) in REAL.items():
        t0 = time.time()
        df = load_dataset(ROOT / path)
        p, r, ctx = ctx_for(df, name)
        t_prof = time.time() - t0
        q = assess_quality(df, p, ctx.semantics)
        t1 = time.time()
        rel = analyse_relationships(df, p, ctx.semantics)
        t_rel = time.time() - t1
        t2 = time.time()
        res = run(df, p, r, ctx, {"operation": op, "target_column": target})
        t_ml = time.time() - t2
        rows[name] = {"rows": len(df), "columns": df.shape[1], "missing_cells": int(df.isna().sum().sum()), "quality_score": round(q["score"], 1), "grade": q["grade"],
                      "issues": len(q["issues"]), "strong_pairs": sum(1 for x in rel["association"]["pairs"] if x["strength"] >= 0.5),
                      "hierarchies": len(rel["hierarchies"]), "derived": [f["formula"] for f in rel["derived"]], "keys": rel["candidate_keys"],
                      "target": target, "task": op, "model": model_summary(res), "seconds": {"profile": round(t_prof, 2), "relationships": round(t_rel, 2), "automl": round(t_ml, 1)}}
        print(name, json.dumps(rows[name]["model"]["test"]), rows[name]["model"]["selected"], f"{t_ml:.0f}s")
    return rows


def inject(df, rng, num_cols, cat_col):
    d = df.copy().reset_index(drop=True)
    n = len(d)
    truth = {}
    miss = []
    for c in num_cols:
        idx = rng.choice(n, int(0.10 * n), replace=False)
        d.loc[idx, c] = np.nan
        miss += [(i, c) for i in idx]
    truth["missing"] = len(miss)
    c0 = num_cols[0]
    out_idx = rng.choice(n, int(0.02 * n), replace=False)
    base = d[c0].dropna()
    d.loc[out_idx, c0] = base.max() * rng.uniform(8, 12, len(out_idx))
    truth["outliers"] = len(out_idx)
    neg_col = num_cols[1]
    neg_idx = rng.choice([i for i in range(n) if pd.notna(d.loc[i, neg_col])], int(0.01 * n), replace=False)
    d.loc[neg_idx, neg_col] = -d.loc[neg_idx, neg_col].abs() - 1
    truth["negatives"] = len(neg_idx)
    d[cat_col] = d[cat_col].astype(object)
    var_idx = rng.choice(n, int(0.05 * n), replace=False)
    d.loc[var_idx, cat_col] = d.loc[var_idx, cat_col].astype(str).str.upper() + " "
    truth["label_variants"] = len(var_idx)
    dup = d.sample(int(0.03 * n), random_state=1)
    d = pd.concat([d, dup], ignore_index=True)
    truth["duplicates"] = len(dup)
    return d, truth, {"outlier_col": c0, "neg_col": neg_col, "out_idx": out_idx, "neg_idx": neg_idx}


def robustness():
    out = {}
    specs = {"IBM HR Attrition": ("backend/sample_data/real/ibm_hr_attrition.csv", ["MonthlyIncome", "TotalWorkingYears"], "Department", "Attrition"),
             "IBM Telco Churn": ("backend/sample_data/real/ibm_telco_churn.csv", ["MonthlyCharges", "tenure"], "Contract", "Churn")}
    for name, (path, nums, cat, target) in specs.items():
        rng = np.random.default_rng(7)
        clean0 = load_dataset(ROOT / path)
        noisy, truth, info = inject(clean0, rng, nums, cat)
        noisy = noisy.reset_index(drop=True)
        res = {"injected": truth}
        versions = {}
        for label, df in (("original", clean0), ("corrupted", noisy)):
            p = profile_dataframe(df, name)
            sem = infer_semantics(df, p)
            versions[label] = (df, p, sem, assess_quality(df, p, sem))
        dfn, pn, semn, qn = versions["corrupted"]
        # detection recall per defect type, measured against what was injected
        det = {"missing": int(dfn[nums].isna().sum().sum() - clean0[nums].isna().sum().sum()), "duplicates": qn["duplicate_rows"]}
        n0 = len(clean0)
        flagged_out = _robust_outliers(dfn[info["outlier_col"]])
        det["outliers"] = int(flagged_out.iloc[info["out_idx"]].sum())
        inv = qn["_invalid"].get(info["neg_col"])
        det["negatives"] = int(inv.iloc[info["neg_idx"]].sum()) if inv is not None else 0
        det["label_variants"] = sum(g["rows_affected"] for g in qn["_variants"].get(cat, []))
        res["detected"] = det
        res["recall"] = {k: round(min(det[k], truth[k]) / truth[k], 3) for k in truth}
        cleaned, log = clean_dataset(dfn, pn, semn)
        pc = profile_dataframe(cleaned, name)
        qc = assess_quality(cleaned, pc, infer_semantics(cleaned, pc))
        res["quality"] = {"original": round(versions["original"][3]["score"], 1), "corrupted": round(qn["score"], 1), "cleaned": round(qc["score"], 1)}
        res["cleaning_actions"] = len(log)
        f1 = {}
        for label, df in (("original", clean0), ("corrupted", dfn), ("cleaned", cleaned)):
            p, r, ctx = ctx_for(df, f"{name}-{label}")
            m = run(df, p, r, ctx, {"operation": "predict_class", "target_column": target})["evaluation"]
            f1[label] = {"model": m["selected_model"], "F1": round(m["metrics"]["F1"], 4), "ROC_AUC": round(m["metrics"]["ROC_AUC"], 4)}
        res["model"] = f1
        out[name] = res
        print(name, res["recall"], res["quality"], {k: v["F1"] for k, v in f1.items()})
    return out


def forecasting():
    df = load_dataset(ROOT / "backend/sample_data/real/global_superstore_sample.csv")
    p, r, ctx = ctx_for(df, "fc")
    res = run(df, p, r, ctx, {"operation": "forecast", "target_column": "Sales", "time_column": "Order Date", "frequency": "M", "forecast_horizon": 6, "aggregation": "sum"})
    ev = res["evaluation"]
    return {"selected": ev["selected_model"], "metrics": ev["metrics"], "baseline": ev["baseline"], "skill_%": ev["skill_vs_baseline_%"],
            "candidates": [{k: c.get(k) for k in ("model", "RMSE", "MAE", "MAPE")} for c in ev["candidates"]], "folds": ev["backtest"]}


def reference_domains():
    targets = {"company": ("Profit", "Risk_Flag"), "hospital": ("Patients", "Capacity_Risk"), "college": ("Attendance", "Attendance_Risk"),
               "municipality": ("Waste_Collected", "Service_Risk"), "ngo": ("Beneficiaries", "Coverage_Risk"),
               "community": ("Resource_Requests", "Shortage_Risk"), "event_organizer": ("Attendance", "Capacity_Risk")}
    out = {}
    for d, (reg, cls) in targets.items():
        df = load_dataset(ROOT / f"backend/sample_data/{d}.csv")
        p, r, ctx = ctx_for(df, d)
        rv = run(df, p, r, ctx, {"operation": "predict_value", "target_column": reg})["evaluation"]
        rc = run(df, p, r, ctx, {"operation": "predict_class", "target_column": cls})["evaluation"]
        out[d] = {"regression": {"target": reg, "model": rv["selected_model"], "R2": round(rv["metrics"]["R2"], 3), "cv_RMSE": round(rv["cv_metrics"]["RMSE"], 2),
                                 "cv_RMSE_std": round(rv["cv_metrics"]["RMSE_std"], 2)},
                  "classification": {"target": cls, "model": rc["selected_model"], "F1": round(rc["metrics"]["F1"], 3), "cv_F1_std": round(rc["cv_metrics"]["F1_std"], 3)}}
        print(d, out[d])
    return out


def figures(res, bench):
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    # robustness figure
    rob = res["robustness"]
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    kinds = ["missing", "duplicates", "outliers", "negatives", "label_variants"]
    w = 0.38
    for i, (name, r) in enumerate(rob.items()):
        ax.bar(np.arange(len(kinds)) + (i - 0.5) * w, [r["recall"][k] * 100 for k in kinds], w, label=name.replace("IBM ", ""), color=["#5b21b6", "#f59e0b"][i])
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(["missing", "duplicate", "outlier", "invalid", "label var."])
    ax.set_ylabel("detection recall (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=6, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_quality_recall.pdf")
    plt.close(fig)
    # benchmark per operation
    if bench:
        po = bench["per_operation_intent"]
        ops = sorted(po, key=lambda o: -po[o]["n"])
        fig, ax = plt.subplots(figsize=(3.4, 2.0))
        ax.bar(range(len(ops)), [po[o]["rate"] * 100 for o in ops], color="#5b21b6")
        ax.set_xticks(range(len(ops)))
        ax.set_xticklabels([o.replace("_", " ") for o in ops], rotation=70, ha="right", fontsize=5.5)
        ax.set_ylabel("intent accuracy (%)")
        ax.set_ylim(0, 105)
        for i, o in enumerate(ops):
            ax.text(i, 101, str(po[o]["n"]), ha="center", fontsize=5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / "fig_benchmark_ops.pdf")
        plt.close(fig)


if __name__ == "__main__":
    t0 = time.time()
    results = {"real_datasets": real_datasets(), "robustness": robustness(), "forecasting": forecasting(), "reference_domains": reference_domains()}
    results["seconds_total"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "real_datasets.json").write_text(json.dumps(results, indent=2, default=str))
    bench = None
    bp = OUT / "benchmark_local.json"
    if bp.exists():
        bench = json.loads(bp.read_text())
    figures(results, bench)
    print("done in", results["seconds_total"], "s")
