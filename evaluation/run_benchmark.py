"""Held-out routing and end-to-end correctness benchmark for UDI 2.0.

    python evaluation/run_benchmark.py --planner local      # offline, reproducible (no API key needed)
    python evaluation/run_benchmark.py --planner gemini     # needs GEMINI_API_KEY / GEMINI_MODEL in backend/.env

Every question in benchmark_questions.jsonl goes through the REAL pipeline (Flask API -> planner -> validation ->
operation). It is scored on:
  intent      the planned operation equals the gold operation
  columns     the planned target / group-by columns equal the gold ones (where the gold specifies them)
  filters     gold filters are present in the plan
  rejection   questions that must fail (missing column, unsupported, insufficient data) are refused, not answered
  numeric     for questions with a closed-form answer the returned number / group equals an independent pandas oracle
  end_to_end  all applicable checks pass
Writes results/benchmark_<planner>.json and a per-question pass/fail log results/benchmark_<planner>_log.csv.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("UDI_DATA_DIR", tempfile.mkdtemp(prefix="udi_bench_"))
for k in ("UDI_RATE_CHAT", "UDI_RATE_UPLOAD", "UDI_RATE_REPORT"):
    os.environ[k] = "100000"

import pandas as pd  # noqa: E402

DATASETS = {"superstore": "backend/sample_data/real/global_superstore_sample.csv", "hr": "backend/sample_data/real/ibm_hr_attrition.csv",
            "telco": "backend/sample_data/real/ibm_telco_churn.csv", "bank": "backend/sample_data/real/uci_bank_marketing.csv",
            "company": "backend/sample_data/company.csv", "hospital": "backend/sample_data/hospital.csv"}
AGG_LABEL = {"mean": "Average", "sum": "Total", "median": "Median", "min": "Minimum", "max": "Maximum"}
POS = ("yes", "1", "true", "churned", "late", "left", "default", "y")


def oracle(item, df):
    """Independent pandas answer for closed-form questions: ('group', name) or ('value', number) or None."""
    op, t, g, agg = item["op"], item["target"], item["group"], item["agg"]
    d = df
    for col, val in item.get("filters") or []:
        d = d[d[col].astype(str) == val]
    asc = any(w in item["q"].lower() for w in ("lowest", "least", "fewest", "worst"))
    if op == "compare_groups" and g and t and agg in ("sum", "mean"):
        s = getattr(d.groupby(g[0])[t], agg)()
        return ("group", str(s.idxmin() if asc else s.idxmax()))
    if op == "compare_groups" and g and t and d[t].dtype == object:
        pos = next((l for l in d[t].astype(str).unique() if l.lower() in POS), None)
        if pos:
            s = (d[t].astype(str) == pos).groupby(d[g[0]]).mean()
            return ("group", str(s.idxmin() if asc else s.idxmax()))
    if op == "compare_groups" and g and not t:
        s = d.groupby(g[0]).size()
        return ("group", str(s.idxmin() if asc else s.idxmax()))
    if op == "get_summary" and t and agg in AGG_LABEL:
        return ("value", float(getattr(pd.to_numeric(d[t], errors="coerce").dropna(), agg)()))
    if op == "get_summary" and agg == "count" and not t:
        return ("value", float(len(d)))
    return None


def check_numeric(item, res, exp):
    m = (res.get("result") or {}).get("metrics") or {}
    kind, val = exp
    if kind == "group":
        got = m.get("Top group") or next((v for k, v in m.items() if k.startswith("Highest share")), None)
        return str(got) == val, got
    if item["agg"] == "count":
        got = m.get("Records")
    else:
        got = m.get(f"{AGG_LABEL[item['agg']]} of {item['target']}")
    return (got is not None and abs(float(got) - val) <= 1e-6 * max(1.0, abs(val))), got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", default="local", choices=["local", "gemini", "auto"])
    args = ap.parse_args()
    from app import create_app
    from gemini_client import GeminiService
    app = create_app(llm=GeminiService(), planner_mode=args.planner)
    c = app.test_client()
    store = app.extensions["udi_sessions"]
    items = [json.loads(l) for l in open(ROOT / "evaluation" / "benchmark_questions.jsonl")]
    sids, frames, rows = {}, {}, []
    t_all = time.time()
    for it in items:
        ds = it["dataset"]
        if ds not in sids:
            p = ROOT / DATASETS[ds]
            r = c.post("/api/upload", data={"domain": "company" if ds not in ("hospital",) else "hospital", "file": (io.BytesIO(p.read_bytes()), p.name)},
                       content_type="multipart/form-data")
            sids[ds] = r.get_json()["session_id"]
            from data_profiler import load_dataset
            frames[ds] = load_dataset(p)
        s = store.get(sids[ds])
        s.reset_conversation()                             # every question is asked fresh (no follow-up context)
        t0 = time.time()
        r = c.post("/api/chat", json={"session_id": sids[ds], "message": it["q"]})
        body = r.get_json()
        secs = time.time() - t0
        plan = body.get("plan") or {}
        answered = r.status_code == 200 and body.get("type") == "answer"
        row = {"id": it["id"], "dataset": ds, "question": it["q"], "gold_op": it["op"], "planned_op": plan.get("operation"),
               "gold_target": it["target"], "planned_target": plan.get("target_column"), "gold_group": ";".join(it["group"] or []),
               "planned_group": ";".join((plan.get("group_by") or [])[:1]), "status": body.get("type") or body.get("code"), "seconds": round(secs, 2)}
        row["intent_ok"] = plan.get("operation") == it["op"]
        col_ok = True
        if it["target"] is not None:
            col_ok &= plan.get("target_column") == it["target"]
        if it["group"]:
            col_ok &= (plan.get("group_by") or [None])[0] == it["group"][0]
        row["columns_ok"] = col_ok if (it["target"] is not None or it["group"]) else None
        if it.get("filters"):
            pf = {(f.get("column"), f.get("value")) for f in plan.get("filters") or []} | {(f.get("column"), v) for f in plan.get("filters") or [] for v in (f.get("values") or [])}
            row["filters_ok"] = all((cn, v) in pf for cn, v in it["filters"])
        else:
            row["filters_ok"] = None
        if it["error"]:
            row["rejection_ok"] = (not answered) and (body.get("code") == it["error"] or body.get("type") == "clarification" or body.get("code") in ("INVALID_COLUMN", "UNSUPPORTED_OPERATION", "INSUFFICIENT_DATA"))
            row["intent_ok"] = row["intent_ok"] or row["rejection_ok"]        # for refusals the outcome, not the label, is what matters
            row["columns_ok"] = None
        else:
            row["rejection_ok"] = None
            row["answered"] = answered
        exp = oracle(it, frames[ds]) if not it["error"] else None
        if exp and answered:
            ok, got = check_numeric(it, body, exp)
            row.update(numeric_ok=ok, oracle=exp[1], returned=got)
        else:
            row["numeric_ok"] = None if not exp else False
        checks = [row[k] for k in ("intent_ok", "columns_ok", "filters_ok", "rejection_ok", "numeric_ok") if row.get(k) is not None]
        if not it["error"]:
            checks.append(answered)
        row["pass"] = all(checks)
        rows.append(row)
        print(f"{'PASS' if row['pass'] else 'FAIL'} {it['id']:14s} {it['q'][:60]:60s} -> {row['planned_op']} {row['planned_target']} {row['planned_group']} [{row['status']}]")

    def rate(key, subset=None):
        vals = [r[key] for r in (subset or rows) if r.get(key) is not None]
        return {"correct": int(sum(vals)), "n": len(vals), "rate": round(sum(vals) / len(vals), 4) if vals else None}
    valid = [r for r in rows if r["gold_op"] and not next(i for i in items if i["id"] == r["id"])["error"]]
    summary = {"planner": args.planner, "questions": len(rows), "datasets": len(sids), "seconds_total": round(time.time() - t_all, 1),
               "intent_accuracy": rate("intent_ok"), "intent_accuracy_answerable": rate("intent_ok", valid), "column_mapping_accuracy": rate("columns_ok"),
               "filter_accuracy": rate("filters_ok"), "invalid_request_rejection": rate("rejection_ok"), "numerical_correctness": rate("numeric_ok"),
               "answered_when_answerable": rate("answered", valid), "end_to_end_pass": rate("pass"),
               "median_seconds": float(pd.Series([r["seconds"] for r in rows]).median()),
               "per_dataset": {ds: rate("pass", [r for r in rows if r["dataset"] == ds]) for ds in sids},
               "per_operation_intent": {op: rate("intent_ok", [r for r in rows if r["gold_op"] == op]) for op in sorted({r["gold_op"] for r in rows})}}
    out = ROOT / "evaluation" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"benchmark_{args.planner}.json").write_text(json.dumps(summary, indent=2))
    with open(out / f"benchmark_{args.planner}_log.csv", "w", newline="", encoding="utf-8") as f:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_operation_intent",)}, indent=2))


if __name__ == "__main__":
    main()
