"""Exact McNemar tests between systems on the same frozen question set, and public vs generated splits.

    python evaluation/paired_tests.py        # reads results/benchmark_*_log.csv, writes results/paired_tests.json
"""
from __future__ import annotations
import csv
import json
from math import comb
from pathlib import Path

from run_benchmark import wilson

OUT = Path(__file__).resolve().parent / "results"
PUBLIC = {"superstore", "hr", "telco", "bank"}
RUNS = {"v1": {"full": "benchmark_v1_local_current", "keyword": "benchmark_v1_keyword", "local_exact": "benchmark_v1_local_exact"},
        "v2": {"full": "benchmark_v2_local_heldout", "keyword": "benchmark_v2_keyword", "local_exact": "benchmark_v2_local_exact"},
        "v3": {"full": "benchmark_v3_local", "keyword": "benchmark_v3_keyword", "local_exact": "benchmark_v3_local_exact", "gemini": "benchmark_v3_gemini", "gemini31": "benchmark_v3_gemini31"}}


def load(tag):
    p = OUT / f"{tag}_log.csv"
    if not p.exists():
        return None
    return {r["id"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}


def ok(r, key="pass"):
    return r.get(key) == "True"


def mcnemar(a, b):
    ids = sorted(set(a) & set(b))
    x = sum(ok(a[i]) and not ok(b[i]) for i in ids)
    y = sum(ok(b[i]) and not ok(a[i]) for i in ids)
    n, k = x + y, min(x, y)
    p = min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    return {"first_only": x, "second_only": y, "mcnemar_exact_p": p}


def rate(rows, key="pass"):
    vals = [ok(r, key) for r in rows if r.get(key) not in ("", None)]
    k = sum(vals)
    return {"correct": k, "n": len(vals), "rate": round(k / len(vals), 4) if vals else None, "ci95": wilson(k, len(vals))}


def main():
    res = {"pairs": {}, "splits": {}}
    for qs, runs in RUNS.items():
        logs = {k: load(t) for k, t in runs.items()}
        logs = {k: v for k, v in logs.items() if v}
        for other in [k for k in logs if k != "full"]:
            res["pairs"][f"{qs}_full_vs_{other}"] = mcnemar(logs["full"], logs[other])
        for k, lg in logs.items():
            rows = list(lg.values())
            pub = [r for r in rows if r["dataset"] in PUBLIC]
            gen = [r for r in rows if r["dataset"] not in PUBLIC]
            res["splits"][f"{qs}_{k}"] = {"all": rate(rows), "public": rate(pub), "generated": rate(gen),
                                         "refusal_all": rate(rows, "rejection_ok"), "refusal_public": rate(pub, "rejection_ok")}
    # UDI systems vs the plain LLM (llm_direct_v3.json) on the closed-form and refusal questions of v3
    llm_path = OUT / "llm_direct_v3.json"
    if llm_path.exists():
        llm = {r["id"]: r["ok"] for r in json.loads(llm_path.read_text())["rows"] if r["ok"] is not None}
        res["vs_plain_llm"] = {}
        for k in ("full", "gemini", "gemini31"):
            lg = load(RUNS["v3"][k])
            if not lg:
                continue
            out = {}
            for part, key in (("closed_form", "numeric_ok"), ("refusal", "rejection_ok")):
                a = {i: r[key] == "True" for i, r in lg.items() if r.get(key) in ("True", "False") and i in llm}
                b = {i: {"pass": "True" if llm[i] else "False"} for i in a}
                out[part] = mcnemar({i: {"pass": "True" if v else "False"} for i, v in a.items()}, b) | {"n": len(a)}
            res["vs_plain_llm"][k] = out
            print(f"v3 {k:9s} vs plain LLM {out}")
    (OUT / "paired_tests.json").write_text(json.dumps(res, indent=2))
    for k, v in res["pairs"].items():
        print(f"{k:28s} {v}")
    for k, v in res["splits"].items():
        print(f"{k:22s} all {v['all']['correct']}/{v['all']['n']}  public {v['public']['correct']}/{v['public']['n']} ({v['public']['rate']})  "
              f"generated {v['generated']['correct']}/{v['generated']['n']}  refusal {v['refusal_all']['correct']}/{v['refusal_all']['n']}")


if __name__ == "__main__":
    main()
