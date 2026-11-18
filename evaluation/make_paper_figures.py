"""Print-friendly figures for the paper, drawn from UDI's own verified outputs (not screenshots).

    python evaluation/make_paper_figures.py

fig_example.pdf      (a) the answer to "Which country generated the most sales?" (compare_groups result) and
                     (b) the signed association heatmap from describe_relationships, Global Superstore
fig_benchmark.pdf    end-to-end accuracy with Wilson 95% CIs: keyword baseline vs exact-linking ablation vs UDI
                     local planner, on benchmark v1 (run 1) and v2 (fresh held-out)
"""
from __future__ import annotations
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("UDI_DATA_DIR", tempfile.mkdtemp(prefix="udi_fig_"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from context import ExecContext  # noqa: E402
from data_profiler import load_dataset, profile_dataframe  # noqa: E402
from executor import execute  # noqa: E402
from schemas import AnalysisPlan  # noqa: E402
from semantics import infer_semantics  # noqa: E402
from validator import roles_from_profile, validate_plan  # noqa: E402

OUT = ROOT / "evaluation" / "results"
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})


def example_figure():
    df = load_dataset(ROOT / "backend/sample_data/real/global_superstore_sample.csv")
    p = profile_dataframe(df)
    r = roles_from_profile(p)
    ctx = ExecContext(roles=r, profile=p, semantics=infer_semantics(df, p), full_data=df)
    run = lambda plan: execute(validate_plan(AnalysisPlan(**plan), df, p, r), ctx)
    cg = run({"operation": "compare_groups", "target_column": "Sales", "group_by": ["Country"], "aggregation": "sum", "sort_order": "desc"})
    rel = run({"operation": "describe_relationships"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1, 1.25]})
    rows = cg["table"][:10][::-1]
    a.barh([x["Country"] for x in rows], [x["Total of Sales"] for x in rows], color="#5b21b6")
    a.set_xlabel("total sales (thousands)")
    a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}"))
    a.set_title("(a) Which country generated the most sales?\ncompare_groups: Australia 190,369 (93 countries; KW p<1e-69)", fontsize=6.5)
    hm = rel["charts"][0]
    cols = hm["columns"][:12]
    M = np.array([[np.nan if v is None else v for v in row[:12]] for row in hm["matrix"][:12]], dtype=float)
    im = b.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    b.set_xticks(range(len(cols)))
    b.set_xticklabels(cols, rotation=60, ha="right", fontsize=5.5)
    b.set_yticks(range(len(cols)))
    b.set_yticklabels(cols, fontsize=5.5)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if not np.isnan(M[i, j]) and i != j and abs(M[i, j]) >= 0.5:
                b.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=4.2, color="white" if abs(M[i, j]) > 0.75 else "black")
    b.set_title("(b) describe_relationships: signed r (numeric pairs),\nη / Cramér's V (pairs with a category)", fontsize=6.5)
    fig.colorbar(im, ax=b, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_example.pdf")
    plt.close(fig)


def benchmark_figure():
    systems = [("Keyword router", "keyword"), ("UDI, exact linking", "local_exact"), ("UDI local planner", None)]
    sets = [("v1 (n=104, same code)", "v1"), ("v2 (fresh held-out, n=99)", "v2")]
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    w = 0.26
    colors = ["#9ca3af", "#f59e0b", "#5b21b6"]
    for si, (label, key) in enumerate(systems):
        vals, lo, hi = [], [], []
        for _, tag in sets:
            f = OUT / (f"benchmark_{tag}_{key}.json" if key else ("benchmark_v1_local_current.json" if tag == "v1" else "benchmark_v2_local_heldout.json"))
            e = json.loads(f.read_text())["end_to_end_pass"]
            vals.append(e["rate"] * 100)
            lo.append(e["rate"] * 100 - e["ci95"][0] * 100)
            hi.append(e["ci95"][1] * 100 - e["rate"] * 100)
        x = np.arange(len(sets)) + (si - 1) * w
        ax.bar(x, vals, w, yerr=[lo, hi], capsize=2, label=label, color=colors[si], error_kw={"elinewidth": 0.7})
        for xi, v in zip(x, vals):
            ax.text(xi, 2, f"{v:.0f}", ha="center", fontsize=5.5, color="white")
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([s for s, _ in sets])
    ax.set_ylabel("end-to-end correct (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=5.5, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(OUT / "fig_benchmark.pdf")
    plt.close(fig)


if __name__ == "__main__":
    example_figure()
    benchmark_figure()
    print("figures written to", OUT)


# ----------------------------------------------------------------------------- results gallery (6 panels)
def _ctx(path):
    df = load_dataset(ROOT / path)
    p = profile_dataframe(df)
    r = roles_from_profile(p)
    return df, p, r, ExecContext(roles=r, profile=p, semantics=infer_semantics(df, p), full_data=df)


def _holdout_proba(ctx):
    m = list(ctx.cache._d.values())[-1]
    X = m.X_reference.loc[m.holdout.index]
    y = m.encoder.transform(m.holdout["actual"].astype(str))
    pos = m.explain_index if m.explain_index is not None else 1
    return (y == pos).astype(int), m.eval_pipeline.predict_proba(X)[:, pos]


def gallery_figure():
    from sklearn.metrics import roc_auc_score, roc_curve
    fig, ax = plt.subplots(2, 3, figsize=(7.1, 4.3))
    C = ["#5b21b6", "#f59e0b", "#10b981", "#ef4444", "#3b82f6"]
    # (a) ROC curves, incl. the leakage pair
    a = ax[0, 0]
    runs = [("HR attrition", "backend/sample_data/real/ibm_hr_attrition.csv", "Attrition", None),
            ("Telco churn", "backend/sample_data/real/ibm_telco_churn.csv", "Churn", None),
            ("Bank, all cols (leaky)", "backend/sample_data/real/uci_bank_marketing.csv", "y", None),
            ("Bank, pre-call", "backend/sample_data/real/uci_bank_marketing.csv", "y", ["duration"])]
    hr_shap = None
    for i, (lab, path, tgt, ex) in enumerate(runs):
        df, p, r, ctx = _ctx(path)
        plan = {"operation": "predict_class", "target_column": tgt}
        if ex:
            plan["exclude_columns"] = ex
        res = execute(validate_plan(AnalysisPlan(**plan), df, p, r), ctx)
        if lab.startswith("HR"):
            hr_shap = res["explainability"]["shap"]["items"][:8]
        y, s = _holdout_proba(ctx)
        fpr, tpr, _ = roc_curve(y, s)
        a.plot(fpr, tpr, color=[C[0], C[2], C[3], C[1]][i], lw=1.2, ls="--" if "leaky" in lab else "-", label=f"{lab} {roc_auc_score(y, s):.2f}")
    a.plot([0, 1], [0, 1], color="#bbb", lw=0.6)
    a.set_xlabel("false positive rate")
    a.set_ylabel("true positive rate")
    a.set_title("(a) Hold-out ROC (AUC)")
    a.legend(fontsize=4.8, loc="lower right", frameon=False)
    # (b) SHAP drivers of attrition, coloured by effect direction
    b = ax[0, 1]
    items = hr_shap[::-1]
    def kind(dirn):
        if dirn.startswith("higher values push the prediction up"):
            return C[3]
        if dirn.startswith("higher values push the prediction down"):
            return C[4]
        return C[0]
    b.barh([i["feature"] for i in items], [i["share_%"] for i in items], color=[kind(i["direction"]) for i in items])
    from matplotlib.patches import Patch
    b.legend(handles=[Patch(color=C[3], label="higher \u2192 more attrition"), Patch(color=C[4], label="higher \u2192 less attrition"),
                      Patch(color=C[0], label="categorical")], fontsize=4.8, frameon=False, loc="lower right")
    b.set_xlim(0, max(i["share_%"] for i in items) * 1.75)
    b.set_xlabel("share of mean |SHAP| (%)")
    b.set_title("(b) What drives attrition (IBM HR)")
    b.tick_params(axis="y", labelsize=5.5)
    # (c) forecast with 80/95% bands
    c = ax[0, 2]
    df, p, r, ctx = _ctx("backend/sample_data/real/global_superstore_sample.csv")
    fc = execute(validate_plan(AnalysisPlan(operation="forecast", target_column="Sales", time_column="Order Date", frequency="M", forecast_horizon=6, aggregation="sum"), df, p, r), ctx)
    rows = fc["charts"][0]["data"]
    xs = np.arange(len(rows))
    act = [r_.get("actual") for r_ in rows]
    f = [r_.get("forecast") for r_ in rows]
    idx = [i for i, r_ in enumerate(rows) if r_.get("lower95") is not None]
    c.fill_between(idx, [rows[i]["lower95"] / 1e3 for i in idx], [rows[i]["upper95"] / 1e3 for i in idx], color=C[1], alpha=0.15, lw=0, label="95%")
    c.fill_between(idx, [rows[i]["lower"] / 1e3 for i in idx], [rows[i]["upper"] / 1e3 for i in idx], color=C[1], alpha=0.35, lw=0, label="80%")
    c.plot(xs, [v / 1e3 if v is not None else np.nan for v in act], color=C[0], lw=1, label="actual")
    c.plot(xs, [v / 1e3 if v is not None else np.nan for v in f], color=C[1], lw=1.3, marker="o", ms=2, label=fc["evaluation"]["selected_model"].split(" (")[0])
    c.set_xticks(xs[::12])
    c.set_xticklabels([rows[i]["period"][:4] for i in range(0, len(rows), 12)], fontsize=5.5)
    c.set_ylabel("monthly sales (thousands)")
    c.set_title(f"(c) Sales forecast, 80/95% bands")
    c.legend(fontsize=4.8, frameon=False, loc="upper left")
    # (d) root-cause waterfall
    d = ax[1, 0]
    ec = execute(validate_plan(AnalysisPlan(operation="explain_change", target_column="Profit", aggregation="sum", frequency="Q", group_by=["Market"]), df, p, r), ctx)
    wf = ec["charts"][0]["data"]
    colors = [C[4] if w["kind"] in ("base", "total") else (C[2] if w["kind"] == "up" else C[3]) for w in wf]
    d.bar(range(len(wf)), [w["delta"] / 1e3 for w in wf], bottom=[w["start"] / 1e3 for w in wf], color=colors)
    d.set_xticks(range(len(wf)))
    d.set_xticklabels([w["label"] for w in wf], rotation=40, ha="right", fontsize=5.3)
    d.set_ylabel("profit (thousands)")
    d.set_title(f"(d) Why profit changed: {ec['metrics']['Change %']:+.0f}% by market")
    # (e) quality robustness
    e = ax[1, 1]
    rob = json.loads((OUT / "real_datasets.json").read_text())["robustness"]
    labels = ["original", "corrupted", "cleaned"]
    for i, (name, rr) in enumerate(rob.items()):
        e.plot(labels, [rr["quality"][k] for k in labels], marker="o", ms=3, color=[C[0], C[1]][i], label=f"{name.replace('IBM ', '')} Q")
        e.plot(labels, [rr["model"][k]["F1"] * 100 for k in labels], marker="s", ms=3, ls="--", color=[C[0], C[1]][i], label=f"{name.replace('IBM ', '')} F1x100")
    e.set_ylim(50, 122)
    e.set_title("(e) Defect injection: quality vs model F1")
    e.legend(fontsize=4.6, frameon=False, ncol=2, loc="upper center")
    # (f) planner accuracy per dataset (v2 held-out)
    g = ax[1, 2]
    v2 = json.loads((OUT / "benchmark_v2_local_heldout.json").read_text())["per_dataset"]
    kw = json.loads((OUT / "benchmark_v2_keyword.json").read_text())["per_dataset"]
    names = list(v2)
    x = np.arange(len(names))
    g.bar(x - 0.2, [v2[n]["rate"] * 100 for n in names], 0.4, color=C[0], label="UDI")
    g.bar(x + 0.2, [kw[n]["rate"] * 100 for n in names], 0.4, color="#9ca3af", label="keyword")
    g.set_xticks(x)
    g.set_xticklabels(names, rotation=45, ha="right", fontsize=5.3)
    g.set_ylabel("end-to-end correct (%)")
    g.set_ylim(0, 118)
    g.set_title("(f) Held-out v2 by dataset")
    g.legend(fontsize=5, frameon=False, loc="upper center", ncol=2)
    for axis in ax.flat:
        axis.title.set_fontsize(6.6)
        axis.tick_params(labelsize=5.5)
    fig.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.6)
    fig.savefig(OUT / "fig_gallery.pdf")
    plt.close(fig)


if __name__ == "__main__" and "--gallery" in sys.argv:
    gallery_figure()


def ops_figure():
    """Per-operation intent accuracy (pooled frozen runs: v1 run 1 + v2) and measured latency per operation."""
    import csv
    rows = list(csv.DictReader(open(OUT / "benchmark_local_heldout_run1_log.csv"))) + list(csv.DictReader(open(OUT / "benchmark_v2_local_heldout_log.csv")))
    by = {}
    for r in rows:
        if r["gold_op"] in ("unsupported",):
            continue
        d = by.setdefault(r["gold_op"], {"ok": 0, "n": 0, "sec": []})
        d["n"] += 1
        d["ok"] += r["intent_ok"] == "True"
        if r["status"] == "answer":
            d["sec"].append(float(r["seconds"]))
    ops = sorted(by, key=lambda o: (-by[o]["ok"] / by[o]["n"], o))
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.45, 2.55), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(ops))
    a.barh(y, [100 * by[o]["ok"] / by[o]["n"] for o in ops], color="#5b21b6")
    for i, o in enumerate(ops):
        a.text(101, i, f"{by[o]['ok']}/{by[o]['n']}", va="center", fontsize=4.6)
    a.set_yticks(y)
    a.set_yticklabels([o.replace("_", " ") for o in ops], fontsize=5)
    a.invert_yaxis()
    a.set_xlim(0, 122)
    a.set_xlabel("intent accuracy (%)", fontsize=5.5)
    a.set_title("(a) Routing by operation", fontsize=6.3)
    med = [float(np.median(by[o]["sec"])) if by[o]["sec"] else np.nan for o in ops]
    b.barh(y, med, color="#f59e0b")
    b.set_xscale("log")
    b.set_xlabel("median seconds (log)", fontsize=5.5)
    b.set_title("(b) Response time", fontsize=6.3)
    for axis in (a, b):
        axis.tick_params(labelsize=5)
    fig.tight_layout(pad=0.3, w_pad=0.4)
    fig.savefig(OUT / "fig_ops.pdf")
    plt.close(fig)
    return {o: {"n": by[o]["n"], "ok": by[o]["ok"], "median_s": med[i]} for i, o in enumerate(ops)}


if __name__ == "__main__" and "--ops" in sys.argv:
    print(json.dumps(ops_figure(), indent=0))
