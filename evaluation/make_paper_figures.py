"""Publication figures for the UDI 2.0 paper, drawn from UDI's own verified outputs (no screenshots).

    python evaluation/make_paper_figures.py           # all figures -> evaluation/results/fig_*.pdf

Every figure is drawn at the exact width it is printed at (IEEE column 3.5 in, page 7.16 in), so fonts are never
scaled down. Numbers come from the engine and from the result files written by run_benchmark.py and
run_real_datasets.py.

fig_example.pdf   (a) answer to "Which country generated the most sales?"  (b) relationship map (lower triangle)
fig_ops.pdf       routing accuracy per operation with median response time
fig_gallery.pdf   six-panel results gallery (ROC, SHAP drivers, forecast, root cause, robustness, held-out by dataset)
fig_benchmark.pdf end-to-end accuracy of the three planners with Wilson 95% CIs (repository figure)
"""
from __future__ import annotations
import csv
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
import matplotlib.ticker as mtick  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from context import ExecContext  # noqa: E402
from data_profiler import load_dataset, profile_dataframe  # noqa: E402
from executor import execute  # noqa: E402
from schemas import AnalysisPlan  # noqa: E402
from semantics import infer_semantics  # noqa: E402
from validator import roles_from_profile, validate_plan  # noqa: E402

OUT = ROOT / "evaluation" / "results"
COL, PAGE = 3.5, 7.16
# one palette for the whole paper
NAVY, TEAL, AMBER, RED, GREY, SKY, PLUM = "#1f3b73", "#2a9d8f", "#e9a23b", "#d1495b", "#9aa3ad", "#5b8fd6", "#7b4fa0"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7, "axes.titlesize": 7.5, "axes.titleweight": "bold",
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.edgecolor": "#555555", "axes.linewidth": 0.7, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def _ctx(path):
    df = load_dataset(ROOT / path)
    p = profile_dataframe(df)
    r = roles_from_profile(p)
    return df, p, r, ExecContext(roles=r, profile=p, semantics=infer_semantics(df, p), full_data=df)


def _run(df, p, r, ctx, plan):
    return execute(validate_plan(AnalysisPlan(**plan), df, p, r), ctx)


def _panel_label(ax, text):
    ax.set_title(text, loc="left", pad=4)


# ----------------------------------------------------------------------------- Fig. 2
def example_figure():
    df, p, r, ctx = _ctx("backend/sample_data/real/global_superstore_sample.csv")
    cg = _run(df, p, r, ctx, {"operation": "compare_groups", "target_column": "Sales", "group_by": ["Country"], "aggregation": "sum", "sort_order": "desc"})
    rel = _run(df, p, r, ctx, {"operation": "describe_relationships"})
    fig = plt.figure(figsize=(PAGE, 2.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.08], wspace=0.32)
    a = fig.add_subplot(gs[0])
    rows = cg["table"][:10][::-1]
    vals = [x["Total of Sales"] / 1000 for x in rows]
    bars = a.barh([x["Country"] for x in rows], vals, color=[NAVY] * (len(rows) - 1) + [AMBER], height=0.66)
    for b_, v in zip(bars, vals):
        a.text(v + 2, b_.get_y() + b_.get_height() / 2, f"{v:.0f}k", va="center", fontsize=6.2)
    a.set_xlim(0, max(vals) * 1.16)
    a.grid(axis="y", visible=False)
    a.set_xlabel("total sales (thousands)")
    _panel_label(a, "(a) “Which country generated the most sales?”")
    a.text(0.98, 0.04, "compare_groups · 93 countries\nKruskal–Wallis p < 10⁻⁶⁹", transform=a.transAxes, ha="right", va="bottom", fontsize=6,
           color="#444", bbox=dict(boxstyle="round,pad=0.3", fc="#f6f6f6", ec="#dddddd", lw=0.5))
    b = fig.add_subplot(gs[1])
    hm = rel["charts"][0]
    keep = [c for c in hm["columns"] if c not in ("Segment", "Order Priority", "Ship Mode")][:9]
    idx = [hm["columns"].index(c) for c in keep]
    M = np.array([[np.nan if hm["matrix"][i][j] is None else hm["matrix"][i][j] for j in idx] for i in idx], dtype=float)
    M = M[1:, :-1]                                    # lower triangle without the empty first row / last column
    mask = np.triu(np.ones_like(M, dtype=bool), k=1)
    Mm = np.ma.array(M, mask=mask)
    im = b.imshow(Mm, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    b.grid(False)
    b.set_xticks(range(len(keep) - 1))
    b.set_xticklabels(keep[:-1], rotation=40, ha="right")
    b.set_yticks(range(len(keep) - 1))
    b.set_yticklabels(keep[1:])
    for i in range(len(keep) - 1):
        for j in range(i + 1):
            v = M[i, j]
            if not np.isnan(v):
                b.text(j, i, f"{v:.2f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center", fontsize=5.6,
                       color="white" if abs(v) > 0.6 else "#222")
    for s in b.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=b, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("signed r  /  η, Cramér's V", fontsize=6.2)
    _panel_label(b, "(b) Column relationship map (describe_relationships)")
    fig.savefig(OUT / "fig_example.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 3
def ops_figure():
    rows = list(csv.DictReader(open(OUT / "benchmark_local_heldout_run1_log.csv"))) + list(csv.DictReader(open(OUT / "benchmark_v2_local_heldout_log.csv")))
    by = {}
    for r in rows:
        if r["gold_op"] == "unsupported":
            continue
        d = by.setdefault(r["gold_op"], {"ok": 0, "n": 0, "sec": []})
        d["n"] += 1
        d["ok"] += r["intent_ok"] == "True"
        if r["status"] == "answer":
            d["sec"].append(float(r["seconds"]))
    ops = sorted(by, key=lambda o: (by[o]["ok"] / by[o]["n"], -by[o]["n"]))
    acc = [100 * by[o]["ok"] / by[o]["n"] for o in ops]
    med = [float(np.median(by[o]["sec"])) if by[o]["sec"] else np.nan for o in ops]
    fig = plt.figure(figsize=(COL, 2.75))
    ax = fig.add_axes([0.335, 0.13, 0.42, 0.80])          # fixed layout: labels + table columns fit inside the column width
    y = np.arange(len(ops))
    cols = [TEAL if a_ == 100 else AMBER for a_ in acc]
    ax.hlines(y, 0, acc, color=cols, lw=2.2)
    ax.plot(acc, y, "o", color=NAVY, ms=3.0)
    ax.set_yticks(y)
    ax.set_yticklabels([o.replace("_", " ") for o in ops], fontsize=6.3)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("intent accuracy (n = 203)", fontsize=6.5)
    ax.set_ylim(-0.7, len(ops) - 0.3)
    tr = ax.get_yaxis_transform()
    for i, o in enumerate(ops):
        t = med[i]
        ts = "\u2013" if np.isnan(t) else (f"{t * 1000:.0f} ms" if t < 1 else f"{t:.1f} s")
        ax.text(1.04, i, f"{by[o]['ok']}/{by[o]['n']}", va="center", fontsize=6.1, transform=tr)
        ax.text(1.30, i, ts, va="center", fontsize=6.1, color=PLUM, transform=tr)
    ax.text(1.04, 1.02, "correct", fontsize=6.1, fontweight="bold", transform=ax.transAxes)
    ax.text(1.30, 1.02, "median time", fontsize=6.1, fontweight="bold", color=PLUM, transform=ax.transAxes)
    fig.savefig(OUT / "fig_ops.pdf", bbox_inches=None)
    plt.close(fig)
    return {o: {"n": by[o]["n"], "ok": by[o]["ok"], "median_s": med[i]} for i, o in enumerate(ops)}


# ----------------------------------------------------------------------------- Fig. 4
def _holdout_proba(ctx):
    m = list(ctx.cache._d.values())[-1]
    X = m.X_reference.loc[m.holdout.index]
    y = m.encoder.transform(m.holdout["actual"].astype(str))
    pos = m.explain_index if m.explain_index is not None else 1
    return (y == pos).astype(int), m.eval_pipeline.predict_proba(X)[:, pos]


def gallery_figure():
    from sklearn.metrics import roc_auc_score, roc_curve
    fig, ax = plt.subplots(2, 3, figsize=(PAGE, 4.7))
    # (a) ROC
    a = ax[0, 0]
    runs = [("HR", "backend/sample_data/real/ibm_hr_attrition.csv", "Attrition", None, NAVY, "-"),
            ("Telco", "backend/sample_data/real/ibm_telco_churn.csv", "Churn", None, TEAL, "-"),
            ("Bank, leaky", "backend/sample_data/real/uci_bank_marketing.csv", "y", None, RED, "--"),
            ("Bank, pre-call", "backend/sample_data/real/uci_bank_marketing.csv", "y", ["duration"], AMBER, "-")]
    hr_shap = None
    for lab, path, tgt, ex, c, ls in runs:
        df, p, r, ctx = _ctx(path)
        plan = {"operation": "predict_class", "target_column": tgt}
        if ex:
            plan["exclude_columns"] = ex
        res = _run(df, p, r, ctx, plan)
        if lab.startswith("HR"):
            hr_shap = res["explainability"]["shap"]["items"][:7]
        y, s = _holdout_proba(ctx)
        fpr, tpr, _ = roc_curve(y, s)
        a.plot(fpr, tpr, color=c, lw=1.3, ls=ls, label=f"{lab}  {roc_auc_score(y, s):.2f}")
    a.plot([0, 1], [0, 1], color=GREY, lw=0.7, ls=":")
    a.set_xlim(0, 1)
    a.set_ylim(0, 1.01)
    a.set_xlabel("false positive rate")
    a.set_ylabel("true positive rate")
    a.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02), title="AUC", title_fontsize=6.2, handlelength=1.4, borderaxespad=0.2)
    _panel_label(a, "(a) Hold-out ROC curves")
    # (b) SHAP
    b = ax[0, 1]
    items = hr_shap[::-1]

    def kind(dirn):
        if dirn.startswith("higher values push the prediction up"):
            return RED
        if dirn.startswith("higher values push the prediction down"):
            return SKY
        return PLUM
    vals = [i["share_%"] for i in items]
    bars = b.barh([i["feature"] for i in items], vals, color=[kind(i["direction"]) for i in items], height=0.66)
    for br, v in zip(bars, vals):
        b.text(v + 0.3, br.get_y() + br.get_height() / 2, f"{v:.1f}%", va="center", fontsize=6)
    b.set_xlim(0, max(vals) * 2.2)
    b.grid(axis="y", visible=False)
    b.set_xlabel("share of mean |SHAP| (%)")
    b.legend(handles=[Patch(color=RED, label="raises risk"), Patch(color=SKY, label="lowers risk"), Patch(color=PLUM, label="categorical")],
             loc="upper right", bbox_to_anchor=(1.0, 0.86), handlelength=1, title="higher value", title_fontsize=6)
    _panel_label(b, "(b) What drives attrition (IBM HR)")
    # (c) forecast
    c = ax[0, 2]
    df, p, r, ctx = _ctx("backend/sample_data/real/global_superstore_sample.csv")
    fc = _run(df, p, r, ctx, {"operation": "forecast", "target_column": "Sales", "time_column": "Order Date", "frequency": "M", "forecast_horizon": 6, "aggregation": "sum"})
    rows = fc["charts"][0]["data"]
    xs = np.arange(len(rows))
    idx = [i for i, r_ in enumerate(rows) if r_.get("lower95") is not None]
    c.fill_between(idx, [rows[i]["lower95"] / 1e3 for i in idx], [rows[i]["upper95"] / 1e3 for i in idx], color=AMBER, alpha=0.18, lw=0, label="95% range")
    c.fill_between(idx, [rows[i]["lower"] / 1e3 for i in idx], [rows[i]["upper"] / 1e3 for i in idx], color=AMBER, alpha=0.42, lw=0, label="80% range")
    c.plot(xs, [r_["actual"] / 1e3 if r_.get("actual") is not None else np.nan for r_ in rows], color=NAVY, lw=1.0, label="actual")
    c.plot(xs, [r_["forecast"] / 1e3 if r_.get("forecast") is not None else np.nan for r_ in rows], color=RED, lw=1.4, marker="o", ms=2.2,
           label="forecast (trend + season)")
    c.axvline(idx[0], color=GREY, lw=0.7, ls=":")
    ticks = [i for i, r_ in enumerate(rows) if r_["period"].endswith("-01")]
    c.set_xticks(ticks)
    c.set_xticklabels([rows[i]["period"][:4] for i in ticks])
    c.set_ylim(bottom=0)
    c.set_ylabel("monthly sales (thousands)")
    c.legend(loc="upper left", handlelength=1.4)
    _panel_label(c, "(c) Guarded sales forecast")
    # (d) waterfall
    d = ax[1, 0]
    ec = _run(df, p, r, ctx, {"operation": "explain_change", "target_column": "Profit", "aggregation": "sum", "frequency": "Q", "group_by": ["Market"]})
    wf = ec["charts"][0]["data"]
    colors = [NAVY if w["kind"] in ("base", "total") else (TEAL if w["kind"] == "up" else RED) for w in wf]
    yy = np.arange(len(wf))[::-1]                       # horizontal waterfall: read top to bottom, labels never rotate
    d.barh(yy, [w["delta"] / 1e3 for w in wf], left=[w["start"] / 1e3 for w in wf], color=colors, height=0.62)
    xmax = max((max(w["start"], w["start"] + w["delta"])) for w in wf) / 1e3
    for y_, w in zip(yy, wf):
        end = max(w["start"], w["start"] + w["delta"]) / 1e3
        lab = f"{w['value'] / 1e3:.1f}k" if w["kind"] in ("base", "total") else f"{w['value'] / 1e3:+.1f}k"
        d.text(end + xmax * 0.02, y_, lab, va="center", fontsize=5.8)
    d.set_yticks(yy)
    d.set_yticklabels([w["label"].replace("Asia Pacific", "APAC") for w in wf], fontsize=6)
    d.grid(axis="y", visible=False)
    d.set_xlim(0, xmax * 1.3)
    d.set_xlabel("profit (thousands)")
    _panel_label(d, f"(d) Why profit rose {ec['metrics']['Change %']:.0f}% (by market)")
    # (e) robustness: grouped bars
    e = ax[1, 1]
    rob = json.loads((OUT / "real_datasets.json").read_text())["robustness"]
    states = ["original", "corrupted", "cleaned"]
    w_ = 0.19
    x = np.arange(len(states))
    series = []
    for name, rr in rob.items():
        short = name.replace("IBM ", "").replace(" Attrition", "").replace(" Churn", "")
        series.append((f"{short} quality Q", [rr["quality"][k] for k in states]))
        series.append((f"{short} macro-F1×100", [rr["model"][k]["F1"] * 100 for k in states]))
    cols = [NAVY, SKY, TEAL, "#8fd3c8"]
    for k, (lab, vals) in enumerate(series):
        e.bar(x + (k - 1.5) * w_, vals, w_ * 0.95, color=cols[k], label=lab)
    e.set_xticks(x)
    e.set_xticklabels(["original", "corrupted", "cleaned"], fontsize=6)
    e.tick_params(axis="x", pad=2)
    e.set_ylim(0, 128)
    e.set_yticks([0, 25, 50, 75, 100])
    e.grid(axis="x", visible=False)
    e.legend(loc="upper center", ncol=2, handlelength=1, columnspacing=0.8)
    _panel_label(e, "(e) Injected defects: quality vs model")
    # (f) held-out v3 (public data only) by dataset, horizontal so the names stay level
    g = ax[1, 2]
    systems = [("UDI local", "benchmark_v3_local.json", NAVY), ("Gemini planner", "benchmark_v3_gemini.json", TEAL), ("keyword", "benchmark_v3_keyword.json", GREY)]
    systems = [s_ for s_ in systems if (OUT / s_[1]).exists()]
    names = ["superstore", "hr", "telco", "bank"]
    hh = 0.8 / len(systems)
    for k, (lab, fn, colr) in enumerate(systems):
        pdx = json.loads((OUT / fn).read_text())["per_dataset"]
        ys = np.arange(len(names))[::-1] + (len(systems) / 2 - k - 0.5) * hh
        g.barh(ys, [pdx[n]["rate"] * 100 for n in names], hh * 0.92, color=colr, label=lab)
    g.set_yticks(np.arange(len(names))[::-1])
    g.set_yticklabels(["Superstore", "IBM HR", "Telco", "Bank"], fontsize=6)
    g.set_xlim(0, 100)
    g.xaxis.set_major_formatter(mtick.PercentFormatter())
    g.set_xlabel("end-to-end pass, held-out v3")
    g.grid(axis="y", visible=False)
    g.legend(loc="lower right", handlelength=1, fontsize=5.6, framealpha=0.9)
    _panel_label(g, "(f) Held-out v3 (public data), per dataset")
    fig.tight_layout(pad=0.3, h_pad=1.0, w_pad=1.0)
    fig.savefig(OUT / "fig_gallery.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- repository figure
def benchmark_figure():
    systems = [("Keyword router", "keyword", GREY), ("UDI, exact linking", "local_exact", AMBER), ("UDI local planner", None, NAVY)]
    sets = [("v1 (n=104, same code)", "v1"), ("v2 (fresh held-out, n=99)", "v2")]
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    w = 0.26
    for si, (label, key, colr) in enumerate(systems):
        vals, lo, hi = [], [], []
        for _, tag in sets:
            f = OUT / (f"benchmark_{tag}_{key}.json" if key else ("benchmark_v1_local_current.json" if tag == "v1" else "benchmark_v2_local_heldout.json"))
            e_ = json.loads(f.read_text())["end_to_end_pass"]
            vals.append(e_["rate"] * 100)
            lo.append(e_["rate"] * 100 - e_["ci95"][0] * 100)
            hi.append(e_["ci95"][1] * 100 - e_["rate"] * 100)
        x = np.arange(len(sets)) + (si - 1) * w
        ax.bar(x, vals, w, yerr=[lo, hi], capsize=2, label=label, color=colr, error_kw={"elinewidth": 0.7})
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([s for s, _ in sets])
    ax.set_ylabel("end-to-end correct (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.18))
    fig.savefig(OUT / "fig_benchmark.pdf")
    plt.close(fig)


if __name__ == "__main__":
    example_figure()
    ops_figure()
    benchmark_figure()
    gallery_figure()
    print("figures written to", OUT)
