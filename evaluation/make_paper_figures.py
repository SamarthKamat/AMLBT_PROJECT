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
