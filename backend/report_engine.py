"""Automated business reports (Phase 2): the generate_report operation and PDF / Excel / JSON export.

The report content is produced by the agents (agents.build_report) from verified tool results. This module only
renders it. Charts in the PDF are drawn with matplotlib from the same chart specs the web UI renders.
"""
from __future__ import annotations
import io
import json
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from openpyxl import Workbook  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

from agents import build_report  # noqa: E402
from context import ExecContext  # noqa: E402
from result_formatter import fmt_num, make_result  # noqa: E402
from validator import ValidatedPlan  # noqa: E402

PALETTE = ["#6d28d9", "#f59e0b", "#10b981", "#ef4444", "#3b82f6", "#ec4899", "#14b8a6", "#8b5cf6"]


def generate_report(vp: ValidatedPlan, ctx: ExecContext) -> Dict[str, Any]:
    fn = vp.plan.business_function or "general"
    rep = build_report(vp.data, ctx, fn, ctx.profile.get("dataset_name", "dataset"), target=vp.plan.target_column)
    charts = [c for s in rep["sections"] for c in s["charts"]][:6]
    summary = " ".join(rep["executive_summary"][:4])
    metrics = {c["label"]: c["value"] for c in rep["kpis"][:8]}
    return make_result("generate_report", summary, title=rep["title"], metrics=metrics, charts=charts,
                       table=[{"priority": r.get("priority"), "finding": r["finding"], "suggested action": r["action"], "evidence": r["evidence"]} for r in rep["recommendations"]],
                       table_title="Findings and suggested actions", insights=rep["executive_summary"], report=rep,
                       limitations=rep["limitations"], assumptions=list(vp.assumptions))


# ----------------------------------------------------------------------------- chart rendering
def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def chart_png(chart: Dict[str, Any], width: float = 6.4, height: float = 3.0) -> bytes:
    fig, ax = plt.subplots(figsize=(width, height), dpi=130)
    t = chart.get("type")
    data = chart.get("data") or []
    try:
        if t == "bar":
            ys = chart["y"] if isinstance(chart.get("y"), list) else [chart.get("y")]
            labels = [str(r.get(chart["x"]))[:22] for r in data][:15]
            if len(ys) == 1:
                vals = [_num(r.get(ys[0])) or 0 for r in data][:15]
                if chart.get("horizontal"):
                    ax.barh(labels[::-1], vals[::-1], color=PALETTE[0])
                else:
                    ax.bar(labels, vals, color=PALETTE[0])
                    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            else:
                import numpy as np
                x = np.arange(len(labels))
                bottom = np.zeros(len(labels))
                for i, y in enumerate(ys[:8]):
                    vals = np.array([_num(r.get(y)) or 0 for r in data][:15])
                    if chart.get("stacked"):
                        ax.bar(x, vals, bottom=bottom, label=str(y)[:18], color=PALETTE[i % 8])
                        bottom += vals
                    else:
                        ax.bar(x + i * 0.8 / len(ys), vals, width=0.8 / len(ys), label=str(y)[:18], color=PALETTE[i % 8])
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=30, ha="right")
                ax.legend(fontsize=6)
        elif t in ("line", "area"):
            ys = chart["y"] if isinstance(chart.get("y"), list) else [chart.get("y")]
            xs = [str(r.get(chart["x"])) for r in data]
            for i, y in enumerate(ys[:6]):
                ax.plot(range(len(xs)), [_num(r.get(y)) for r in data], label=str(y)[:20], color=PALETTE[i % 8], linewidth=1.6)
            step = max(1, len(xs) // 8)
            ax.set_xticks(range(0, len(xs), step))
            ax.set_xticklabels(xs[::step], rotation=30, ha="right", fontsize=6)
            if len(ys) > 1:
                ax.legend(fontsize=6)
        elif t == "forecast":
            xs = [str(r.get("period")) for r in data]
            ax.plot(range(len(xs)), [_num(r.get("actual")) for r in data], color=PALETTE[0], label="actual")
            ax.plot(range(len(xs)), [_num(r.get("forecast")) for r in data], color=PALETTE[1], label="forecast")
            lo = [_num(r.get("lower")) for r in data]
            hi = [_num(r.get("upper")) for r in data]
            idx = [i for i, (a, b) in enumerate(zip(lo, hi)) if a is not None and b is not None]
            if idx:
                ax.fill_between(idx, [lo[i] for i in idx], [hi[i] for i in idx], color=PALETTE[1], alpha=0.2, label="80% range")
            step = max(1, len(xs) // 8)
            ax.set_xticks(range(0, len(xs), step))
            ax.set_xticklabels(xs[::step], rotation=30, ha="right", fontsize=6)
            ax.legend(fontsize=6)
        elif t == "pie":
            vals = [_num(r.get(chart["y"])) or 0 for r in data][:8]
            ax.pie(vals, labels=[str(r.get(chart["x"]))[:18] for r in data][:8], autopct="%1.0f%%", colors=PALETTE, textprops={"fontsize": 6})
            ax.axis("equal")
        elif t == "scatter":
            g = chart.get("group")
            if g:
                for i, key in enumerate(sorted({r[g] for r in data})):
                    pts = [r for r in data if r[g] == key]
                    ax.scatter([r[chart["x"]] for r in pts], [r[chart["y"]] for r in pts], s=6, color=PALETTE[i % 8], label=str(key))
                ax.legend(fontsize=6)
            else:
                hl = chart.get("highlight")
                ax.scatter([r.get(chart["x"]) for r in data if not (hl and r.get(hl))], [r.get(chart["y"]) for r in data if not (hl and r.get(hl))], s=6, color=PALETTE[0])
                if hl:
                    ax.scatter([r.get(chart["x"]) for r in data if r.get(hl)], [r.get(chart["y"]) for r in data if r.get(hl)], s=12, color=PALETTE[3])
        elif t == "heatmap":
            import numpy as np
            M = np.array([[(_num(v) if v is not None else np.nan) for v in row] for row in chart["matrix"]], dtype=float)
            im = ax.imshow(M, cmap="viridis" if chart.get("absolute") else "RdBu_r", vmin=None if chart.get("absolute") else -1, vmax=None if chart.get("absolute") else 1, aspect="auto")
            cols = chart["columns"]
            rows = chart.get("rows") or cols
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels([str(c)[:12] for c in cols], rotation=60, ha="right", fontsize=5)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([str(c)[:14] for c in rows], fontsize=5)
            fig.colorbar(im, ax=ax, fraction=0.03)
        elif t == "box":
            stats_ = [{"label": b["name"][:12], "q1": b["q1"], "med": b["median"], "q3": b["q3"], "whislo": b["min"], "whishi": b["max"], "fliers": []} for b in data[:8]]
            ax.bxp(stats_, showfliers=False)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=6)
        elif t == "waterfall":
            labels = [str(r["label"])[:18] for r in data]
            cols = [PALETTE[4] if r["kind"] in ("base", "total") else (PALETTE[2] if r["kind"] == "up" else PALETTE[3]) for r in data]
            ax.bar(range(len(data)), [r["delta"] for r in data], bottom=[r["start"] for r in data], color=cols)
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6)
        else:
            ax.text(0.5, 0.5, "chart not available", ha="center")
    except Exception as exc:                    # a bad chart must not break the report
        ax.clear()
        ax.text(0.5, 0.5, f"chart could not be drawn ({type(exc).__name__})", ha="center", fontsize=7)
    ax.set_title(str(chart.get("title", ""))[:90], fontsize=8)
    ax.tick_params(labelsize=6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


# ----------------------------------------------------------------------------- PDF
def _p(text: Any, style) -> Paragraph:
    t = str(text if text is not None else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(t, style)


def _table(rows: List[Dict[str, Any]], style, max_cols: int = 6, max_rows: int = 12):
    if not rows:
        return None
    keys = list(rows[0].keys())[:max_cols]
    data = [[_p(k.replace("_", " "), style) for k in keys]] + [[_p(fmt_num(r.get(k)) if isinstance(r.get(k), (int, float)) else r.get(k), style) for k in keys] for r in rows[:max_rows]]
    tb = Table(data, repeatRows=1, colWidths=[(17.5 * cm) / len(keys)] * len(keys))
    tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ede9fe")), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c4b5fd")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("FONTSIZE", (0, 0), (-1, -1), 7)]))
    return tb


def report_pdf(rep: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            title=rep["title"], author="UDI 2.0")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=17, textColor=colors.HexColor("#4c1d95"))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, textColor=colors.HexColor("#5b21b6"), spaceBefore=8)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.5, leading=11)
    small = ParagraphStyle("s", parent=body, fontSize=7, leading=9, textColor=colors.HexColor("#555555"))
    cell = ParagraphStyle("c", parent=body, fontSize=7, leading=8.5)
    el: List[Any] = [_p(rep["title"], h1), _p(f"Generated {rep['generated_at']} by UDI 2.0 · {rep['dataset']['rows']:,} rows × {rep['dataset']['columns']} columns · "
                                              f"data quality {rep['data_quality']['score']:.0f}/100 (grade {rep['data_quality']['grade']})", small), Spacer(1, 6)]
    el.append(_p("Executive summary", h2))
    for b in rep["executive_summary"]:
        el.append(_p("• " + b, body))
    el.append(_p("Key performance indicators", h2))
    kp = [[_p(c["label"], cell) for c in rep["kpis"][:6]], [_p(f"{fmt_num(c['value'])}{c.get('unit', '')}", ParagraphStyle('k', parent=body, fontSize=11, textColor=colors.HexColor('#4c1d95'))) for c in rep["kpis"][:6]]]
    t = Table(kp, colWidths=[(17.5 * cm) / max(1, len(kp[0]))] * len(kp[0]))
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#c4b5fd")), ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#ddd6fe")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f3ff"))]))
    el.append(t)
    el.append(_p("Findings and suggested actions", h2))
    rec_rows = [{"priority": r.get("priority"), "finding": r["finding"], "suggested action": r["action"], "evidence / estimate": r["evidence"] + (" " + r["impact_estimate"] if r.get("impact_estimate") else "")}
                for r in rep["recommendations"]]
    tb = _table(rec_rows, cell, 4, 12)
    if tb:
        el.append(tb)
    for s in rep["sections"]:
        el.append(_p(s["title"], h2))
        el.append(_p(f"[{s['agent']}] " + s["summary"], body))
        for ins in s.get("insights") or []:
            el.append(_p("– " + ins, small))
        for c in s["charts"][:2]:
            el.append(Image(io.BytesIO(chart_png(c)), width=16 * cm, height=7.5 * cm))
        tb = _table(s.get("table") or [], cell, 6, 8)
        if tb:
            el.append(tb)
    el.append(_p("Data quality", h2))
    el.append(_p(rep["data_quality"]["summary"], body))
    tb = _table(rep["data_quality"]["top_issues"], cell, 3, 6)
    if tb:
        el.append(tb)
    el.append(_p("How columns relate", h2))
    for f in rep["relationships"]:
        el.append(_p("• " + f, small))
    el.append(_p("Method and limitations", h2))
    for l in rep["limitations"]:
        el.append(_p("• " + l, small))
    el.append(_p("Agent trace", h2))
    tb = _table([{"agent": st["agent"], "tool": st["tool"], "status": st["status"], "seconds": st["seconds"], "result": st["headline"]} for st in rep["agent_trace"]], cell, 5, 40)
    if tb:
        el.append(tb)
    doc.build(el)
    return buf.getvalue()


# ----------------------------------------------------------------------------- Excel
def _sheet(wb: Workbook, title: str, rows: List[Dict[str, Any]]):
    ws = wb.create_sheet(title[:31])
    if not rows:
        ws.append(["(no rows)"])
        return
    keys = list(dict.fromkeys(k for r in rows for k in r.keys()))
    ws.append([k.replace("_", " ") for k in keys])
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="5B21B6")
    for r in rows:
        ws.append([v if isinstance(v, (int, float, str)) or v is None else json.dumps(v, default=str) for v in (r.get(k) for k in keys)])
    for i, k in enumerate(keys, 1):
        ws.column_dimensions[get_column_letter(i)].width = min(60, max(10, len(str(k)) + 2, *[min(60, len(str(r.get(k, "")))) for r in rows[:50]]))
        for cell in ws[get_column_letter(i)][1:]:
            cell.alignment = Alignment(wrap_text=True, vertical="top")


def report_xlsx(rep: Dict[str, Any]) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append([rep["title"]])
    ws["A1"].font = Font(bold=True, size=14, color="4C1D95")
    ws.append([f"Generated {rep['generated_at']} · {rep['dataset']['rows']} rows × {rep['dataset']['columns']} columns"])
    ws.append([])
    ws.append(["Executive summary"])
    ws["A4"].font = Font(bold=True)
    for b in rep["executive_summary"]:
        ws.append([b])
    ws.column_dimensions["A"].width = 120
    _sheet(wb, "KPIs", [{"kpi": c["label"], "value": c["value"], "note": c.get("note")} for c in rep["kpis"]])
    _sheet(wb, "Recommendations", rep["recommendations"])
    for i, s in enumerate(rep["sections"], 1):
        _sheet(wb, f"{i:02d} {s['title']}"[:31].replace("/", "-").replace(":", "-").replace("?", "").replace("*", "").replace("[", "(").replace("]", ")"), s.get("table") or [])
    _sheet(wb, "Data quality", rep["data_quality"]["top_issues"])
    _sheet(wb, "Agent trace", [{k: v for k, v in st.items() if k != "plan"} for st in rep["agent_trace"]])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
