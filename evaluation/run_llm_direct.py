"""Comparison: a plain LLM answering benchmark questions DIRECTLY from the data (no planner, no validation, no Python).

    python evaluation/run_llm_direct.py --questions benchmark_questions_v2.jsonl      # needs GEMINI_API_KEY / GEMINI_MODEL

For every question with a closed-form oracle (the same oracles as run_benchmark.py) the CSV is placed in the prompt
(truncated to --max-chars) and the model is asked for the answer only. The answer is scored against the pandas oracle;
refusal questions are scored on whether the model declines. Not executed for the paper (no API key in the build
environment); provided so the comparison can be reproduced.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "evaluation"))

from run_benchmark import DATASETS, oracle, wilson  # noqa: E402

PROMPT = ("You are given a CSV table. Answer the question using only this table. If the table cannot answer it, reply exactly "
          "CANNOT_ANSWER. Otherwise reply with only the answer: a single number, or a single category name.\n\nCSV:\n{csv}\n\nQuestion: {q}\nAnswer:")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="benchmark_questions_v2.jsonl")
    ap.add_argument("--max-chars", type=int, default=400000)
    args = ap.parse_args()
    import pandas as pd
    from data_profiler import load_dataset
    from gemini_client import GeminiService
    g = GeminiService()
    items = [json.loads(l) for l in open(ROOT / "evaluation" / args.questions)]
    rows, frames = [], {}
    for it in items:
        ds = it["dataset"]
        if ds not in frames:
            frames[ds] = load_dataset(ROOT / DATASETS[ds])
        exp = None if it["error"] else oracle(it, frames[ds])
        if not exp and not it["error"]:
            continue
        csv = (ROOT / DATASETS[ds]).read_text(encoding="utf-8", errors="ignore")[: args.max_chars]
        ans = g._generate("Answer data questions precisely.", PROMPT.format(csv=csv, q=it["q"]), json_output=False, temperature=0.0, max_tokens=64).strip()
        if it["error"]:
            ok = "CANNOT_ANSWER" in ans.upper()
        elif exp[0] == "group":
            ok = ans.strip().strip(".").lower() == exp[1].lower()
        else:
            m = re.search(r"-?\d[\d,]*\.?\d*", ans)
            ok = bool(m) and abs(float(m.group(0).replace(",", "")) - exp[1]) <= 0.01 * max(1.0, abs(exp[1]))
        rows.append({"id": it["id"], "question": it["q"], "expected": exp[1] if exp else "refuse", "answer": ans, "ok": ok})
        print("OK " if ok else "BAD", it["id"], it["q"][:60], "->", ans[:40])
    k = sum(r["ok"] for r in rows)
    summary = {"system": f"plain LLM ({g.model})", "questions": len(rows), "correct": k, "rate": k / max(1, len(rows)), "ci95": wilson(k, len(rows))}
    (ROOT / "evaluation" / "results" / "llm_direct.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
