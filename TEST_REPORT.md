# UDI 2.0 — executed test report

This report records tests that were **executed**, not only defined.

| Item | Value |
|---|---|
| Date (UTC) | 2026-11-10 |
| Environment | Linux, Python 3.13.18, packages exactly as pinned in `backend/requirements.txt` (venv); Node 22, npm 10 |
| Command | `cd backend && python -m pytest -q --junitxml=../evaluation/results/pytest_junit.xml` |
| Result | **168 passed, 0 failed, 0 errors, 0 skipped** in 265.9 s |
| Machine-readable log | `evaluation/results/pytest_junit.xml` |

## Executed tests per module
| Module | Executed | What it covers |
|---|---:|---|
| `test_loading_profiling.py` | 22 | CSV/XLSX loading, delimiters, encodings, header detection, type coercion, role inference |
| `test_validation.py` | 17 | invalid columns rejected (never substituted), types, filters, sufficiency, clarification |
| `test_operations.py` | 21 | the deterministic operations (summary, current state, groups, trend, correlation, anomalies) |
| `test_ml.py` | 12 | dynamic model selection, metrics, caching, leakage/weak-model warnings, scenarios |
| `test_forecast.py` | 8 | rolling-origin backtest, baselines, horizon limits, frequencies |
| `test_reference_models.py` | 12 | Phase 1 reference models reproduce stored metrics |
| `test_explainer.py` | 3 | Gemini wording rejected if it contains numbers not in the verified result |
| `test_gemini_client.py` | 23 | Gemini error mapping, strict JSON, schema, **source scan for exec/eval/subprocess/`__import__`**, no hard-coded key/model |
| `test_api_e2e.py` | 24 | API end to end: **all 18 operations routed on one dataset**, follow-ups, clarification, errors, Gemini-only mode, **auto mode falling back to the local planner**, new chat, sessions |
| `test_phase2.py` | 26 | semantics, relationships (hierarchy + formula detection), data quality + cleaning, **SHAP exactness on a linear model**, one-standard-error rule, ETS/ARIMA + 95% band, no target substitution in forecasting, root-cause decomposition adds up, pivot/query/segment/EDA, reports + PDF/Excel export, local planner routing (10 questions), Phase 2 endpoints, **persistence across restart**, **auth/ownership**, rate limiting |

## Phase 1 vs Phase 2 reconciliation
Phase 1's paper stated "112 test functions defined"; its test report recorded 12 executed tests. Executing the Phase 1 suite here gave **141 passing test cases** (112 functions, several parametrised). Phase 2 adds 27 cases (26 in `test_phase2.py`, 1 in `test_api_e2e.py`) for **168 executed, all passing**.

## Other verification
| Check | Result |
|---|---|
| Frontend production build (`npm run build`) | passed (Vite 5.4.11, 2,390 modules) — Phase 1 could not certify this |
| Browser end-to-end (Playwright, headless Chromium) | upload of a bundled dataset, 8 chat questions, Relationships, EDA, Data quality (cleaning + before/after), Reports (sales report) and History tabs all rendered; the only console error was the intended 422 for a missing column |
| Held-out NL benchmark (local planner) | `evaluation/results/benchmark_local_heldout_run1.json` (run 1, 88.5% end-to-end) and `benchmark_local.json` (run 2, 96.2%), with per-question pass/fail logs |
| Gemini-mode benchmark | provided (`python evaluation/run_benchmark.py --planner gemini`) but **not executed** in this environment (no API key) |
| `run_windows.bat` | not executed here (Linux environment); `run_unix.sh` steps were executed individually |
