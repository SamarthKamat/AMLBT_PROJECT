# UDI 2.0 — Universal Dataset Intelligence
### An AI-powered autonomous data analyst and business decision-intelligence platform

Upload a CSV / Excel file (or load one of the bundled public datasets) and ask questions in plain language:

> *Which country has the highest sales?* · *Why did profit drop last quarter?* · *What predicts attrition?* · *Are there outliers in shipping cost?* · *Total sales by market and category* · *Forecast demand for the next 6 months* · *Generate an HR report*

UDI plans each question, **validates the plan against the real columns**, runs one **controlled Python operation**, explains models with **SHAP**, turns verified numbers into **business findings and suggested actions**, and shows every answer with charts you can switch (bar / pie / line / table) and export.

**Stack:** React 18 + Vite + Recharts · Flask / Python · pandas, scikit-learn, XGBoost, statsmodels, SciPy · ReportLab + Matplotlib (PDF) + openpyxl (Excel) · SQLite persistence · Google Gemini *(optional)* or the built-in **local intent model** for planning.

---

## 1. What UDI 2.0 does

| Capability | Where | How |
|---|---|---|
| **Answer any question on the data** | *Ask* tab | 18 controlled operations (table below), chosen by a planner, verified by a validation gate. |
| **Column-wise relationships** | *Relationships* tab, "how are the columns related?" | Mixed-type association matrix (Pearson/Spearman, correlation ratio η, bias-corrected Cramér's V), hierarchies (City → State → Country), derived formulas (Profit = Sales − Cost), date rules, keys, data dictionary with semantic types. |
| **Outlier detection** | "any outliers in X?" | IQR fences + z-score + Isolation Forest (≥ 2 of 3 agree), or multivariate Isolation Forest with per-record drivers. |
| **EDA** | *EDA* tab, "perform EDA" | Distributions, skew, box plots, category mixes, missingness, association heatmap, KPI over time, auto-written insights; or a focused deep dive on one column. |
| **Data quality + cleaning** | *Data quality* tab | 0–100 score over five dimensions, issues with recommended treatments, selectable auto-cleaning, before/after comparison, revert, download. |
| **Business reports** (sales, marketing, HR, operations, finance, customer, general) | *Reports* tab, "generate an HR report" | Multi-agent pipeline → KPIs, segment comparisons, margins, trend, root cause, driver model + SHAP, segments, anomalies, forecast, findings with evidence and illustrative impact; **PDF / Excel / JSON** export. |
| **Visualisations** | every answer | bar, grouped/stacked bar, line, area, pie, scatter, forecast bands (80/95%), heatmap, box plot, waterfall (SHAP and root cause); switch view, CSV export. |
| **Predictions & what-if** | "predict X", "what if Y rises 10%?" | AutoML: 5 candidates, CV mean ± std, one-standard-error selection, tuning, learning curve, time-ordered validation, leakage screening, SHAP global + per-prediction waterfall. |
| **Forecasting** | "forecast X for the next 6 months" | Naive / seasonal naive / moving average / trend+season / ridge-AR / RF / GBM / Holt-Winters ETS / ARIMA-SARIMA, rolling-origin backtest, horizon-dependent 80 % and 95 % ranges, skill vs baseline. |
| **Root cause** | "why did profit fall?" | Exact decomposition of the change across groups (mix vs rate effects for averages), waterfall chart, most concentrated dimension. |
| **Segmentation** | "segment the customers" | K-means, k chosen by silhouette, readable segment profiles, PCA map. |

### The 18 operations
`get_summary` · `get_current_state` · `compare_groups` · `analyze_trend` · `detect_anomalies` · `calculate_correlation` · `predict_value` · `predict_class` · `forecast` · `run_scenario` (Phase 1) and `query_records` · `pivot_table` · `run_eda` · `assess_data_quality` · `describe_relationships` · `explain_change` · `segment_records` · `generate_report` (Phase 2). Nothing outside this list can run: the planner chooses a name, Python executes it. There is no `exec`/`eval` (a test scans the source).

## 2. Architecture (who does what)

```text
question ─► Planner agent ─► Validation gate ─► Analyst / ML agent ─► Business Analyst agent ─► Narrator ─► UI
            Gemini (if set)    columns exist?      one of 18 Python      findings + actions       Gemini wording only if
            or local intent    types fit? filters   operations           from verified numbers    every number is in the
            model (offline)    real? enough data?   (pandas/sklearn…)                              verified result
```
* **Planner.** `UDI_PLANNER=auto` (default) uses Gemini when a key is configured and the **local intent model** otherwise (or when a Gemini call fails); `gemini` = Gemini only (Phase 1 behaviour); `local` = fully offline. The local planner links words to columns and category values, masks them (`which catcol has the highest numcol`), classifies the intent with TF-IDF + logistic regression (19 classes, trained on a templated corpus in `planner_corpus.py`) and fills slots deterministically. Low confidence → clarification.
* **Validation gate.** An unknown column is **rejected with the list of available columns, never substituted**; filters must match real values; each operation has data-sufficiency checks.
* **Agents** (`agents.py`). Data Analyst (quality, relationships) · Business Analyst (KPIs, segments, margins, trends, root cause) · ML agent (drivers + SHAP, segments, anomalies, forecast) · Report agent. Agents only call validated tools; every call is logged in an **agent trace** shown in the UI and the PDF.

Details: `docs/ARCHITECTURE.md`.

## 3. Install and run

Prerequisites: **Python 3.11+**, **Node.js 18+**.

**Windows:** double-click `run_windows.bat`. **macOS / Linux:** `./run_unix.sh`. Both create `backend/.venv`, install packages, create `backend/.env`, train the local planner, build the frontend and start the server. Open **http://localhost:5000**.

Gemini is optional. To use it, put `GEMINI_API_KEY` and `GEMINI_MODEL` in `backend/.env`; without them UDI plans with its local model (shown in the header chip).

Manual equivalent:
```bash
cd backend
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip
cp .env.example .env                              # optional: add GEMINI_API_KEY / GEMINI_MODEL
.venv/bin/python train_planner.py
cd ../frontend && npm install && npm run build
cd ../backend && .venv/bin/python serve.py        # http://localhost:5000
```
**Docker:** `docker compose up --build` → http://localhost:5000 (data persisted in the `udi-data` volume).
**Development:** `python app.py` in `backend` and `npm run dev` in `frontend` (http://localhost:5173, `/api` proxied).

## 4. Configuration (`backend/.env`)
| Variable | Meaning |
|---|---|
| `UDI_PLANNER` | `auto` (default) · `gemini` · `local` |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Optional Gemini planning / wording. No default model is hard-coded. |
| `UDI_AUTH` | `on` = users register / sign in (PBKDF2 passwords, hashed bearer tokens, first user is admin); sessions are private to their owner. Default `off` (single local user). |
| `UDI_PERSIST_SESSIONS`, `UDI_DATA_DIR` | Datasets and conversations are stored (SQLite + snapshots) and restored after a restart. |
| `UDI_SESSION_TTL_SECONDS` | Inactive sessions — including stored files — are deleted after this time (default 6 h). |
| `UDI_RATE_CHAT`, `UDI_RATE_UPLOAD`, `UDI_RATE_REPORT` | Requests per minute per user. |
| `UDI_MAX_UPLOAD_MB`, `FLASK_PORT`, `UDI_CORS_ORIGINS` | Upload limit, port, CORS. |

## 5. Bundled datasets
* **Real, public** (`backend/sample_data/real/`): Global Superstore 1,000-order extract (sales/operations/finance), IBM HR Analytics Employee Attrition (HR), IBM Telco Customer Churn (marketing/customer), UCI Bank Marketing *bank-additional-full* (marketing). Sources in `backend/sample_data/real/SOURCES.md`.
* **Generated reference data** (`backend/sample_data/*.csv`): seven 900-row domain datasets from `generate_samples.py` (seed 42). They are synthetic and are reported separately from the real datasets.

## 6. Tests and evaluation
```bash
cd backend
python -m pytest -q                                    # 168 tests (offline; Gemini replaced by a test double)
cd ..
python evaluation/run_benchmark.py --planner local     # held-out NL benchmark, 104 questions, 6 datasets
python evaluation/run_benchmark.py --planner gemini    # same benchmark with Gemini (needs a key)
python evaluation/run_real_datasets.py                 # real datasets, robustness, forecasting, reference domains
```
Executed results, pass/fail logs and figures are in `evaluation/results/`; the executed test log is `TEST_REPORT.md`.

## 7. API (selection)
`POST /api/upload` · `POST /api/samples/load` · `POST /api/chat` · `GET /api/session/<id>/dictionary` · `GET …/eda` · `GET …/quality` · `POST …/clean` · `POST …/revert` · `GET …/download` · `POST …/report` → `GET /api/jobs/<id>` → `GET /api/jobs/<id>/download?format=pdf|xlsx|json` · `GET …/history` · `DELETE /api/session/<id>` · `POST /api/auth/register|login|logout` · `GET /api/metrics` (admin) · `GET /api/health`.
Errors are structured `{code, error, retryable}`: `INVALID_COLUMN`, `INSUFFICIENT_DATA`, `UNSUPPORTED_OPERATION`, `INVALID_PLAN`, `GEMINI_*`, `RATE_LIMITED`, `UNAUTHORIZED`, `FORBIDDEN`, …

## 8. Security and governance
* Only the schema, column statistics and at most `GEMINI_SAMPLE_ROWS` sample rows are sent to Gemini; **never the dataset**. In `local` mode nothing leaves the machine.
* Gemini output is parsed as data (pydantic). Gemini wording is shown only if every number in it appears in the verified result.
* Raw uploads are deleted after parsing; stored snapshots are removed when a session is deleted or expires. `.env` is git-ignored.

## 9. Limitations
* One flat table per session; no joins. Planning quality depends on column naming; ambiguous questions trigger clarification.
* All analyses are descriptive or predictive, **not causal**: root-cause decompositions show *where* a change happened, SHAP shows what a model relies on, and impact estimates are illustrative arithmetic.
* Trained models are cached in memory (retrained after a restart). Run one server process with several threads.
* The local planner was evaluated on 104 held-out questions written by the authors; it is not a general-purpose language model.

## 10. Repository map
`backend/` API and engines (`app.py`, `planner.py`, `local_planner.py`, `validator.py`, `*_engine.py`, `agents.py`, `report_engine.py`, `data_quality.py`, `relationships.py`, `semantics.py`, `explainability_engine.py`, `storage.py`, `ops.py`) · `frontend/` React app · `evaluation/` benchmark and experiments · `report/` final IEEE paper (LaTeX + PDF) · `docs/` architecture · `MODEL_REPORT.md` reference models · `TEST_REPORT.md` executed tests.
